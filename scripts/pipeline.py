#!/usr/bin/env python3
"""Run the LegalQA pipeline locally."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.artifacts import (
    assemble_training_actions,
    build_or_load_candidate_audit,
    build_or_load_pair_features,
    build_or_load_singleton_features,
    build_or_load_top_documents,
    cache_manifest,
    load_task2,
    split_training_data,
)
from legalqa.engine import ACTION_FEATURES, LegalQAEngine, configure_cuda, ensure_nltk_resources, train_ranker_ensemble
from legalqa.settings import PipelineSettings
from legalqa.text_utils import load_stopwords


def json_dump(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def pickle_dump(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def runtime_info(device: str) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
    }
    modules = (
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "sentence_transformers",
        "xgboost",
        "nltk",
    )
    for module in modules:
        try:
            loaded = __import__(module)
            info[module] = getattr(loaded, "__version__", "unknown")
        except Exception as exc:
            info[module] = f"unavailable: {exc!r}"
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_runtime"] = torch.version.cuda
            info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
            info["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    return info


def validate_base_cache_files(cfg: PipelineSettings, cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    child_path = next(
        (
            cache_dir / name
            for name in cfg.child_cache_names
            if (cache_dir / name).is_file() and (cache_dir / name).stat().st_size > 0
        ),
        None,
    )
    required = {
        "bm25": cache_dir / cfg.bm25_cache_name,
        "dense_parent": cache_dir / cfg.dense_parent_cache_name,
        "child": child_path,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError("Thiếu base cache bắt buộc:\n- " + "\n- ".join(missing))
    return {name: str(path) for name, path in required.items()}


def prepare_features(cfg, engine, train, split, cache_dir, checkpoint_every):
    base1000, holdout1000, training5000 = split
    print("\n=== PREPARE FEATURE CACHES ===", flush=True)

    top_documents, top_state = build_or_load_top_documents(
        cfg,
        engine,
        train,
        base1000,
        cache_dir,
        checkpoint_every,
    )
    audit, audit_state = build_or_load_candidate_audit(
        cfg,
        engine,
        train,
        base1000,
        cache_dir,
        top_documents,
        checkpoint_every,
    )
    pair_cache, pair_state = build_or_load_pair_features(
        cfg,
        engine,
        train,
        base1000,
        holdout1000,
        training5000,
        cache_dir,
        checkpoint_every,
    )
    singleton_cache, singleton_state, score_diffs = build_or_load_singleton_features(
        cfg,
        engine,
        train,
        holdout1000,
        training5000,
        cache_dir,
        pair_cache,
        checkpoint_every,
    )
    provenance = {
        cfg.top_documents_cache_name: top_state,
        cfg.candidate_audit_cache_name: audit_state,
        cfg.pair_features_cache_name: pair_state,
        cfg.singleton_features_cache_name: singleton_state,
        "cached_score_differences": score_diffs,
    }
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    return audit, pair_cache, singleton_cache, provenance


def train_answer_rankers(cfg, actions, output_dir: Path):
    print("\n=== TRAIN FIVE ANSWER RANKERS ===", flush=True)
    started = time.time()
    models = train_ranker_ensemble(cfg, actions, ACTION_FEATURES)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed, model in zip(cfg.model_seeds, models):
        model.save_model(model_dir / f"answer_ranker_seed_{seed}.json")
    print(f"Five rankers trained in {(time.time() - started) / 60:.1f}m", flush=True)
    return models


def run_public_inference(
    cfg,
    engine,
    models,
    public,
    output_dir: Path,
    limit: int = 0,
    checkpoint_every: int = 25,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    full_run = not limit or limit >= len(public)
    selected_public = list(public.items()) if full_run else list(public.items())[:limit]

    progress_path = output_dir / (
        "inference_progress.pkl" if full_run else "inference_smoke_progress.pkl"
    )
    submission_path = output_dir / (
        "submission.json" if full_run else f"submission_smoke_{len(selected_public)}.json"
    )
    log_path = output_dir / (
        "inference_log.json" if full_run else f"inference_log_smoke_{len(selected_public)}.json"
    )

    predictions, inference_log = {}, []
    if progress_path.exists():
        with progress_path.open("rb") as handle:
            progress = pickle.load(handle)
        if progress.get("version") == "legalqa_inference_progress_v1":
            predictions = {str(key): value for key, value in progress.get("predictions", {}).items()}
            inference_log = progress.get("inference_log", [])
            print(
                f"Loaded inference progress: {len(predictions)}/{len(selected_public)}",
                flush=True,
            )

    allowed = {str(qid) for qid, _ in selected_public}
    predictions = {key: value for key, value in predictions.items() if key in allowed}
    pending = [
        (str(qid), item)
        for qid, item in selected_public
        if str(qid) not in predictions
    ]

    def save_progress() -> None:
        pickle_dump(
            {
                "version": "legalqa_inference_progress_v1",
                "predictions": predictions,
                "inference_log": inference_log,
            },
            progress_path,
        )

    print(f"PUBLIC todo={len(pending)}", flush=True)
    started = time.time()
    for index, (qid, item) in enumerate(pending, start=1):
        item_started = time.time()
        try:
            prediction, debug = engine.predict(
                item["question"],
                models,
                return_debug=True,
            )
            error = None
        except Exception as exc:
            prediction, debug, error = "", {}, repr(exc)
        predictions[qid] = {"answer": prediction}
        inference_log.append(
            {
                "qid": qid,
                "pred_len": len(prediction),
                "error": error,
                "time": time.time() - item_started,
                **debug,
            }
        )

        if index % checkpoint_every == 0 or index == len(pending):
            save_progress()
            elapsed = time.time() - started
            eta = elapsed / max(index, 1) * (len(pending) - index)
            print(
                f"[{index}/{len(pending)} new] total={len(predictions)}/{len(selected_public)} "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    expected = len(selected_public)
    if len(predictions) != expected:
        raise RuntimeError(f"Prediction count {len(predictions)} != expected {expected}")

    json_dump(predictions, submission_path)
    json_dump(inference_log, log_path)
    empty_count = sum(not row["answer"].strip() for row in predictions.values())
    error_count = sum(
        row.get("error") is not None
        for row in inference_log
        if row.get("qid") in allowed
    )
    if empty_count or error_count:
        raise RuntimeError(
            f"Output guard failed: rows={expected}, empty={empty_count}, errors={error_count}; "
            f"inspect {log_path}"
        )
    print(f"READY: {submission_path}", flush=True)
    return submission_path, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalQA Task 2 pipeline")
    parser.add_argument(
        "--mode",
        choices=["check", "prepare", "infer"],
        default="infer",
    )
    parser.add_argument("--task2-dir", type=Path, default=ROOT / "data" / "TASK2")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "legalqa")
    parser.add_argument("--stopwords", type=Path, default=ROOT / "data" / "stopwords.txt")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--ce-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--public-limit", type=int, default=0)
    args = parser.parse_args()

    if args.ce_batch_size <= 0 or args.checkpoint_every <= 0:
        parser.error("Batch size and checkpoint interval must be positive")
    if not 0 <= args.public_limit <= 1000:
        parser.error("--public-limit must be in 0..1000")

    cfg = PipelineSettings(ce_batch_size=args.ce_batch_size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_cache_paths = validate_base_cache_files(cfg, args.cache_dir)

    device = detect_device(args.device)
    configure_cuda(cfg.seed)
    ensure_nltk_resources()
    task2_dir, train, public, passages, names, links = load_task2(args.task2_dir)
    train = {str(key): value for key, value in train.items()}
    public = {str(key): value for key, value in public.items()}
    stopwords = load_stopwords(args.stopwords)

    engine = LegalQAEngine(
        cfg,
        args.cache_dir,
        passages,
        names,
        links,
        stopwords,
        device=device,
        ce_batch_size=args.ce_batch_size,
        load_models=args.mode != "check",
    )
    print("Base cache validation PASS.", flush=True)

    split = split_training_data(train, cfg.seed)
    base1000, holdout1000, training5000 = split
    manifest = {
        "version": "legalqa_pipeline_v1",
        "runtime": runtime_info(device),
        "task2_dir": str(task2_dir),
        "base_cache_paths": base_cache_paths,
        "cache": cache_manifest(cfg, args.cache_dir),
        "counts": {
            "train": len(train),
            "public": len(public),
            "corpus": len(passages),
            "base_group": len(base1000),
            "holdout_group": len(holdout1000),
            "training_group": len(training5000),
        },
    }

    if args.mode == "check":
        json_dump(manifest, args.output_dir / "environment_report.json")
        print("CHECK PASS", flush=True)
        return

    audit, pair_cache, singleton_cache, provenance = prepare_features(
        cfg,
        engine,
        train,
        split,
        args.cache_dir,
        args.checkpoint_every,
    )
    manifest["feature_cache_provenance"] = provenance
    if args.mode == "prepare":
        json_dump(manifest, args.output_dir / "run_manifest.json")
        print("PREPARE PASS", flush=True)
        return

    assembled = assemble_training_actions(
        cfg,
        engine,
        train,
        base1000,
        holdout1000,
        training5000,
        audit,
        pair_cache,
        singleton_cache,
    )
    actions = assembled["actions"]
    models = train_answer_rankers(cfg, actions, args.output_dir)
    submission, inference_log = run_public_inference(
        cfg,
        engine,
        models,
        public,
        args.output_dir,
        limit=args.public_limit,
        checkpoint_every=args.checkpoint_every,
    )
    manifest["submission"] = str(submission)
    manifest["inference_log"] = str(inference_log)
    manifest["public_limit"] = int(args.public_limit)
    json_dump(manifest, args.output_dir / "run_manifest.json")


if __name__ == "__main__":
    main()
