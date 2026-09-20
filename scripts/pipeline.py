#!/usr/bin/env python3
"""Run the LegalQA pipeline locally."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.artifacts import (
    build_or_load_candidate_audit,
    build_or_load_pair_features,
    build_or_load_singleton_features,
    build_or_load_top_documents,
    cache_manifest,
    load_evaluation_questions,
    load_task2,
    split_training_data,
)
from legalqa.engine import LegalQAEngine, configure_cuda, ensure_nltk_resources
from legalqa.model_bundle import load_bundle
from legalqa.selector_training import train_selector_v21
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_hf_revisions(engine) -> dict:
    """Return resolved Hub commit hashes when model configs expose them."""
    revisions = {}
    try:
        revision = getattr(engine.bi_encoder[0].auto_model.config, "_commit_hash", None)
        if revision:
            revisions["bi_encoder"] = revision
    except Exception:
        pass
    try:
        revision = getattr(engine.cross_encoder.model.config, "_commit_hash", None)
        if revision:
            revisions["reranker"] = revision
    except Exception:
        pass
    return revisions


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


def inference_artifact_names(dataset_name: str, full_run: bool, selected_count: int):
    if dataset_name not in {'public', 'private'}:
        raise ValueError(f'Unsupported inference dataset: {dataset_name}')
    label = '' if dataset_name == 'public' else '_private'
    if full_run:
        return (
            f'inference{label}_progress.pkl',
            f'submission{label}.json',
            f'inference_log{label}.json',
        )
    return (
        f'inference{label}_smoke_progress.pkl',
        f'submission{label}_smoke_{selected_count}.json',
        f'inference_log{label}_smoke_{selected_count}.json',
    )


def run_public_inference(
    cfg,
    engine,
    selector,
    public,
    output_dir: Path,
    limit: int = 0,
    checkpoint_every: int = 25,
    dataset_name: str = 'public',
):
    output_dir.mkdir(parents=True, exist_ok=True)
    full_run = not limit or limit >= len(public)
    selected_public = list(public.items()) if full_run else list(public.items())[:limit]
    progress_name, submission_name, log_name = inference_artifact_names(
        dataset_name,
        full_run,
        len(selected_public),
    )
    progress_path = output_dir / progress_name
    submission_path = output_dir / submission_name
    log_path = output_dir / log_name

    selector_signature = getattr(selector, "bundle_signature", None)
    if not isinstance(selector_signature, str):
        selector_signature = "test-or-unversioned"
    predictions, inference_log = {}, []
    if progress_path.exists():
        with progress_path.open("rb") as handle:
            progress = pickle.load(handle)
        if (
            progress.get("version") == "legalqa_inference_progress_selector_v21_v1"
            and progress.get("selector_signature") == selector_signature
        ):
            predictions = {str(key): value for key, value in progress.get("predictions", {}).items()}
            inference_log = progress.get("inference_log", [])
            retry_qids = {
                str(qid) for qid, value in predictions.items()
                if not str(value.get("answer", "")).strip()
            }
            retry_qids.update(
                str(row.get("qid")) for row in inference_log
                if row.get("error") is not None
            )
            if retry_qids:
                predictions = {
                    qid: value for qid, value in predictions.items()
                    if qid not in retry_qids
                }
                inference_log = [
                    row for row in inference_log
                    if str(row.get("qid")) not in retry_qids
                ]
                print(
                    f"Retrying {len(retry_qids)} failed/empty predictions from checkpoint",
                    flush=True,
                )
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
                "version": "legalqa_inference_progress_selector_v21_v1",
                "selector_signature": selector_signature,
                "predictions": predictions,
                "inference_log": inference_log,
            },
            progress_path,
        )

    print(f"{dataset_name.upper()} todo={len(pending)}", flush=True)
    started = time.time()
    for index, (qid, item) in enumerate(pending, start=1):
        item_started = time.time()
        try:
            prediction, debug = engine.predict(
                item["question"],
                selector,
                qid=qid,
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
    if set(predictions) != allowed:
        raise RuntimeError("Submission qid set does not exactly match the selected evaluation set")

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
    zip_path = submission_path.with_suffix(".zip")
    temporary_zip = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, arcname="submission.json")
    os.replace(temporary_zip, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.json"]:
            raise RuntimeError("Submission ZIP must contain only submission.json")
    print(f"READY: {submission_path}\nREADY ZIP: {zip_path}", flush=True)
    return submission_path, log_path


def load_complete_feature_caches(cfg, cache_dir: Path, scale_qids: list[str]):
    with (cache_dir / cfg.pair_features_cache_name).open("rb") as handle:
        pair_cache = pickle.load(handle)
    with (cache_dir / cfg.singleton_features_cache_name).open("rb") as handle:
        singleton_cache = pickle.load(handle)
    expected = set(map(str, scale_qids))
    pair_qids = set(map(str, pair_cache.get("rows_by_qid", {})))
    singleton_qids = set(map(str, singleton_cache.get("items_by_qid", {})))
    if not expected <= pair_qids or not expected <= singleton_qids:
        raise RuntimeError("Feature caches do not contain all frozen SCALE5000 qids")
    return pair_cache, singleton_cache


def lightweight_engine(cfg, cache_dir, passages, names, links, stopwords):
    engine = LegalQAEngine.__new__(LegalQAEngine)
    engine.cfg = cfg
    engine.cache_dir = Path(cache_dir)
    engine.doc_id_to_passage = dict(passages)
    engine.doc_id_to_name = dict(names)
    engine.doc_id_to_link = dict(links)
    engine.stopwords = stopwords
    engine.device = "cpu"
    engine.ce_batch_size = cfg.ce_batch_size
    engine.query_cache = {}
    engine.bi_encoder = None
    engine.cross_encoder = None
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalQA Task 2 pipeline")
    parser.add_argument(
        "--mode",
        choices=["check", "prepare", "train-selector-v21", "infer"],
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
    parser.add_argument("--private-limit", type=int, default=0)
    parser.add_argument("--split", choices=["public", "private"], default="public")
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "data" / "models" / "legalqa_selector_v21",
    )
    args = parser.parse_args()

    if args.ce_batch_size <= 0 or args.checkpoint_every <= 0:
        parser.error("Batch size and checkpoint interval must be positive")
    if args.public_limit < 0 or args.private_limit < 0:
        parser.error("Inference limits must be non-negative")

    cfg = PipelineSettings(ce_batch_size=args.ce_batch_size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_cache_paths = (
        {} if args.mode == "train-selector-v21"
        else validate_base_cache_files(cfg, args.cache_dir)
    )

    device = detect_device(args.device)
    configure_cuda(cfg.seed)
    ensure_nltk_resources()
    task2_dir, train, public, passages, names, links = load_task2(args.task2_dir)
    train = {str(key): value for key, value in train.items()}
    public = {str(key): value for key, value in public.items()}
    stopwords = load_stopwords(args.stopwords)

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
        LegalQAEngine(
            cfg, args.cache_dir, passages, names, links, stopwords,
            device=device, ce_batch_size=args.ce_batch_size, load_models=False,
        )
        json_dump(manifest, args.output_dir / "environment_report.json")
        print("CHECK PASS", flush=True)
        return

    if args.mode == "prepare":
        engine = LegalQAEngine(
            cfg, args.cache_dir, passages, names, links, stopwords,
            device=device, ce_batch_size=args.ce_batch_size, load_models=True,
        )
        _audit, _pair, _single, provenance = prepare_features(
            cfg, engine, train, split, args.cache_dir, args.checkpoint_every,
        )
        manifest["feature_cache_provenance"] = provenance
        json_dump(manifest, args.output_dir / "run_manifest.json")
        print("PREPARE PASS", flush=True)
        return

    if args.mode == "train-selector-v21":
        pair_cache, singleton_cache = load_complete_feature_caches(
            cfg, args.cache_dir, training5000,
        )
        engine = lightweight_engine(
            cfg, args.cache_dir, passages, names, links, stopwords,
        )
        selector_manifest = train_selector_v21(
            engine, train, training5000, pair_cache, singleton_cache,
            args.cache_dir, args.model_dir,
            checkpoint_every=args.checkpoint_every,
        )
        manifest["selector_bundle"] = selector_manifest
        json_dump(manifest, args.output_dir / "run_manifest.json")
        print("TRAIN SELECTOR V2.1 PASS", flush=True)
        return

    evaluation = public if args.split == "public" else load_evaluation_questions(
        args.task2_dir, "private",
    )
    evaluation = {str(key): value for key, value in evaluation.items()}
    selector, selector_manifest = load_bundle(args.model_dir, verify_hashes=True)
    engine = LegalQAEngine(
        cfg, args.cache_dir, passages, names, links, stopwords,
        device=device, ce_batch_size=args.ce_batch_size, load_models=True,
    )
    inference_limit = args.public_limit if args.split == "public" else args.private_limit
    submission, inference_log = run_public_inference(
        cfg,
        engine,
        selector,
        evaluation,
        args.output_dir,
        limit=inference_limit,
        checkpoint_every=args.checkpoint_every,
        dataset_name=args.split,
    )
    manifest["submission"] = str(submission)
    manifest["submission_zip"] = str(submission.with_suffix(".zip"))
    manifest["submission_zip_sha256"] = sha256_file(submission.with_suffix(".zip"))
    manifest["inference_log"] = str(inference_log)
    manifest["dataset"] = args.split
    manifest["inference_limit"] = int(inference_limit)
    manifest["selector_bundle_version"] = selector_manifest["version"]
    manifest["huggingface_revisions"] = resolved_hf_revisions(engine)
    json_dump(manifest, args.output_dir / "run_manifest.json")


if __name__ == "__main__":
    main()
