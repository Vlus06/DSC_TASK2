#!/usr/bin/env python3
"""Independent Modal stages for cache building, training and inference."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
CACHE = DATA / "cache"
TASK2 = DATA / "TASK2"
MODEL_ROOT = Path("/models/legalqa/legalqa_selector_v21")


def run_command(command: list[str]) -> None:
    print("Command:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def build_base(kind: str, batch_size: int) -> None:
    commands = {
        "bm25": [sys.executable, "-u", str(ROOT / "scripts/build_bm25.py")],
        "dense": [
            sys.executable,
            "-u",
            str(ROOT / "scripts/build_dense_parent.py"),
            "--device",
            "cuda",
            "--batch-size",
            str(batch_size),
        ],
        "child": [
            sys.executable,
            "-u",
            str(ROOT / "scripts/build_child.py"),
            "--device",
            "cuda",
            "--batch-size",
            str(batch_size),
        ],
    }
    run_command(commands[kind])


def pack_corpus() -> None:
    from legalqa.artifacts import load_corpus_metadata
    from legalqa.settings import PipelineSettings, resolve_cache_file

    cfg = PipelineSettings()
    existing = resolve_cache_file(CACHE, cfg.corpus_cache_name)
    if existing.is_file() and existing.stat().st_size > 0:
        print(f"[corpus] packed corpus already exists: {existing}", flush=True)
        return

    output = CACHE / cfg.corpus_cache_name
    print("[corpus] CPU packing 8,532 JSON files", flush=True)
    passages, names, links = load_corpus_metadata(TASK2)
    if not (len(passages) == len(names) == len(links) == cfg.expected_docs):
        raise RuntimeError(
            f"Corpus size mismatch: passages={len(passages)}, "
            f"names={len(names)}, links={len(links)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "version": "legalqa_corpus_metadata_v1",
                "passages": passages,
                "names": names,
                "links": links,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary, output)
    print(f"[corpus] saved {output} ({output.stat().st_size:,} bytes)", flush=True)


def load_pickle(path: Path):
    print(f"[cache] loading {path.name}", flush=True)
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_complete_features(cfg, base1000, holdout1000, training5000):
    from legalqa.settings import resolve_cache_file

    paths = {
        "top": resolve_cache_file(CACHE, cfg.top_documents_cache_name),
        "audit": resolve_cache_file(CACHE, cfg.candidate_audit_cache_name),
        "pair": resolve_cache_file(CACHE, cfg.pair_features_cache_name),
        "single": resolve_cache_file(CACHE, cfg.singleton_features_cache_name),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Feature cache stage is incomplete: " + ", ".join(missing))

    top = load_pickle(paths["top"])
    audit = load_pickle(paths["audit"])
    pair = load_pickle(paths["pair"])
    single = load_pickle(paths["single"])
    remaining = set(holdout1000) | set(training5000)

    if list(map(str, top)) != list(base1000):
        raise RuntimeError("Top-document split/order mismatch")
    if set(map(str, audit)) != set(base1000):
        raise RuntimeError("Candidate-audit qids mismatch")
    if not isinstance(pair, dict) or "rows_by_qid" not in pair:
        raise RuntimeError("Pair-feature cache structure mismatch")
    if set(map(str, pair.get("rows_by_qid", {}))) != remaining:
        raise RuntimeError("Pair-feature cache is incomplete")
    if not isinstance(single, dict) or "items_by_qid" not in single:
        raise RuntimeError("Singleton-feature cache structure mismatch")
    if set(map(str, single.get("items_by_qid", {}))) != remaining:
        raise RuntimeError("Singleton-feature cache is incomplete")
    print("[cache] all four feature caches PASS", flush=True)
    return audit, pair, single


def load_context(*, load_models: bool, ce_batch_size: int = 32, evaluation_split: str = "public"):
    from legalqa.artifacts import load_evaluation_questions, load_task2, split_training_data
    from legalqa.engine import LegalQAEngine, configure_cuda, ensure_nltk_resources
    from legalqa.settings import PipelineSettings
    from legalqa.text_utils import load_stopwords

    cfg = PipelineSettings(ce_batch_size=ce_batch_size)
    configure_cuda(cfg.seed)
    ensure_nltk_resources()
    _, train, public, passages, names, links = load_task2(TASK2)
    train = {str(k): value for k, value in train.items()}
    evaluation = (
        public
        if evaluation_split == "public"
        else load_evaluation_questions(TASK2, evaluation_split)
    )
    evaluation = {str(k): value for k, value in evaluation.items()}
    if load_models:
        engine = LegalQAEngine(
            cfg,
            CACHE,
            passages,
            names,
            links,
            load_stopwords(DATA / "stopwords.txt"),
            device="cuda",
            ce_batch_size=ce_batch_size,
            load_models=True,
        )
    else:
        # Feature caches already contain every retrieval/model score required
        # for the training actions. Training only needs the answer builder and corpus text.
        engine = LegalQAEngine.__new__(LegalQAEngine)
        engine.cfg = cfg
        engine.cache_dir = CACHE
        engine.doc_id_to_passage = dict(passages)
        engine.doc_id_to_name = dict(names)
        engine.doc_id_to_link = dict(links)
        engine.stopwords = load_stopwords(DATA / "stopwords.txt")
        engine.device = "cpu"
        engine.ce_batch_size = ce_batch_size
        engine.query_cache = {}
        engine.bi_encoder = None
        engine.cross_encoder = None
        print("[train] lightweight answer-builder ready; base vectors not loaded", flush=True)
    return cfg, train, evaluation, engine, split_training_data(train, cfg.seed)


def train_selector(run_dir: Path, checkpoint_every: int) -> None:
    from legalqa.selector_training import train_selector_v21

    cfg, train, _public, engine, split = load_context(load_models=False)
    _base1000, _holdout1000, training5000 = split
    _audit, pair, single = load_complete_features(cfg, *split)
    manifest = train_selector_v21(
        engine, train, training5000, pair, single, CACHE, MODEL_ROOT,
        checkpoint_every=checkpoint_every,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "selector_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[train] frozen selector V2.1 bundle ready", flush=True)


def load_selector():
    from legalqa.model_bundle import load_bundle

    selector, manifest = load_bundle(MODEL_ROOT, verify_hashes=True)
    print(f"[infer] loaded frozen selector {manifest['version']}", flush=True)
    return selector, manifest

def infer_public(
    run_dir: Path,
    public_limit: int,
    checkpoint_every: int,
    ce_batch_size: int,
) -> None:
    from pipeline import resolved_hf_revisions, run_public_inference, sha256_file

    cfg, _train, public, engine, _split = load_context(
        load_models=True,
        ce_batch_size=ce_batch_size,
    )
    selector, selector_manifest = load_selector()
    submission, inference_log = run_public_inference(
        cfg,
        engine,
        selector,
        public,
        run_dir,
        limit=public_limit,
        checkpoint_every=checkpoint_every,
    )
    manifest = {
        "version": "legalqa_modal_selector_v21_run_v1",
        "public_limit": public_limit,
        "submission": submission.name,
        "submission_zip": submission.with_suffix(".zip").name,
        "submission_zip_sha256": sha256_file(submission.with_suffix(".zip")),
        "inference_log": inference_log.name,
        "ce_batch_size": ce_batch_size,
        "selector_bundle_version": selector_manifest["version"],
        "huggingface_revisions": resolved_hf_revisions(engine),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[infer] DONE", flush=True)


def infer_private(
    run_dir: Path,
    private_limit: int,
    checkpoint_every: int,
    ce_batch_size: int,
) -> None:
    from pipeline import resolved_hf_revisions, run_public_inference, sha256_file

    cfg, _train, private, engine, _split = load_context(
        load_models=True,
        ce_batch_size=ce_batch_size,
        evaluation_split="private",
    )
    selector, selector_manifest = load_selector()
    submission, inference_log = run_public_inference(
        cfg,
        engine,
        selector,
        private,
        run_dir,
        limit=private_limit,
        checkpoint_every=checkpoint_every,
        dataset_name="private",
    )
    manifest = {
        "version": "legalqa_modal_selector_v21_run_v1",
        "dataset": "private",
        "private_limit": private_limit,
        "submission": submission.name,
        "submission_zip": submission.with_suffix(".zip").name,
        "submission_zip_sha256": sha256_file(submission.with_suffix(".zip")),
        "inference_log": inference_log.name,
        "ce_batch_size": ce_batch_size,
        "selector_bundle_version": selector_manifest["version"],
        "huggingface_revisions": resolved_hf_revisions(engine),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[infer-private] DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "pack-corpus", "build-bm25", "build-dense", "build-child",
            "train-selector-v21", "infer", "infer-private",
        ],
    )
    parser.add_argument("--run-dir", type=Path, default=Path("/results/manual"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--public-limit", type=int, default=0)
    parser.add_argument("--private-limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--ce-batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.stage == "pack-corpus":
        pack_corpus()
    elif args.stage.startswith("build-"):
        build_base(args.stage.removeprefix("build-"), args.batch_size)
    elif args.stage == "train-selector-v21":
        train_selector(args.run_dir, args.checkpoint_every)
    elif args.stage == "infer":
        infer_public(
            args.run_dir,
            args.public_limit,
            args.checkpoint_every,
            args.ce_batch_size,
        )
    else:
        infer_private(
            args.run_dir,
            args.private_limit,
            args.checkpoint_every,
            args.ce_batch_size,
        )


if __name__ == "__main__":
    main()
