#!/usr/bin/env python3
import argparse
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.config import OUTPUT_DIR, STOPWORDS_FILE, PipelineConfig
from legalqa.corpus import load_corpus, load_public, load_train
from legalqa.metrics import compute_meteor, coverage_score
from legalqa.pipeline import LegalQAPipeline
from legalqa.retrieval import Retriever
from legalqa.text_utils import load_stopwords


def ensure_base_caches(cfg: PipelineConfig, build_missing: bool):
    missing = [p for p in (cfg.bm25_cache, cfg.dense_parent_cache) if not p.exists()]
    if missing and not build_missing:
        names = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing base cache(s):\n" + names +
            "\nRun again with --build-missing to build them from data/."
        )
    for path, script in (
        (cfg.bm25_cache, "build_bm25.py"),
        (cfg.dense_parent_cache, "build_dense_parent.py"),
    ):
        if path.exists():
            print(f"Using existing cache; skipping build: {path}", flush=True)
        else:
            print(f"Missing cache; building: {path}", flush=True)
            subprocess.check_call([sys.executable, "-u", str(ROOT / "scripts" / script)])


def validation_qids(train_raw, cfg: PipelineConfig):
    qids = list(train_raw.keys())
    rng = random.Random(cfg.seed)
    rng.shuffle(qids)
    return qids[:cfg.val_size]


def get_val_top_docs(train_raw, val_qids, retriever, cfg):
    cfg.raw_cands_cache.parent.mkdir(parents=True, exist_ok=True)
    if cfg.raw_cands_cache.exists():
        with cfg.raw_cands_cache.open("rb") as f:
            raw = pickle.load(f)
    else:
        raw = []
        for i, qid in enumerate(val_qids, 1):
            item = train_raw[qid]
            question = item["question"]
            gold = item.get("answer", "") or item.get("gold_answer", "") or ""
            bm25 = retriever.bm25_retrieve(question)
            dense = retriever.dense_retrieve_full_corpus(question)
            raw.append((qid, question, gold, bm25, dense))
            if i % 100 == 0:
                print(f"raw candidates: {i}/{len(val_qids)}")
        with cfg.raw_cands_cache.open("wb") as f:
            pickle.dump(raw, f)

    if cfg.val_top_docs_cache.exists():
        with cfg.val_top_docs_cache.open("rb") as f:
            return pickle.load(f)

    top_docs = {
        qid: (question, gold, retriever.fuse_from_raw(bm25, dense, top_n=cfg.n_docs_sweep_max))
        for qid, question, gold, bm25, dense in raw
    }
    with cfg.val_top_docs_cache.open("wb") as f:
        pickle.dump(top_docs, f)
    return top_docs


def validate(pipeline, retriever, stopwords, cfg):
    train_raw = load_train()
    val_qids = validation_qids(train_raw, cfg)
    top_docs = get_val_top_docs(train_raw, val_qids, retriever, cfg)

    rows = []
    for i, qid in enumerate(val_qids, 1):
        question, gold, docs = top_docs[qid]
        pred = pipeline.predict_one(
            question,
            docs,
            n_docs=cfg.final_top_n_docs,
            use_automerge=cfg.use_automerge,
            ce_topk=cfg.ce_top_k_candidates,
            margin=cfg.ce_margin,
            max_keep=cfg.ce_max_keep,
            max_chars=cfg.post_process_max_chars,
        )
        rows.append({
            "qid": qid,
            "question": question,
            "gold": gold,
            "pred": pred,
            "meteor": compute_meteor(pred, gold),
            "coverage": coverage_score(pred, gold, stopwords),
        })
        if i % 100 == 0:
            print(f"validation: {i}/{len(val_qids)}")

    meteor = float(np.mean([r["meteor"] for r in rows]))
    with (OUTPUT_DIR / "val_debug.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"METEOR = {meteor:.4f}")
    print("Historical full-corpus notebook target: 0.5780")
    print("=" * 64)

    if not (cfg.expected_meteor_min <= meteor <= cfg.expected_meteor_max):
        raise RuntimeError(
            f"Regression check failed: METEOR={meteor:.4f}, expected a 0.57x result "
            f"in [{cfg.expected_meteor_min:.4f}, {cfg.expected_meteor_max:.4f}]. "
            "Check model snapshot, caches, package versions, and GPU environment."
        )
    return meteor


def infer(pipeline, retriever, cfg):
    public = load_public()
    predictions = {}
    for i, (qid, item) in enumerate(public.items(), 1):
        question = item["question"]
        bm25 = retriever.bm25_retrieve(question)
        dense = retriever.dense_retrieve_full_corpus(question)
        docs = retriever.fuse_from_raw(bm25, dense, top_n=cfg.n_docs_sweep_max)
        pred = pipeline.predict_one(
            question,
            docs,
            n_docs=cfg.final_top_n_docs,
            use_automerge=cfg.use_automerge,
            ce_topk=cfg.ce_top_k_candidates,
            margin=cfg.ce_margin,
            max_keep=cfg.ce_max_keep,
            max_chars=cfg.post_process_max_chars,
        )
        predictions[qid] = {"answer": pred}
        if i % 50 == 0:
            print(f"inference: {i}/{len(public)}")

    out = OUTPUT_DIR / "submission.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "infer", "full"], default="full")
    parser.add_argument("--build-missing", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    ensure_base_caches(cfg, args.build_missing)

    import torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    passages, names = load_corpus()
    stopwords = load_stopwords(STOPWORDS_FILE)
    retriever = Retriever(cfg, stopwords, passages, device)
    pipeline = LegalQAPipeline(cfg, retriever, passages, names, device)

    if args.mode in ("validate", "full"):
        validate(pipeline, retriever, stopwords, cfg)
    if args.mode in ("infer", "full"):
        infer(pipeline, retriever, cfg)


if __name__ == "__main__":
    main()
