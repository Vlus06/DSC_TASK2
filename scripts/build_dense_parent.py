#!/usr/bin/env python3
import argparse
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.config import BaseCacheSettings
from legalqa.corpus import load_corpus
from legalqa.parent_cache import chunk_by_dieu_khoan, cache_meta, expected_cache_filename


def main():
    parser = argparse.ArgumentParser(description="Build the dense parent cache")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    import numpy as np

    cfg = BaseCacheSettings()
    if cfg.dense_parent_cache.name != expected_cache_filename():
        raise RuntimeError(
            f"Dense cache filename mismatch: config={cfg.dense_parent_cache.name}, "
            f"expected={expected_cache_filename()}"
        )

    passages, names = load_corpus()
    chunk_doc_ids_all, chunk_texts_all = [], []
    for doc_id, passage in passages.items():
        for chunk in chunk_by_dieu_khoan(passage, doc_id, names.get(doc_id, ""), max_chars=5117):
            chunk_doc_ids_all.append(chunk["doc_id"])
            chunk_texts_all.append(chunk["text"])

    print(f"Parent chunks: {len(chunk_texts_all)} (expected: 738506)")
    if len(passages) == 8532 and len(chunk_texts_all) != 738506:
        raise RuntimeError(
            "Parent chunk count does not match the expected value (738506). "
            "Check the corpus and preprocessing settings."
        )

    from sentence_transformers import SentenceTransformer
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; enable GPU or use --device cpu")
    print(f"Encoding parent chunks on {device}; batch size={args.batch_size}", flush=True)
    model = SentenceTransformer(cfg.bi_encoder_model, device=device, trust_remote_code=True)
    model.max_seq_length = cfg.bi_encoder_max_seq_length
    vecs = model.encode(
        chunk_texts_all,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cfg.dense_parent_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cfg.dense_parent_cache.with_suffix(cfg.dense_parent_cache.suffix + ".tmp")
    with temporary_path.open("wb") as f:
        pickle.dump({
            "chunk_doc_ids_all": chunk_doc_ids_all,
            "chunk_texts_all": chunk_texts_all,
            "chunk_vecs_all": vecs,
            "meta": cache_meta(len(passages)),
        }, f)
    temporary_path.replace(cfg.dense_parent_cache)
    print(f"Saved: {cfg.dense_parent_cache}")


if __name__ == "__main__":
    main()
