#!/usr/bin/env python3
"""Build child embeddings without loading BM25, dense parent, or reranker."""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.config import BaseCacheSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    cfg = BaseCacheSettings()
    if cfg.child_cache.exists():
        print(f"Child cache already exists; skipping build: {cfg.child_cache}", flush=True)
        return

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from legalqa.child_cache import load_or_build_child_cache
    from legalqa.corpus import load_corpus

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

    print(f"Building child cache on {device}; batch size={args.batch_size}", flush=True)
    passages, names = load_corpus()
    encoder = SentenceTransformer(cfg.bi_encoder_model, device=device, trust_remote_code=True)
    encoder.max_seq_length = cfg.bi_encoder_max_seq_length
    load_or_build_child_cache(cfg, passages, names, encoder, args.batch_size)


if __name__ == "__main__":
    main()
