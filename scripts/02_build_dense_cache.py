#!/usr/bin/env python
"""
Stage 2 -- Build the dense chunk cache (chunk the corpus + embed every
chunk with the bi-encoder), using the authoritative v4.1 chunk logic. Run
this TWICE across a full reproduction:

  1. First with the BASE bi-encoder (before Stage 1 fine-tuning) --
     Stage 1's hard-negative miner needs a chunk+embedding cache to mine
     Dense hard negatives from.
  2. Again with the FINE-TUNED bi-encoder (after Stage 1) -- prefer
     `scripts/04_refinetune_ce_and_rebuild_cache.py` for this step, since
     it also re-finetunes the cross-encoder from the existing mined pairs
     in the same run (mirrors `t-ng-2(2).ipynb`). This script still works
     standalone if you only need the cache rebuilt.

Usage:
    python scripts/02_build_dense_cache.py \
        --task2-data-dir /data/TASK2/TASK2 \
        --output-dir /data/outputs \
        --bi-encoder-name AITeamVN/Vietnamese_Embedding
"""
from __future__ import annotations

import argparse
import os

from legalqa.config import PipelineConfig
from legalqa.corpus import Corpus
from legalqa.retrieval.dense import DenseChunkCache, DenseChunkCacheBuilder
from legalqa.utils import get_device, logger, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: build dense chunk cache")
    p.add_argument("--task2-data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bi-encoder-name", required=True, help="Base model name OR path to fine-tuned model dir.")
    p.add_argument("--cache-out", default=None, help="Where to write the .pkl cache (default: hash-versioned path under --output-dir/cache).")
    p.add_argument("--encode-max-seq-length", type=int, default=2048)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--is-finetuned", action="store_true", help="Set this when --bi-encoder-name points at a fine-tuned model dir, so the cache's chunk_logic_version doesn't collide with the base-model cache.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    config = PipelineConfig()
    config.paths.task2_data_dir = args.task2_data_dir
    config.paths.output_dir = args.output_dir
    config.paths.ensure_dirs()
    config.chunk_cache.encode_batch_size = args.encode_batch_size
    if args.is_finetuned:
        config.chunk_cache.chunk_logic_version_suffix = "_FINETUNED_BI"

    corpus = Corpus.load(config.paths.context_dir)

    from sentence_transformers import SentenceTransformer

    device = get_device(config.models.device)
    logger.info(f"Loading bi-encoder for chunking/embedding: {args.bi_encoder_name} (device={device})")
    bi_encoder = SentenceTransformer(args.bi_encoder_name, device=device, trust_remote_code=True)
    bi_encoder.max_seq_length = args.encode_max_seq_length

    builder = DenseChunkCacheBuilder(corpus, config.chunk_cache)

    cache_out = args.cache_out or builder.default_cache_path(
        config.paths.cache_dir,
        model_name=("finetuned_bi_encoder" if args.is_finetuned else args.bi_encoder_name),
    )

    if os.path.exists(cache_out):
        logger.info(f"Cache already exists at {cache_out} -- loading instead of rebuilding.")
        cache = DenseChunkCache.load(cache_out)
    else:
        cache = builder.build(bi_encoder)
        cache.meta["model_name"] = args.bi_encoder_name
        cache.save(cache_out)

    logger.info(f"Dense chunk cache ready at: {cache_out}")
    logger.info(f"Cache meta: {cache.meta}")


if __name__ == "__main__":
    main()
