#!/usr/bin/env python
"""
Stage 1 -- Fine-tune the bi-encoder (retriever) and cross-encoder (reranker).

Requires a dense chunk cache built with the BASE (pre-finetune) bi-encoder
-- if you don't have one yet, run `scripts/02_build_dense_cache.py` first
with `models.bi_encoder_name_for_inference=None` (defaults to the base
model), THEN come back and run this script, THEN re-run
`02_build_dense_cache.py` pointed at the fine-tuned model to get the final
cache used by the QA pipeline.

Usage:
    python scripts/01_finetune_retriever_reranker.py \
        --task2-data-dir /data/TASK2/TASK2 \
        --output-dir /data/outputs \
        --dense-cache-file /data/outputs/cache/dense_chunk_index_base.pkl \
        --stopwords-file /data/stopwords.txt \
        --bm25-cache-file /data/bm25_index_stopword.pkl
"""
from __future__ import annotations

import argparse

from legalqa.config import PipelineConfig
from legalqa.retrieval.dense import DenseChunkCache
from legalqa.training.finetune_orchestrator import RetrieverFinetuner
from legalqa.utils import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: fine-tune retriever + reranker")
    p.add_argument("--task2-data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dense-cache-file", required=True, help="Dense chunk cache built with the BASE bi-encoder.")
    p.add_argument("--stopwords-file", default=None)
    p.add_argument("--bm25-cache-file", default=None)
    p.add_argument("--base-bi-encoder", default="AITeamVN/Vietnamese_Embedding")
    p.add_argument("--base-cross-encoder", default="AITeamVN/Vietnamese_Reranker")
    p.add_argument("--n-hard-neg-per-query", type=int, default=4)
    p.add_argument("--bi-epochs", type=int, default=2)
    p.add_argument("--bi-batch-size", type=int, default=16)
    p.add_argument("--bi-mini-batch-size", type=int, default=4)
    p.add_argument("--ce-epochs", type=int, default=2)
    p.add_argument("--ce-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig()
    config.paths.task2_data_dir = args.task2_data_dir
    config.paths.output_dir = args.output_dir
    config.paths.dense_cache_file = args.dense_cache_file
    config.paths.stopwords_file = args.stopwords_file
    config.paths.bm25_cache_file = args.bm25_cache_file
    config.paths.ensure_dirs()

    config.models.base_bi_encoder_name = args.base_bi_encoder
    config.models.base_cross_encoder_name = args.base_cross_encoder

    config.mining.n_hard_neg_per_query = args.n_hard_neg_per_query
    config.mining.seed = args.seed

    config.bi_train.epochs = args.bi_epochs
    config.bi_train.batch_size = args.bi_batch_size
    config.bi_train.mini_batch_size = args.bi_mini_batch_size

    config.ce_train.epochs = args.ce_epochs
    config.ce_train.batch_size = args.ce_batch_size

    logger.info(f"Loading dense chunk cache from: {config.paths.dense_cache_file}")
    chunk_cache = DenseChunkCache.load(config.paths.dense_cache_file)

    finetuner = RetrieverFinetuner(config, chunk_cache)
    finetuner.run()

    logger.info("Stage 1 done. Next: rebuild the dense chunk cache with the fine-tuned bi-encoder")
    logger.info(f"  (point --bi-encoder-name at {config.paths.finetuned_bi_encoder_dir}),")
    logger.info("  then run scripts/03_run_pipeline.py.")


if __name__ == "__main__":
    main()