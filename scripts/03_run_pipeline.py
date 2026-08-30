#!/usr/bin/env python
"""
Stage 3 -- Main QA pipeline: grid-search tune on a held-out slice of
train.json (reproducing the v9 notebook's staged sweeps), then run
inference over public-official.json and write submission.json.

Usage:
    python scripts/03_run_pipeline.py \
        --task2-data-dir /data/TASK2/TASK2 \
        --output-dir /data/outputs \
        --dense-cache-file /data/outputs/cache/dense_chunk_index_finetuned.pkl \
        --bi-encoder-name /data/outputs/finetuned_bi_encoder \
        --cross-encoder-name /data/outputs/finetuned_cross_encoder \
        --stopwords-file /data/stopwords.txt \
        --bm25-cache-file /data/bm25_index_stopword.pkl \
        --val-size 1000
"""
from __future__ import annotations

import argparse
import json
import os

from legalqa.config import PipelineConfig
from legalqa.pipeline.builder import build_corpus_and_bm25, load_bi_encoder, load_dense_chunk_cache
from legalqa.pipeline.inference_runner import InferenceRunner
from legalqa.pipeline.qa_pipeline import QAPipeline
from legalqa.pipeline.tuner import PipelineTuner
from legalqa.utils import logger, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: tune + run the QA pipeline")
    p.add_argument("--task2-data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dense-cache-file", required=True)
    p.add_argument("--bi-encoder-name", required=True, help="Fine-tuned bi-encoder dir (or base model name).")
    p.add_argument("--cross-encoder-name", default=None, help="Fine-tuned cross-encoder dir (defaults to base).")
    p.add_argument("--stopwords-file", default=None)
    p.add_argument("--bm25-cache-file", default=None)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--skip-tuning", action="store_true", help="Skip grid-search, use config defaults as-is.")
    p.add_argument("--run-tag", default="final")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    config = PipelineConfig()
    config.paths.task2_data_dir = args.task2_data_dir
    config.paths.output_dir = args.output_dir
    config.paths.dense_cache_file = args.dense_cache_file
    config.paths.stopwords_file = args.stopwords_file
    config.paths.bm25_cache_file = args.bm25_cache_file
    config.paths.ensure_dirs()

    config.models.bi_encoder_name_for_inference = args.bi_encoder_name
    if args.cross_encoder_name:
        config.models.cross_encoder_name_for_inference = args.cross_encoder_name

    config.tuning.val_size = args.val_size
    config.tuning.seed = args.seed

    logger.info("Loading corpus + BM25 index ...")
    corpus, bm25 = build_corpus_and_bm25(config)

    logger.info("Loading dense chunk cache ...")
    dense_cache = load_dense_chunk_cache(config)

    logger.info("Loading bi-encoder for inference ...")
    bi_encoder = load_bi_encoder(config)

    if not args.skip_tuning:
        logger.info("Running grid-search tuning on the validation split ...")
        tuner = PipelineTuner(config, corpus, bm25, dense_cache, bi_encoder)
        final_meteor = tuner.run()

        tuning_log_path = os.path.join(config.paths.output_dir, f"tuning_log_{args.run_tag}.json")
        with open(tuning_log_path, "w", encoding="utf-8") as f:
            json.dump({"final_meteor_on_val": final_meteor, **tuner.log}, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved tuning log to: {tuning_log_path}")
    else:
        logger.info("Skipping tuning -- using config defaults as-is.")

    logger.info("Final config:")
    logger.info(f"  W_BM25={config.retrieval.w_bm25} W_DENSE={config.retrieval.w_dense} TOP_N_DOCS={config.retrieval.top_n_docs}")
    logger.info(f"  USE_CROSS_ENCODER={config.rerank.use_cross_encoder} POST_PROCESS_MAX_CHARS={config.answer.post_process_max_chars}")
    logger.info(f"  USE_DEDUPE={config.answer.use_dedupe}")

    logger.info("Building the final QAPipeline and running inference ...")
    pipeline = QAPipeline(config, corpus, bm25, dense_cache, bi_encoder)
    InferenceRunner(config, pipeline, run_tag=args.run_tag).run()


if __name__ == "__main__":
    main()