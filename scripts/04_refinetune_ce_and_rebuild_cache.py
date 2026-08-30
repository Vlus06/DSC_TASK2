#!/usr/bin/env python
"""
Stage 1b -- mirrors `t-ng-2(2).ipynb`:

  PART 1: Re-finetune the CROSS-ENCODER from an EXISTING
          mined_training_pairs.json (no re-mining), with the save-verify
          fix (explicit .save() in a finally-block + check that model
          weights/config.json actually landed on disk).
  PART 2: Build a NEW dense chunk cache for the FINE-TUNED bi-encoder,
          using the exact v4.1 chunk logic (join_broken_lines with the
          list-marker fix, Dieu/paragraph/Khoan fallback chunking,
          BOILERPLATE_TITLE_PATTERN filtering, hash-versioned cache path)
          -- completely independent from the cache built from the base
          model, so it never gets silently overwritten.

Usage:
    python scripts/04_refinetune_ce_and_rebuild_cache.py \
        --task2-data-dir /data/TASK2/TASK2 \
        --output-dir /data/outputs \
        --mined-pairs-path /data/outputs/mined_training_pairs.json \
        --finetuned-bi-encoder-dir /data/outputs/finetuned_bi_encoder
"""
from __future__ import annotations

import argparse
import os

from legalqa.config import PipelineConfig
from legalqa.corpus import Corpus
from legalqa.retrieval.dense import DenseChunkCacheBuilder
from legalqa.training.cross_encoder_trainer import CrossEncoderTrainer
from legalqa.utils import get_device, logger, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1b: re-finetune CE from existing pairs + rebuild dense cache")
    p.add_argument("--task2-data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mined-pairs-path", required=True, help="Existing mined_training_pairs.json (no re-mining).")
    p.add_argument("--finetuned-bi-encoder-dir", required=True, help="Bi-encoder already fine-tuned (Stage 1 output).")
    p.add_argument("--base-cross-encoder", default="AITeamVN/Vietnamese_Reranker")
    p.add_argument("--ce-epochs", type=int, default=2)
    p.add_argument("--ce-batch-size", type=int, default=8)
    p.add_argument("--encode-max-seq-length", type=int, default=2048)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    config = PipelineConfig()
    config.paths.task2_data_dir = args.task2_data_dir
    config.paths.output_dir = args.output_dir
    config.paths.ensure_dirs()

    config.ce_train.epochs = args.ce_epochs
    config.ce_train.batch_size = args.ce_batch_size

    device = get_device(config.models.device)

    # ---------------- PART 1: re-finetune cross-encoder ----------------
    logger.info("=" * 70)
    logger.info("PART 1 -- re-finetune cross-encoder from existing mined pairs")
    logger.info("=" * 70)

    ft_cross_encoder_out_dir = config.paths.finetuned_cross_encoder_dir
    CrossEncoderTrainer.train_from_existing_pairs(
        model_name=args.base_cross_encoder,
        device=device,
        config=config.ce_train,
        mined_pairs_path=args.mined_pairs_path,
        output_path=ft_cross_encoder_out_dir,
        seed=args.seed,
        eval_holdout_max=config.bi_train.eval_holdout_max,
        eval_holdout_fraction=config.bi_train.eval_holdout_fraction,
    )
    logger.info(f"PART 1 done. Fine-tuned cross-encoder saved (and verified) at: {ft_cross_encoder_out_dir}")

    # ---------------- PART 2: rebuild dense cache for the fine-tuned bi-encoder ----------------
    logger.info("=" * 70)
    logger.info("PART 2 -- build NEW dense chunk cache for the FINE-TUNED bi-encoder")
    logger.info("=" * 70)

    from sentence_transformers import SentenceTransformer

    bi_encoder_name = args.finetuned_bi_encoder_dir
    bi_encoder = SentenceTransformer(bi_encoder_name, device=device, trust_remote_code=True)
    bi_encoder.max_seq_length = args.encode_max_seq_length
    logger.info(f"Loaded FINE-TUNED bi-encoder: {bi_encoder_name} (max_seq_length={bi_encoder.max_seq_length})")

    config.chunk_cache.encode_batch_size = args.encode_batch_size
    # Distinguish this cache's chunk_logic_version from the base-model
    # cache so the two never collide/overwrite each other.
    config.chunk_cache.chunk_logic_version_suffix = "_FINETUNED_BI"

    corpus = Corpus.load(config.paths.context_dir)
    builder = DenseChunkCacheBuilder(corpus, config.chunk_cache)

    cache_path = builder.default_cache_path(config.paths.cache_dir, model_name="finetuned_bi_encoder")
    if os.path.exists(cache_path):
        logger.info(f"Cache already exists at {cache_path} -- loading instead of rebuilding.")
        from legalqa.retrieval.dense import DenseChunkCache

        cache = DenseChunkCache.load(cache_path)
    else:
        cache = builder.build(bi_encoder)
        cache.meta["model_name"] = bi_encoder_name
        cache.meta["base_model_finetuned_from"] = config.models.base_bi_encoder_name
        cache.save(cache_path)

    assert len(cache.chunk_texts_all) > 0 and len(cache.chunk_vecs_all) == len(cache.chunk_texts_all), (
        "Cache data is invalid (empty or mismatched chunk/vector counts)!"
    )
    logger.info(f"VERIFY OK: cache has {len(cache.chunk_texts_all)} chunks, vectors shape={cache.chunk_vecs_all.shape}")

    logger.info("=" * 70)
    logger.info("STAGE 1b COMPLETE.")
    logger.info(f"  Fine-tuned cross-encoder (NEW) : {ft_cross_encoder_out_dir}")
    logger.info(f"  Dense cache for fine-tuned bi-encoder (NEW): {cache_path}")
    logger.info("=" * 70)
    logger.info("NEXT STEP: point scripts/03_run_pipeline.py at:")
    logger.info(f"  --bi-encoder-name {bi_encoder_name}")
    logger.info(f"  --cross-encoder-name {ft_cross_encoder_out_dir}")
    logger.info(f"  --dense-cache-file {cache_path}")


if __name__ == "__main__":
    main()
