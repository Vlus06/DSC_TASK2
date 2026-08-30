"""
All hyperparameters / paths for the LegalQA pipeline, as dataclasses.

This file was shipped EMPTY in the original package export -- every other
module does `from .config import PipelineConfig` (or one of the nested
config classes below), so nothing could import without this file. Values
here mirror the constants used across the two original notebooks
(hard-negative-mining/fine-tuning notebook + the v9 RAG QA pipeline
notebook) and the CE-refinetune + dense-cache-rebuild notebook.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathsConfig:
    task2_data_dir: str = ""
    output_dir: str = ""

    # Optional pre-built caches -- if given and present on disk, loaded
    # instead of rebuilt from scratch.
    dense_cache_file: Optional[str] = None
    stopwords_file: Optional[str] = None
    bm25_cache_file: Optional[str] = None

    @property
    def context_dir(self) -> str:
        return os.path.join(self.task2_data_dir, "selected-contexts")

    @property
    def train_path(self) -> str:
        return os.path.join(self.task2_data_dir, "train.json")

    @property
    def public_official_path(self) -> str:
        return os.path.join(self.task2_data_dir, "public-official.json")

    @property
    def mined_pairs_path(self) -> str:
        return os.path.join(self.output_dir, "mined_training_pairs.json")

    @property
    def finetuned_bi_encoder_dir(self) -> str:
        return os.path.join(self.output_dir, "finetuned_bi_encoder")

    @property
    def finetuned_cross_encoder_dir(self) -> str:
        return os.path.join(self.output_dir, "finetuned_cross_encoder")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.output_dir, "cache")

    def ensure_dirs(self) -> None:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class ModelsConfig:
    base_bi_encoder_name: str = "AITeamVN/Vietnamese_Embedding"
    base_cross_encoder_name: str = "AITeamVN/Vietnamese_Reranker"

    # If set, used for inference/tuning instead of the base model name
    # (point these at finetuned_bi_encoder_dir / finetuned_cross_encoder_dir
    # after Stage 1 has produced them).
    bi_encoder_name_for_inference: Optional[str] = None
    cross_encoder_name_for_inference: Optional[str] = None

    device: Optional[str] = None  # None -> auto-detect (cuda if available)

    encode_max_seq_length: int = 2048  # used when embedding the full corpus / queries at inference time
    train_max_seq_length: int = 512  # used only while fine-tuning the bi-encoder (memory constraint)

    def resolved_bi_encoder_name(self) -> str:
        return self.bi_encoder_name_for_inference or self.base_bi_encoder_name

    def resolved_cross_encoder_name(self) -> str:
        return self.cross_encoder_name_for_inference or self.base_cross_encoder_name


@dataclass
class ChunkCacheConfig:
    """Controls `DenseChunkCacheBuilder` (chunk-by-Dieu/Khoan + embed).

    Mirrors the authoritative v4.1 `build_dense_cache` script exactly:
    chunk_logic_version, chars-per-token calibration and all
    enable_* flags feed into the MD5-hashed cache filename so that
    changing any of them (or the model) never silently overwrites a
    different cache.
    """

    chunk_max_chars: Optional[int] = None  # None -> derived from encode_max_seq_length * chars_per_token_init * safety_margin
    chars_per_token_init: float = 2.94
    safety_margin: float = 0.85

    enable_doc_name_prefix: bool = True
    enable_dieu_title_prefix: bool = True
    enable_boilerplate_filter: bool = True
    enable_line_join_fix: bool = True

    encode_batch_size: int = 64

    # Appended to the base "v4_1_..." chunk_logic_version string so the
    # cache built from the fine-tuned bi-encoder never collides with (or
    # silently overwrites) the cache built from the base model.
    chunk_logic_version_suffix: str = ""


@dataclass
class MiningConfig:
    seed: int = 42
    overlap_fallback_threshold: float = 0.15

    n_hard_neg_per_query: int = 4
    max_train_chars: int = 1500  # ~512 tokens, matches bi_train.train_max_seq_length

    bm25_top_k_for_mining: int = 60
    dense_top_k_for_mining: int = 60
    n_bm25_neg_docs: int = 6
    n_dense_neg_docs: int = 6
    n_intra_doc_candidates: int = 10


@dataclass
class BiEncoderTrainConfig:
    epochs: int = 2
    batch_size: int = 16
    mini_batch_size: int = 4  # CachedMultipleNegativesRankingLoss inner forward-pass chunk size
    warmup_ratio: float = 0.1
    lr: float = 2e-5
    use_gradient_checkpointing: bool = True

    eval_holdout_max: int = 200
    eval_holdout_fraction: float = 0.05  # 1/20, same split used for both bi-encoder and cross-encoder holdout


@dataclass
class CrossEncoderTrainConfig:
    epochs: int = 2
    batch_size: int = 8
    warmup_ratio: float = 0.1
    lr: float = 2e-5
    max_length: int = 512


@dataclass
class RetrievalConfig:
    bm25_top_k: int = 400
    dense_top_k: int = 250
    n_docs_sweep_max: int = 30  # how many fused docs to keep before TOP_N_DOCS narrows further

    w_bm25: float = 0.5
    w_dense: float = 1.0
    top_n_docs: int = 5

    recency_max_boost: float = 0.10
    recency_base_year: int = 2015


@dataclass
class RerankConfig:
    use_cross_encoder: bool = True
    top_k_candidates: int = 50

    # Two mutually-exclusive keep strategies -- PipelineTuner decides which
    # one wins on the validation split and sets use_dynamic_margin
    # accordingly.
    keep_top_n: int = 2
    use_dynamic_margin: bool = True
    margin: float = 0.5
    max_keep: int = 2

    alpha: float = 1.0  # blend: alpha*ce_score + (1-alpha)*emb_score


@dataclass
class AnswerConfig:
    emb_threshold: float = 0.6
    emb_max_total_chunks: int = 8
    emb_min_kept_fallback: int = 1
    lex_tiebreak_margin: float = 0.02

    post_process_max_chars: int = 5800
    use_dedupe: bool = False


@dataclass
class TuningConfig:
    val_size: int = 1000
    seed: int = 42

    w_bm25_candidates: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5])
    w_dense_fixed: float = 1.0

    n_docs_candidates: List[int] = field(default_factory=lambda: [5, 8, 10, 12, 15, 18, 20, 25, 30])

    ce_top_k_candidates: List[int] = field(default_factory=lambda: [10, 15, 20, 30, 50])
    ce_margin_candidates: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5])
    ce_max_keep_candidates: List[int] = field(default_factory=lambda: [2, 3, 4])
    ce_alpha_candidates: List[float] = field(default_factory=lambda: [1.0, 0.9, 0.75, 0.5, 0.25])

    max_chars_candidates: List[int] = field(
        default_factory=lambda: [3000, 3400, 3800, 4200, 4600, 5000, 5200, 5400, 5600, 5800, 6200, 6600, 7000]
    )


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    chunk_cache: ChunkCacheConfig = field(default_factory=ChunkCacheConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    bi_train: BiEncoderTrainConfig = field(default_factory=BiEncoderTrainConfig)
    ce_train: CrossEncoderTrainConfig = field(default_factory=CrossEncoderTrainConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
