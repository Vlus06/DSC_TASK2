"""Central configuration for the LegalQA pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import CACHE_DIR, OUTPUT_DIR


def resolve_cache_file(
    cache_dir: Path,
    primary_name: str,
    legacy_names: Iterable[str] = (),
) -> Path:
    """Return an existing cache path, or the preferred path for a new cache."""
    cache_dir = Path(cache_dir)
    for name in (primary_name, *legacy_names):
        path = cache_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return cache_dir / primary_name


@dataclass(frozen=True)
class PipelineSettings:
    seed: int = 42

    bm25_top_k: int = 400
    dense_doc_top_k: int = 250
    fusion_top_k: int = 30
    final_top_docs: int = 3
    pre_ce_top_k: int = 10
    ce_top_k: int = 5

    bi_encoder_model: str = "AITeamVN/Vietnamese_Embedding"
    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    bi_encoder_max_seq_length: int = 2048
    reranker_max_length: int = 512
    ce_batch_size: int = 32
    max_chars: int = 5800

    model_seeds: tuple = (42, 10042, 20042, 30042, 40042)
    n_estimators: int = 350
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: int = 5
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 2.0
    reg_alpha: float = 0.0

    bm25_cache_name: str = "bm25_index.pkl"
    dense_parent_cache_name: str = "parent_embeddings.pkl"
    child_cache_name: str = "child_embeddings.pkl"
    corpus_cache_name: str = "corpus_metadata.pkl"
    top_documents_cache_name: str = "top_documents.pkl"
    candidate_audit_cache_name: str = "candidate_audit.pkl"
    pair_features_cache_name: str = "pair_features.pkl"
    singleton_features_cache_name: str = "singleton_features.pkl"

    pair_features_cache_version: str = "legalqa_pair_features_v1"
    singleton_features_cache_version: str = "legalqa_singleton_features_v1"

    expected_docs: int = 8532
    expected_parent_docs: int = 8512
    expected_parent_chunks: int = 738506
    expected_child_chunks: int = 595597
    expected_child_docs: int = 7078

    @property
    def child_cache_names(self) -> tuple:
        return (self.child_cache_name,)

    @staticmethod
    def cache_dir() -> Path:
        return CACHE_DIR

    @staticmethod
    def output_dir() -> Path:
        return OUTPUT_DIR
