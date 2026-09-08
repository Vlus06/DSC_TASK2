from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TASK2_DIR = DATA_DIR / "TASK2"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"


def resolve_context_dir(task2_dir: Path = TASK2_DIR) -> Path:
    candidates = [task2_dir / "selected-contexts" / "selected-contexts", task2_dir / "selected-contexts"]
    for p in candidates:
        if p.is_dir() and any(p.rglob("*.json")):
            return p
    raise FileNotFoundError(f"Cannot locate selected-contexts under {task2_dir}")


@dataclass(frozen=True)
class PipelineConfig:
    # Exact validation split used by the notebooks.
    seed: int = 42
    val_size: int = 1000

    # Retrieval.
    bm25_top_k: int = 400
    dense_top_k: int = 250
    n_docs_sweep_max: int = 30
    best_w_bm25: float = 0.5
    best_w_dense: float = 1.0

    # Final 0.57x configuration from the full-corpus hierarchical run.
    final_top_n_docs: int = 3
    ce_top_k_candidates: int = 10
    ce_margin: float = 0.5
    ce_max_keep: int = 2
    post_process_max_chars: int = 5800
    use_automerge: bool = False

    # Kept because v11 fixed the rehydration/output bugs around the same pipeline.
    automerge_margin: float = 0.08
    automerge_min_siblings: int = 2

    bi_encoder_model: str = "AITeamVN/Vietnamese_Embedding"
    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    bi_encoder_max_seq_length: int = 2048
    reranker_max_length: int = 512

    bm25_cache_name: str = "bm25_index_stopword.pkl"
    dense_parent_cache_name: str = "dense_chunk_index_AITeamVN_Vietnamese_Embedding_48ba5ddb4b.pkl"
    child_cache_name: str = "child_chunk_vecs_FULL_corpus.pkl"
    raw_cands_cache_name: str = "raw_cands_cache.pkl"
    val_top_docs_cache_name: str = "val_top_docs.pkl"

    # Regression guard: the original full-corpus notebook reported 0.5780.
    expected_meteor_min: float = 0.5700
    expected_meteor_max: float = 0.5799

    @property
    def bm25_cache(self): return CACHE_DIR / self.bm25_cache_name
    @property
    def dense_parent_cache(self): return CACHE_DIR / self.dense_parent_cache_name
    @property
    def child_cache(self): return CACHE_DIR / self.child_cache_name
    @property
    def raw_cands_cache(self): return CACHE_DIR / self.raw_cands_cache_name
    @property
    def val_top_docs_cache(self): return CACHE_DIR / self.val_top_docs_cache_name
