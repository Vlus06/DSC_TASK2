"""Filesystem paths and base-cache settings."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TASK2_DIR = DATA_DIR / "TASK2"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"


def resolve_context_dir(task2_dir: Path = TASK2_DIR) -> Path:
    candidates = (
        task2_dir / "selected-contexts" / "selected-contexts",
        task2_dir / "selected-contexts",
    )
    for path in candidates:
        if path.is_dir() and any(path.rglob("*.json")):
            return path
    raise FileNotFoundError(f"Cannot locate selected-contexts under {task2_dir}")


@dataclass(frozen=True)
class BaseCacheSettings:
    seed: int = 42
    bi_encoder_model: str = "AITeamVN/Vietnamese_Embedding"
    bi_encoder_max_seq_length: int = 2048
    bm25_cache_name: str = "bm25_index.pkl"
    dense_parent_cache_name: str = "parent_embeddings.pkl"
    child_cache_name: str = "child_embeddings.pkl"

    @property
    def bm25_cache(self) -> Path:
        return CACHE_DIR / self.bm25_cache_name

    @property
    def dense_parent_cache(self) -> Path:
        return CACHE_DIR / self.dense_parent_cache_name

    @property
    def child_cache(self) -> Path:
        return CACHE_DIR / self.child_cache_name
