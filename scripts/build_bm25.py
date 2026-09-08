#!/usr/bin/env python3
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rank_bm25 import BM25Plus
from legalqa.config import PipelineConfig, STOPWORDS_FILE
from legalqa.corpus import load_corpus
from legalqa.text_utils import load_stopwords, vi_tokenize_clean

cfg = PipelineConfig()
passages, _ = load_corpus()
stopwords = load_stopwords(STOPWORDS_FILE)
doc_ids = sorted(passages)
corpus_tokens = [vi_tokenize_clean(passages[d], stopwords) for d in doc_ids]
index = BM25Plus(corpus_tokens)
cfg.bm25_cache.parent.mkdir(parents=True, exist_ok=True)
with cfg.bm25_cache.open("wb") as f:
    pickle.dump((doc_ids, index), f)
print(f"Saved {cfg.bm25_cache} ({len(doc_ids)} docs)")
