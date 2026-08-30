from __future__ import annotations

import os
from typing import Iterable, List, Optional, Set


class VietnameseTokenizer:
    """Thin wrapper around `underthesea.word_tokenize` with stopword filtering.

    Centralizes the `vi_tokenize_clean` / `token_overlap_score` /
    `lexical_overlap_score` helpers that were duplicated across the
    original mining and QA-pipeline notebooks.
    """

    def __init__(self, stopwords_file: Optional[str] = None):
        self.stopwords: Set[str] = set()
        if stopwords_file and os.path.exists(stopwords_file):
            with open(stopwords_file, "r", encoding="utf-8") as f:
                raw = f.read().splitlines()
            self.stopwords = {w.replace(" ", "_") for w in raw}

    def tokenize_clean(self, text: str, max_chars: Optional[int] = None) -> List[str]:
        if max_chars:
            text = text[:max_chars]
        try:
            from underthesea import word_tokenize

            tokens = word_tokenize(text, format="text").split()
        except Exception:
            tokens = text.split()
        return [t.lower() for t in tokens if t.lower() not in self.stopwords]

    def token_overlap_score(self, a_tokens_set: Set[str], text_b: str, max_chars: int = 2500) -> float:
        if not a_tokens_set:
            return 0.0
        b_tokens = set(self.tokenize_clean(text_b, max_chars=max_chars))
        if not b_tokens:
            return 0.0
        return len(a_tokens_set & b_tokens) / max(1, len(a_tokens_set))

    def lexical_overlap_score(self, question_tokens_set: Set[str], chunk_text: str, max_chars: int = 1500) -> float:
        # Same formula, kept as a separate name to mirror the original code's
        # two call-sites (mining uses `max_chars=2500`, QA pipeline uses 1500).
        return self.token_overlap_score(question_tokens_set, chunk_text, max_chars=max_chars)