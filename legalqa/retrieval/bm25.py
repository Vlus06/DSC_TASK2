from __future__ import annotations

import pickle
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Plus

from ..legal_metadata import LegalMetadataExtractor
from ..tokenization import VietnameseTokenizer
from ..utils import logger


class BM25Retriever:
    """Document-level BM25+ retriever with optional number-boosting.

    Number-boosting re-weights candidates whose document text shares
    numeric literals (fees, deadlines, percentages, ...) with the query,
    since Vietnamese legal QA frequently hinges on exact numbers that
    plain BM25 term weighting can under-value.
    """

    def __init__(
        self,
        tokenizer: VietnameseTokenizer,
        doc_id_to_passage: dict[str, str],
        number_boost_factor: float = 0.4,
    ):
        self.tokenizer = tokenizer
        self.doc_id_to_passage = doc_id_to_passage
        self.number_boost_factor = number_boost_factor
        self.doc_ids: List[str] = []
        self.index: Optional[BM25Plus] = None

    @classmethod
    def from_cache(
        cls,
        cache_path: str,
        tokenizer: VietnameseTokenizer,
        doc_id_to_passage: dict[str, str],
        number_boost_factor: float = 0.4,
    ) -> "BM25Retriever":
        with open(cache_path, "rb") as f:
            doc_ids, index = pickle.load(f)
        obj = cls(tokenizer, doc_id_to_passage, number_boost_factor)
        obj.doc_ids = list(doc_ids)
        obj.index = index
        logger.info(f"Loaded BM25 index from cache: {len(obj.doc_ids)} documents.")
        return obj

    def build(self, max_chars: int = 4000) -> "BM25Retriever":
        self.doc_ids = list(self.doc_id_to_passage.keys())
        tokenized_corpus = [
            self.tokenizer.tokenize_clean(self.doc_id_to_passage[d], max_chars=max_chars) for d in self.doc_ids
        ]
        self.index = BM25Plus(tokenized_corpus)
        logger.info(f"Built BM25 index from scratch: {len(self.doc_ids)} documents.")
        return self

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump((self.doc_ids, self.index), f)

    def _number_boost(self, query: str, doc_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        query_numbers = set(LegalMetadataExtractor.extract_numbers(query))
        if not query_numbers:
            return doc_scores
        boosted = []
        for doc_id, score in doc_scores:
            doc_numbers = set(LegalMetadataExtractor.extract_numbers(self.doc_id_to_passage.get(doc_id, "")))
            matches = query_numbers & doc_numbers
            if matches:
                score = score * (1 + self.number_boost_factor * len(matches))
            boosted.append((doc_id, score))
        return boosted

    def retrieve(self, query: str, top_k: int = 400) -> List[Tuple[str, float]]:
        assert self.index is not None, "BM25 index not built/loaded yet."
        tokenized_query = self.tokenizer.tokenize_clean(query)
        scores = self.index.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        raw_scores = [(self.doc_ids[i], float(scores[i])) for i in top_idx]
        return sorted(self._number_boost(query, raw_scores), key=lambda x: x[1], reverse=True)

    def retrieve_doc_ids_only(self, query: str, top_k: int = 60) -> List[str]:
        """Cheaper variant used by hard-negative mining (doesn't need scores)."""
        tokenized_query = self.tokenizer.tokenize_clean(query)
        scores = self.index.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.doc_ids[i] for i in top_idx]