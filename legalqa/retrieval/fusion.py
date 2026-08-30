from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

_RECENCY_KEYWORDS = re.compile(r"hiện\s+hành|mới\s+nhất|còn\s+hiệu\s+lực", re.IGNORECASE)
_YEAR_IN_NAME_RE = re.compile(r"-((?:19|20)\d{2})-")


class RecencyBooster:
    """Boosts documents from more recent years when the query asks for
    the "current" / "latest" / "still in effect" version of a regulation.
    """

    def __init__(self, doc_id_to_name: Dict[str, str], max_boost: float = 0.10, base_year: int = 2015):
        self.doc_id_to_name = doc_id_to_name
        self.max_boost = max_boost
        self.base_year = base_year

    @staticmethod
    def _get_doc_years(name_slug: str) -> List[int]:
        return [int(y) for y in _YEAR_IN_NAME_RE.findall(name_slug or "")]

    def apply(self, query: str, doc_scores: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not _RECENCY_KEYWORDS.search(query):
            return list(doc_scores)
        boosted = []
        for doc_id, score in doc_scores:
            years = self._get_doc_years(self.doc_id_to_name.get(doc_id, ""))
            if years:
                latest_year = max(years)
                boost = min(self.max_boost, max(0.0, (latest_year - self.base_year) * 0.01))
                score = score * (1 + boost)
            boosted.append((doc_id, score))
        return boosted


class ScoreFusion:
    """Min-max normalizes and linearly combines ranked candidate lists
    (e.g. BM25 doc scores + Dense doc scores) into a single fused ranking.
    """

    def __init__(self, recency_booster: RecencyBooster | None = None):
        self.recency_booster = recency_booster

    @staticmethod
    def minmax_normalize(items_scores) -> Dict[str, float]:
        items = list(items_scores.items()) if isinstance(items_scores, dict) else list(items_scores)
        if not items:
            return {}
        scores = [s for _, s in items]
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-12:
            return {did: 0.5 for did, _ in items}
        return {did: (s - lo) / (hi - lo) for did, s in items}

    def weighted_fusion(
        self, ranked_lists: Sequence[Sequence[Tuple[str, float]]], weights: Sequence[float] | None = None
    ) -> List[Tuple[str, float]]:
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        fused_scores: Dict[str, float] = defaultdict(float)
        for ranked, w in zip(ranked_lists, weights):
            norm = self.minmax_normalize(ranked)
            for did, s in norm.items():
                fused_scores[did] += w * s
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def fuse(
        self,
        bm25_cands: Sequence[Tuple[str, float]],
        dense_cands: Sequence[Tuple[str, float]],
        question: str,
        w_bm25: float,
        w_dense: float,
        top_n: int,
        use_recency_boost: bool = True,
    ) -> List[str]:
        b = self.recency_booster.apply(question, bm25_cands) if (use_recency_boost and self.recency_booster) else bm25_cands
        d = self.recency_booster.apply(question, dense_cands) if (use_recency_boost and self.recency_booster) else dense_cands
        fused = self.weighted_fusion([b, d], weights=[w_bm25, w_dense])
        return [did for did, _ in fused[:top_n]]