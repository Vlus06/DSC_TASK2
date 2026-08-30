from __future__ import annotations

from typing import List, Optional, Tuple

from ..tokenization import VietnameseTokenizer


class ChunkSelector:
    """Selects which scored chunks to keep for the final answer:
    threshold-filter, then (if too many survive) break ties with lexical
    overlap against the question so near-duplicate embedding scores don't
    get resolved arbitrarily.
    """

    def __init__(self, tokenizer: VietnameseTokenizer, lex_tiebreak_margin: float = 0.02):
        self.tokenizer = tokenizer
        self.lex_tiebreak_margin = lex_tiebreak_margin

    def _lexical_tiebreak_sort(
        self, scored_chunks: List[Tuple[str, str, int, float]], question: str
    ) -> List[Tuple[str, str, int, float]]:
        if not scored_chunks:
            return scored_chunks
        q_tokens = set(self.tokenizer.tokenize_clean(question))

        ordered = sorted(scored_chunks, key=lambda x: -x[3])
        result = []
        i, n = 0, len(ordered)
        while i < n:
            j = i + 1
            band_top_score = ordered[i][3]
            while j < n and (band_top_score - ordered[j][3]) <= self.lex_tiebreak_margin:
                j += 1
            band = ordered[i:j]
            if len(band) > 1:
                band = sorted(
                    band,
                    key=lambda c: (c[3], self.tokenizer.lexical_overlap_score(q_tokens, c[1])),
                    reverse=True,
                )
            result.extend(band)
            i = j
        return result

    def apply_threshold(
        self,
        scored_chunks: List[Tuple[str, str, int, float]],
        threshold: float,
        max_total_chunks: int = 8,
        min_kept_fallback: int = 1,
        question: Optional[str] = None,
        use_lexical_tiebreak: bool = True,
    ) -> List[Tuple[str, str, int, float]]:
        if not scored_chunks:
            return []
        kept = [c for c in scored_chunks if c[3] >= threshold]
        if not kept:
            ordered = sorted(scored_chunks, key=lambda x: -x[3])
            return ordered[:min_kept_fallback]

        if max_total_chunks and len(kept) > max_total_chunks:
            if question is not None and use_lexical_tiebreak:
                kept = self._lexical_tiebreak_sort(kept, question)[:max_total_chunks]
            else:
                kept = sorted(kept, key=lambda x: -x[3])[:max_total_chunks]
        return kept