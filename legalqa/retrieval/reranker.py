from __future__ import annotations

from typing import List, Tuple

from ..utils import logger


class CrossEncoderReranker:
    """Wraps a `sentence_transformers.CrossEncoder` for chunk reranking."""

    def __init__(self, model_name: str, device: str, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name, device=self.device, trust_remote_code=True, max_length=self.max_length
            )
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load cross-encoder ({e}) -- reranking will be skipped.")
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def rerank(
        self,
        question: str,
        scored_chunks: List[Tuple[str, str, int, float]],
        top_k_candidates: int,
        split_prefix_fn=None,
    ) -> List[Tuple[str, str, int, float, float]]:
        """Rerank the top `top_k_candidates` (by embedding score) chunks.

        Returns list of (doc_id, text, pos, emb_score, ce_score).
        """
        if not self.is_available or not scored_chunks:
            return []
        candidates = sorted(scored_chunks, key=lambda x: -x[3])[:top_k_candidates]
        pairs = []
        for doc_id, text, pos, _emb_score in candidates:
            if split_prefix_fn is not None:
                _, real_body = split_prefix_fn(text)
            else:
                real_body = text
            pairs.append((question, real_body or text))
        ce_scores = self.model.predict(pairs, show_progress_bar=False)
        return [
            (doc_id, text, pos, float(emb_score), float(ce_s))
            for (doc_id, text, pos, emb_score), ce_s in zip(candidates, ce_scores)
        ]