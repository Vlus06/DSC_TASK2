from __future__ import annotations


class MeteorScorer:
    """Thin wrapper around NLTK's METEOR score (the official metric used
    by the DSC2026 Task-2 organizers), with lazy NLTK-data bootstrap.
    """

    def __init__(self):
        self._ensure_nltk_data()
        from nltk.translate.meteor_score import meteor_score

        self._meteor_score = meteor_score

    @staticmethod
    def _ensure_nltk_data() -> None:
        import nltk

        for pkg in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
            try:
                nltk.data.find(pkg)
            except LookupError:
                try:
                    nltk.download(pkg.split("/")[-1], quiet=True)
                except Exception:
                    pass

    def score(self, pred: str, gold: str) -> float:
        if not pred or not gold:
            return 0.0
        try:
            return float(self._meteor_score([gold.split()], pred.split()))
        except Exception:
            return 0.0

    def score_batch(self, preds: list[str], golds: list[str]) -> list[float]:
        return [self.score(p, g) for p, g in zip(preds, golds)]

    def mean_score(self, preds: list[str], golds: list[str]) -> float:
        scores = self.score_batch(preds, golds)
        return sum(scores) / max(1, len(scores))