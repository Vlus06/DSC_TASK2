import numpy as np
from nltk.translate.meteor_score import meteor_score
from .text_utils import vi_tokenize_clean


def compute_meteor(pred, gold):
    if not pred or not gold:
        return 0.0
    try:
        return float(meteor_score([gold.split()], pred.split()))
    except Exception:
        return 0.0


def coverage_score(kept_text, gold, stopwords):
    g = set(vi_tokenize_clean(gold, stopwords))
    if not g:
        return None
    k = set(vi_tokenize_clean(kept_text, stopwords))
    return len(g & k) / len(g)
