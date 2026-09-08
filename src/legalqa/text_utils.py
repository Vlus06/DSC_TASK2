import re
from typing import Iterable, List, Set


def load_stopwords(path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {w.replace(" ", "_") for w in f.read().splitlines() if w.strip()}


def vi_tokenize_clean(text: str, stopwords: Set[str], max_chars=None) -> List[str]:
    if max_chars:
        text = text[:max_chars]
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text, format="text").split()
    except Exception:
        tokens = text.split()
    return [t.lower() for t in tokens if t.lower() not in stopwords]


def extract_numbers(text: str):
    return re.findall(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?", text)
