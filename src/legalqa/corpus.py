import json
import re
from pathlib import Path
from typing import Dict, Tuple

from .config import TASK2_DIR, resolve_context_dir


def _doc_name(doc: dict) -> str:
    name = doc.get("name", "") or ""
    if name:
        return name
    link = doc.get("link", "") or ""
    if not link:
        return ""
    slug = link.rstrip("/").split("/")[-1]
    slug = re.sub(r"-\d+\.aspx$", "", slug)
    return slug.replace("-", " ")


def load_corpus(task2_dir: Path = TASK2_DIR) -> Tuple[Dict[str, str], Dict[str, str]]:
    context_dir = resolve_context_dir(task2_dir)
    doc_id_to_passage: Dict[str, str] = {}
    doc_id_to_name: Dict[str, str] = {}
    # Stable ordering makes cache generation reproducible after re-extraction.
    for fp in sorted(context_dir.rglob("*.json")):
        with fp.open(encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = str(doc["id"]) if "id" in doc else fp.stem.replace("context_", "")
        doc_id_to_passage[doc_id] = doc.get("passage", "") or ""
        doc_id_to_name[doc_id] = _doc_name(doc)
    return doc_id_to_passage, doc_id_to_name


def load_train(task2_dir: Path = TASK2_DIR):
    with (task2_dir / "train.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_public(task2_dir: Path = TASK2_DIR):
    with (task2_dir / "public-official.json").open(encoding="utf-8") as f:
        return json.load(f)
