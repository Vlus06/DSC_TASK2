import json
import pickle
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
    # The base parent/child builders only need passages and names. Prefer the
    # packed file produced by this exact loader so an H100 is not held while
    # opening 8,532 small JSON files through a remote Volume.
    cache_dir = task2_dir.parent / "cache"
    packed_path = next(
        (
            path for path in (cache_dir / "corpus_metadata.pkl",)
            if path.is_file() and path.stat().st_size > 0
        ),
        None,
    )
    if packed_path is not None:
        print(f"Loading packed base corpus: {packed_path}", flush=True)
        with packed_path.open("rb") as f:
            packed = pickle.load(f)
        if isinstance(packed, dict):
            passages, names = packed.get("passages"), packed.get("names")
        elif isinstance(packed, (tuple, list)) and len(packed) == 2:
            passages, names = packed
        else:
            raise RuntimeError(f"Unexpected base packed corpus schema: {packed_path}")
        if not isinstance(passages, dict) or not isinstance(names, dict):
            raise RuntimeError(f"Packed base corpus is missing passages/names: {packed_path}")
        passages = {str(k): v for k, v in passages.items()}
        names = {str(k): v for k, v in names.items()}
        if len(passages) != 8532 or len(names) != 8532:
            raise RuntimeError(
                f"Packed base corpus size mismatch: passages={len(passages)}, names={len(names)}"
            )
        print(f"Packed base corpus loaded: {len(passages)} docs", flush=True)
        return passages, names

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
