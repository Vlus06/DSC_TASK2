from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict

from tqdm.auto import tqdm

from .utils import logger


@dataclass
class Corpus:
    """In-memory view of the legal-document corpus (selected-contexts/*.json)."""

    doc_id_to_passage: Dict[str, str] = field(default_factory=dict)
    doc_id_to_name: Dict[str, str] = field(default_factory=dict)
    # Raw `link` field from the context JSON -- used as a fallback source
    # for the chunk-prefix doc name (see dense.resolve_doc_name) when
    # `name` is empty, matching the authoritative v4.1 build_dense_cache
    # script.
    doc_id_to_link: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.doc_id_to_passage)

    def get_passage(self, doc_id: str) -> str:
        return self.doc_id_to_passage.get(doc_id, "")

    def get_name(self, doc_id: str) -> str:
        return self.doc_id_to_name.get(doc_id, "")

    def get_link(self, doc_id: str) -> str:
        return self.doc_id_to_link.get(doc_id, "")

    @classmethod
    def load(cls, context_dir: str, show_progress: bool = True) -> "Corpus":
        context_files = glob.glob(os.path.join(context_dir, "**", "*.json"), recursive=True)
        doc_id_to_passage: Dict[str, str] = {}
        doc_id_to_name: Dict[str, str] = {}
        doc_id_to_link: Dict[str, str] = {}

        iterator = tqdm(context_files, desc="Loading contexts") if show_progress else context_files
        for fp in iterator:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
            doc_id = str(doc.get("id", os.path.splitext(os.path.basename(fp))[0].replace("context_", "")))
            doc_id_to_passage[doc_id] = doc.get("passage", "") or ""
            doc_id_to_name[doc_id] = doc.get("name", "") or ""
            doc_id_to_link[doc_id] = doc.get("link", "") or ""

        logger.info(f"Loaded corpus: {len(doc_id_to_passage)} documents from {context_dir}")
        return cls(doc_id_to_passage=doc_id_to_passage, doc_id_to_name=doc_id_to_name, doc_id_to_link=doc_id_to_link)


def load_train_json(train_path: str) -> dict:
    with open(train_path, encoding="utf-8") as f:
        return json.load(f)


def load_public_official_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
