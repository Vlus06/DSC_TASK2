from __future__ import annotations

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..config import ChunkCacheConfig
from ..corpus import Corpus
from ..legal_metadata import LegalMetadataExtractor
from ..utils import free_memory, get_device, logger

_DIEU_SPLIT_RE = re.compile(r"(?=Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:])")
_CHUNK_PREFIX_SPLIT_RE = re.compile(r"^(.{0,200}?\.)\n(.*)$", re.DOTALL)

# Same boilerplate-line filters used everywhere else in the pipeline.
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\s*$", re.IGNORECASE),
    re.compile(r"^\s*Độc\s*lập\s*[-–]\s*Tự\s*do\s*[-–]\s*Hạnh\s*phúc\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–_]{3,}\s*$"),
    re.compile(r"^\s*Chương\s+[IVXLCDM\d]+\s*[.:]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Mục\s+\d+\s*[.:]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:BỘ|CHÍNH\s+PHỦ|QUỐC\s+HỘI)[^\n]{0,60}$"),
    re.compile(r"^\s*Số\s*[:\.]?\s*[\dA-Za-zĐđ/\-]+\s*$"),
]

# Fixes lines that got broken mid-word / mid-sentence by the source PDF/HTML
# extraction (a single trailing hyphen or a lowercase-continuation newline).
_LINE_JOIN_HYPHEN_RE = re.compile(r"-\n(?=[a-zàáâãèéêìíòóôõùúăđĩũơư])")
_LINE_JOIN_LOWER_CONTINUATION_RE = re.compile(r"\n(?=[a-zàáâãèéêìíòóôõùúăđĩũơư])")
_LIST_MARKER_FIX_RE = re.compile(r"\n(?=[a-zđ]\)\s)")


def strip_boilerplate_lines(text: str) -> str:
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if any(pat.match(stripped) for pat in _BOILERPLATE_LINE_PATTERNS):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def fix_broken_lines(text: str) -> str:
    """Rejoin lines that were broken mid-word/mid-sentence by extraction."""
    text = _LINE_JOIN_HYPHEN_RE.sub("", text)
    text = _LINE_JOIN_LOWER_CONTINUATION_RE.sub(" ", text)
    return text


def split_chunk_prefix_and_body(
    chunk_text: str, has_prefix: bool = True
) -> Tuple[str, str]:
    if not has_prefix:
        return "", chunk_text
    m = _CHUNK_PREFIX_SPLIT_RE.match(chunk_text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", chunk_text


@dataclass
class DenseChunkCache:
    """In-memory + on-disk representation of the corpus, chunked & embedded."""

    chunk_doc_ids_all: List[str] = field(default_factory=list)
    chunk_texts_all: List[str] = field(default_factory=list)
    chunk_vecs_all: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_doc_ids_all = list(self.chunk_doc_ids_all)
        if self.chunk_vecs_all is not None:
            self.chunk_vecs_all = np.ascontiguousarray(self.chunk_vecs_all, dtype=np.float32)
        self._chunk_doc_ids_arr = np.array(self.chunk_doc_ids_all)
        self.doc_to_chunk_positions: Dict[str, List[int]] = defaultdict(list)
        for i, did in enumerate(self.chunk_doc_ids_all):
            self.doc_to_chunk_positions[did].append(i)

    @property
    def chunk_doc_ids_arr(self) -> np.ndarray:
        return self._chunk_doc_ids_arr

    def save(self, path: str) -> None:
        payload = {
            "chunk_doc_ids_all": self.chunk_doc_ids_all,
            "chunk_texts_all": self.chunk_texts_all,
            "chunk_vecs_all": self.chunk_vecs_all,
            "meta": self.meta,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved dense chunk cache ({len(self.chunk_texts_all)} chunks) to {path}")

    @classmethod
    def load(cls, path: str) -> "DenseChunkCache":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            chunk_doc_ids_all=payload["chunk_doc_ids_all"],
            chunk_texts_all=payload["chunk_texts_all"],
            chunk_vecs_all=payload["chunk_vecs_all"],
            meta=payload.get("meta", {}),
        )
        logger.info(f"Loaded dense chunk cache ({len(obj.chunk_texts_all)} chunks) from {path}")
        return obj


class DenseChunkCacheBuilder:
    """Builds a `DenseChunkCache` from a `Corpus` + bi-encoder.

    Chunking strategy (mirrors the `chunk_logic_version` recorded in the
    original notebooks' cache meta,
    ``v4_1_docname_dieutitle_boilerplatefilter_keepgiaithich_linejoinfix_listmarkerfix``):

    1. Split each document on `Điều N` boundaries.
    2. Fix line breaks introduced by PDF/HTML extraction.
    3. Strip boilerplate lines (letterhead, chapter/section-only lines...)
       while explicitly keeping "Giải thích từ ngữ" (definitions) sections.
    4. Prefix each chunk with "<doc name>. <Điều title>." so the chunk is
       self-contained once retrieved.
    5. Merge/greedily pack Điều-level pieces into chunks up to
       `chunk_max_chars` (derived from the encoder's max_seq_length via
       `chars_per_token_init * safety_margin`), never splitting a single
       Điều across chunks unless it alone exceeds the budget.
    """

    def __init__(self, corpus: Corpus, config: ChunkCacheConfig, encode_max_seq_length: int = 2048):
        self.corpus = corpus
        self.config = config
        self.chunk_max_chars = config.chunk_max_chars or int(
            encode_max_seq_length * config.chars_per_token_init * config.safety_margin
        )

    def _chunk_document(self, doc_id: str, passage: str) -> List[str]:
        cfg = self.config
        doc_name = self.corpus.get_name(doc_id) or ""
        doc_title_line = LegalMetadataExtractor.extract_doc_title_line(passage) or doc_name

        parts = [p for p in _DIEU_SPLIT_RE.split(passage) if p.strip()]
        if len(parts) <= 1:
            parts = [passage]

        pieces: List[str] = []
        for part in parts:
            text = part
            if cfg.enable_line_join_fix:
                text = fix_broken_lines(text)
            if cfg.enable_boilerplate_filter:
                # Explicitly keep "Giải thích từ ngữ" (definitions) sections --
                # the filter only removes letterhead/section-marker lines, never
                # substantive Điều content.
                text = strip_boilerplate_lines(text)
            text = text.strip()
            if not text:
                continue

            dieu_info = LegalMetadataExtractor.extract_dieu_info(text)
            dieu_title = dieu_info.dieu_title

            prefix_bits = []
            if cfg.enable_doc_name_prefix and doc_title_line:
                prefix_bits.append(doc_title_line.rstrip("."))
            if cfg.enable_dieu_title_prefix and dieu_title:
                prefix_bits.append(dieu_title.rstrip("."))
            prefix = (". ".join(prefix_bits) + ".\n") if prefix_bits else ""

            body = prefix + text
            # Greedy packing: split overly long single-Điều pieces at
            # sentence-ish boundaries so no chunk exceeds the budget.
            while len(body) > self.chunk_max_chars:
                cut = body.rfind("\n", 0, self.chunk_max_chars)
                if cut < self.chunk_max_chars * 0.5:
                    cut = self.chunk_max_chars
                pieces.append(body[:cut].strip())
                body = (prefix + body[cut:].strip()) if prefix else body[cut:].strip()
            if body.strip():
                pieces.append(body.strip())
        return pieces

    def build(self, bi_encoder, show_progress: bool = True) -> DenseChunkCache:
        chunk_doc_ids_all: List[str] = []
        chunk_texts_all: List[str] = []

        items = self.corpus.doc_id_to_passage.items()
        iterator = tqdm(items, total=len(self.corpus), desc="Chunking corpus") if show_progress else items
        for doc_id, passage in iterator:
            for piece in self._chunk_document(doc_id, passage):
                chunk_doc_ids_all.append(doc_id)
                chunk_texts_all.append(piece)

        logger.info(f"Chunked corpus into {len(chunk_texts_all)} chunks (max_chars={self.chunk_max_chars}).")

        vecs = bi_encoder.encode(
            chunk_texts_all,
            batch_size=self.config.encode_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        free_memory()

        meta = {
            "model_name": getattr(bi_encoder, "model_card_data", None) and None,
            "chars_per_token_init": self.config.chars_per_token_init,
            "safety_margin": self.config.safety_margin,
            "chunk_max_chars": self.chunk_max_chars,
            "enable_doc_name_prefix": self.config.enable_doc_name_prefix,
            "enable_dieu_title_prefix": self.config.enable_dieu_title_prefix,
            "enable_boilerplate_filter": self.config.enable_boilerplate_filter,
            "enable_line_join_fix": self.config.enable_line_join_fix,
            "chunk_logic_version": "v4_1_docname_dieutitle_boilerplatefilter_keepgiaithich_linejoinfix_listmarkerfix",
            "n_docs": len(self.corpus),
        }
        return DenseChunkCache(
            chunk_doc_ids_all=chunk_doc_ids_all,
            chunk_texts_all=chunk_texts_all,
            chunk_vecs_all=vecs,
            meta=meta,
        )


class DenseRetriever:
    """Query-time dense retrieval over a pre-built `DenseChunkCache`."""

    def __init__(self, bi_encoder, cache: DenseChunkCache, max_seq_length: int = 2048, query_cache_max: int = 2000):
        self.bi_encoder = bi_encoder
        self.bi_encoder.max_seq_length = max_seq_length
        self.cache = cache
        self._query_vec_cache: Dict[str, np.ndarray] = {}
        self._query_cache_max = query_cache_max
        self.has_prefix = bool(cache.meta.get("enable_doc_name_prefix", True)) or bool(
            cache.meta.get("enable_dieu_title_prefix", True)
        )

    def get_query_vec(self, query: str) -> np.ndarray:
        if query in self._query_vec_cache:
            return self._query_vec_cache[query]
        vec = self.bi_encoder.encode(
            [query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        if len(self._query_vec_cache) >= self._query_cache_max:
            self._query_vec_cache.clear()
        self._query_vec_cache[query] = vec
        return vec

    def retrieve_doc_level(self, query: str, top_k: int = 250) -> List[Tuple[str, float]]:
        """Best chunk-level similarity per document -> ranked doc list."""
        query_vec = self.get_query_vec(query)[0]
        sims = self.cache.chunk_vecs_all @ query_vec
        doc_best: Dict[str, float] = {}
        for pos, sim in enumerate(sims):
            did = str(self.cache.chunk_doc_ids_arr[pos])
            if did not in doc_best or sim > doc_best[did]:
                doc_best[did] = float(sim)
        ranked = sorted(doc_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        del sims
        return ranked

    def retrieve_doc_ids_only(self, query: str, top_k: int = 60) -> List[str]:
        """Cheap variant used by hard-negative mining (no query-vec caching)."""
        q_vec = self.bi_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype(
            np.float32
        )
        sims = self.cache.chunk_vecs_all @ q_vec
        doc_best: Dict[str, float] = {}
        for pos, sim in enumerate(sims):
            did = str(self.cache.chunk_doc_ids_arr[pos])
            if did not in doc_best or sim > doc_best[did]:
                doc_best[did] = float(sim)
        ranked = sorted(doc_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [d for d, _ in ranked]

    def get_all_scored_chunks(
        self, question: str, doc_ids: List[str]
    ) -> List[Tuple[str, str, int, float]]:
        """Score every chunk belonging to `doc_ids` against the question.

        Returns list of (doc_id, chunk_text, local_pos_in_doc, score).
        """
        all_candidate_chunks = []
        for doc_id in doc_ids:
            positions = self.cache.doc_to_chunk_positions.get(doc_id, [])
            for local_pos, global_idx in enumerate(positions):
                all_candidate_chunks.append((doc_id, self.cache.chunk_texts_all[global_idx], local_pos, global_idx))

        if not all_candidate_chunks:
            return []

        query_vec = self.get_query_vec(question)[0]
        global_indices = np.array([c[3] for c in all_candidate_chunks])
        cand_vecs = self.cache.chunk_vecs_all[global_indices]
        sims = cand_vecs @ query_vec

        return [
            (doc_id, text, pos, float(s))
            for (doc_id, text, pos, _gi), s in zip(all_candidate_chunks, sims)
        ]

    @staticmethod
    def filter_scored_chunks_by_docs(
        scored_chunks_full: List[Tuple[str, str, int, float]], doc_id_subset: List[str]
    ) -> List[Tuple[str, str, int, float]]:
        doc_id_set = set(doc_id_subset)
        return [c for c in scored_chunks_full if c[0] in doc_id_set]

    def split_chunk_prefix_and_body(self, chunk_text: str) -> Tuple[str, str]:
        return split_chunk_prefix_and_body(chunk_text, has_prefix=self.has_prefix)