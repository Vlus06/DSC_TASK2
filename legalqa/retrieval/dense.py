from __future__ import annotations

import hashlib
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..config import ChunkCacheConfig
from ..corpus import Corpus
from ..utils import free_memory, get_device, logger

# =============================================================================
# Chunk-prefix splitting + answer-time boilerplate stripping.
#
# These two are used AFTER retrieval, when assembling the final answer text
# (see legalqa/answer/answer_builder.py) -- they strip letterhead lines from
# the chunk BODY that gets shown to the user. This is a different mechanism
# from `BOILERPLATE_TITLE_PATTERN` below, which discards whole CHUNKS at
# cache-build time (e.g. a "Pham vi dieu chinh" Dieu is rarely useful as a
# standalone retrieval hit). Both existed independently in the original
# notebooks and are kept independently here.
# =============================================================================
_CHUNK_PREFIX_SPLIT_RE = re.compile(r"^(.{0,200}?\.)\n(.*)$", re.DOTALL)

_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\s*$", re.IGNORECASE),
    re.compile(r"^\s*Độc\s*lập\s*[-–]\s*Tự\s*do\s*[-–]\s*Hạnh\s*phúc\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–_]{3,}\s*$"),
    re.compile(r"^\s*Chương\s+[IVXLCDM\d]+\s*[.:]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Mục\s+\d+\s*[.:]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:BỘ|CHÍNH\s+PHỦ|QUỐC\s+HỘI)[^\n]{0,60}$"),
    re.compile(r"^\s*Số\s*[:\.]?\s*[\dA-Za-zĐđ/\-]+\s*$"),
]


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


def split_chunk_prefix_and_body(chunk_text: str, has_prefix: bool = True) -> Tuple[str, str]:
    if not has_prefix:
        return "", chunk_text
    m = _CHUNK_PREFIX_SPLIT_RE.match(chunk_text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", chunk_text


# =============================================================================
# join_broken_lines -- v4.1, WITH the list-marker fix.
#
# Joins single \n that got inserted mid-word/mid-sentence by the source
# extraction, WITHOUT touching real paragraph breaks (\n\n) and WITHOUT
# joining two consecutive list items ("a) ...\nb) ..."), which the v4 (no
# suffix) regex used to join incorrectly.
# =============================================================================
_VN_LOWER = "a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
_TRAILING_SPACE_BEFORE_NL = re.compile(r"[ \t]+\n")
_PARA_BREAK_PLACEHOLDER = "\uE000"
_SINGLE_NL_MIDWORD = re.compile(
    r"(?<=[^\.\:\;\)\-\uE000])\n"
    r"(?=[" + _VN_LOWER + r"])"
    r"(?!\s*[a-zđ]\)\s)"
)


def join_broken_lines(text: str) -> str:
    """Rejoin lines broken mid-word/mid-sentence by extraction, preserving
    real paragraph breaks (\\n\\n) and consecutive letter-list items
    ("a) ...\\nb) ...")."""
    if not text:
        return text
    text = _TRAILING_SPACE_BEFORE_NL.sub("\n", text)
    text = text.replace("\n\n", _PARA_BREAK_PLACEHOLDER)
    text = _SINGLE_NL_MIDWORD.sub(" ", text)
    text = text.replace(_PARA_BREAK_PLACEHOLDER, "\n\n")
    return text


_SANITY_EXAMPLES_SHOULD_JOIN = [
    "hoạt động quản lý, điều \nhành của tổ chức đại diện",
    "Trình tự cấp \nlại, điều chỉnh Giấy chứng nhận đủ điều kiện kinh doanh dược",
    "Cơ cấu tổ \nchức",
]
_SANITY_EXAMPLES_SHOULD_NOT_JOIN = [
    "a) Nội dung thứ nhất\nb) Nội dung thứ hai",
    "cấp phép cho tổ chức\nb) Điều kiện khác",
]


def sanity_check_join_broken_lines() -> bool:
    """Run before building a full-corpus cache to catch a regressed
    _SINGLE_NL_MIDWORD regex early (cheap; matches the notebook's checks)."""
    ok = True
    for ex in _SANITY_EXAMPLES_SHOULD_JOIN:
        result = "\n" not in join_broken_lines(ex)
        if not result:
            logger.warning(f"[sanity] SHOULD have joined but didn't: {ex!r}")
        ok = ok and result
    for ex in _SANITY_EXAMPLES_SHOULD_NOT_JOIN:
        result = "\n" in join_broken_lines(ex)
        if not result:
            logger.warning(f"[sanity] SHOULD NOT have joined but did: {ex!r}")
        ok = ok and result
    logger.info(f"Sanity check join_broken_lines: {'OK' if ok else 'FAIL -- check _SINGLE_NL_MIDWORD regex!'}")
    return ok


# =============================================================================
# Chunk-by-Dieu/Khoan (v4.1) -- Dieu split, falling back to paragraph split,
# falling back to Khoan split; oversized pieces are further split by Khoan
# with a sliding window; whole "Pham vi dieu chinh"/"Doi tuong ap dung"
# Dieu are dropped; each surviving chunk is prefixed with "<doc name>.
# <Dieu title>." for self-containedness.
# =============================================================================
DIEU_PATTERN = re.compile(r"(?=Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:])")
DIEU_TITLE_PATTERN = re.compile(r"^\s*(Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:]\s*[^\n]{0,120})")
KHOAN_PATTERN = re.compile(r"(?=(?:^|\n)\s*\d+[\.\)]\s)")

BOILERPLATE_TITLE_PATTERN = re.compile(
    r"^\s*Điều\s+\d+[a-zA-Z]?\s*[\.:]\s*"
    r"(Phạm\s+vi\s+điều\s+chỉnh|Đối\s+tượng\s+áp\s+dụng)",
    re.IGNORECASE,
)


def clean_text(text: str, enable_line_join_fix: bool = True) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ")
    text = re.sub(r"[ \t\x0b\x0c]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if enable_line_join_fix:
        text = join_broken_lines(text)
    return text


def _split_by_khoan(text: str, max_chars: int) -> List[str]:
    parts = [p for p in KHOAN_PATTERN.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [text]
    final_parts = []
    step = max(max_chars - 200, 1)
    for p in parts:
        if len(p) <= max_chars:
            final_parts.append(p)
        else:
            for i in range(0, len(p), step):
                final_parts.append(p[i : i + max_chars])
    return final_parts


def _extract_dieu_title(dieu_part: str) -> str:
    m = DIEU_TITLE_PATTERN.match(dieu_part)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def resolve_doc_name(name: str, link: str) -> str:
    """Doc-name resolution used for the chunk prefix: prefer the `name`
    field from the context JSON; otherwise derive a readable slug from
    `link` (strip trailing `-<id>.aspx`, replace hyphens with spaces)."""
    if name:
        return name
    if not link:
        return ""
    slug = link.rstrip("/").split("/")[-1]
    slug = re.sub(r"-\d+\.aspx$", "", slug)
    return slug.replace("-", " ")


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
    """Builds a `DenseChunkCache` from a `Corpus` + bi-encoder, following
    the authoritative v4.1 `build_dense_cache` logic 1:1:

      1. clean_text() -> join_broken_lines() (v4.1, list-marker-safe).
      2. Split on `Điều N` boundaries; if the document has no `Điều`
         markers, fall back to paragraph split (`\\n\\n`); if that also
         fails, fall back to Khoan split.
      3. Drop whole chunks matching BOILERPLATE_TITLE_PATTERN ("Phạm vi
         điều chỉnh" / "Đối tượng áp dụng" Điều).
      4. If a piece still exceeds `chunk_max_chars`, split it further by
         Khoan with a sliding window.
      5. Prefix each surviving chunk with "<doc name>. <Điều title>.\\n"
         so it's self-contained once retrieved.

    Note: unlike a naive "pack chunks up to chunk_max_chars" strategy, this
    does NOT merge multiple Điều into a single chunk -- each Điều (or
    Khoan-split piece of an oversized Điều) becomes exactly one chunk.
    """

    def __init__(self, corpus: Corpus, config: ChunkCacheConfig):
        self.corpus = corpus
        self.config = config
        self.chunk_max_chars: Optional[int] = config.chunk_max_chars  # resolved once the model's max_seq_length is known

    def _resolve_chunk_max_chars(self, bi_encoder) -> int:
        cfg = self.config
        if cfg.chunk_max_chars:
            return cfg.chunk_max_chars
        try:
            model_max_seq = bi_encoder.max_seq_length
        except Exception:
            model_max_seq = None
        if not model_max_seq or model_max_seq > 8192:
            model_max_seq = 256
            logger.warning(f"Could not read a valid max_seq_length from model -- falling back to {model_max_seq} tokens.")
        chunk_max_chars = int(model_max_seq * cfg.chars_per_token_init * cfg.safety_margin)
        logger.info(
            f"max_seq_length={model_max_seq} tokens -> chunk_max_chars="
            f"{chunk_max_chars} chars (chars_per_token_init={cfg.chars_per_token_init})."
        )
        return chunk_max_chars

    def _doc_name(self, doc_id: str) -> str:
        return resolve_doc_name(self.corpus.get_name(doc_id), self.corpus.get_link(doc_id))

    def _chunk_document(self, doc_id: str, passage: str) -> List[dict]:
        cfg = self.config
        max_chars = self.chunk_max_chars
        doc_name = self._doc_name(doc_id) if cfg.enable_doc_name_prefix else ""

        passage = clean_text(passage, enable_line_join_fix=cfg.enable_line_join_fix)
        if not passage:
            return []

        dieu_parts = [p for p in DIEU_PATTERN.split(passage) if p.strip()]
        if len(dieu_parts) <= 1:
            paras = [p for p in passage.split("\n\n") if p.strip()]
            if len(paras) <= 1:
                dieu_parts = _split_by_khoan(passage, max_chars)
            else:
                dieu_parts = paras

        chunks_raw: List[Tuple[str, str]] = []
        for part in dieu_parts:
            part = part.strip()
            if not part:
                continue

            dieu_title = _extract_dieu_title(part) if cfg.enable_dieu_title_prefix else ""

            if cfg.enable_boilerplate_filter and BOILERPLATE_TITLE_PATTERN.match(part):
                continue

            if len(part) <= max_chars:
                chunks_raw.append((part, dieu_title))
            else:
                for sub in _split_by_khoan(part, max_chars):
                    chunks_raw.append((sub, dieu_title))

        result = []
        for i, (text, dieu_title) in enumerate(chunks_raw):
            text = text.strip()
            if not text:
                continue

            prefix_parts = []
            if cfg.enable_doc_name_prefix and doc_name:
                prefix_parts.append(doc_name)
            if cfg.enable_dieu_title_prefix and dieu_title and not text.startswith(dieu_title):
                prefix_parts.append(dieu_title)

            final_text = (". ".join(prefix_parts) + ".\n" + text) if prefix_parts else text
            result.append({"doc_id": doc_id, "chunk_id": f"{doc_id}_{i}", "text": final_text})
        return result

    def build(self, bi_encoder, show_progress: bool = True) -> DenseChunkCache:
        if not sanity_check_join_broken_lines():
            logger.warning("join_broken_lines sanity check FAILED -- proceeding anyway, but inspect the regex first.")

        self.chunk_max_chars = self._resolve_chunk_max_chars(bi_encoder)

        chunk_doc_ids_all: List[str] = []
        chunk_texts_all: List[str] = []

        items = self.corpus.doc_id_to_passage.items()
        iterator = tqdm(items, total=len(self.corpus), desc="Chunking corpus (v4.1)") if show_progress else items
        for doc_id, passage in iterator:
            for piece in self._chunk_document(doc_id, passage):
                chunk_doc_ids_all.append(piece["doc_id"])
                chunk_texts_all.append(piece["text"])

        logger.info(f"Chunked corpus into {len(chunk_texts_all)} chunks (chunk_max_chars={self.chunk_max_chars}).")

        vecs = bi_encoder.encode(
            chunk_texts_all,
            batch_size=self.config.encode_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        free_memory()

        meta = {
            "chars_per_token_init": self.config.chars_per_token_init,
            "safety_margin": self.config.safety_margin,
            "chunk_max_chars": self.chunk_max_chars,
            "enable_doc_name_prefix": self.config.enable_doc_name_prefix,
            "enable_dieu_title_prefix": self.config.enable_dieu_title_prefix,
            "enable_boilerplate_filter": self.config.enable_boilerplate_filter,
            "enable_line_join_fix": self.config.enable_line_join_fix,
            "chunk_logic_version": self.chunk_logic_version(),
            "n_docs": len(self.corpus),
        }
        return DenseChunkCache(
            chunk_doc_ids_all=chunk_doc_ids_all,
            chunk_texts_all=chunk_texts_all,
            chunk_vecs_all=vecs,
            meta=meta,
        )

    def chunk_logic_version(self) -> str:
        base = "v4_1_docname_dieutitle_boilerplatefilter_keepgiaithich_linejoinfix_listmarkerfix"
        suffix = self.config.chunk_logic_version_suffix
        return f"{base}{suffix}" if suffix else base

    def default_cache_path(self, cache_dir: str, model_name: str) -> str:
        """Hash-versioned cache filename -- identical scheme to the
        notebook's `make_cache_path()`, so changing the model or any
        cfg flag never silently collides with a different cache."""
        cfg = self.config
        raw = "|".join(
            [
                model_name,
                str(cfg.chars_per_token_init),
                str(cfg.safety_margin),
                str(cfg.enable_doc_name_prefix),
                str(cfg.enable_dieu_title_prefix),
                str(cfg.enable_boilerplate_filter),
                str(cfg.enable_line_join_fix),
                self.chunk_logic_version(),
            ]
        )
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
        safe_model = model_name.replace("/", "_")
        return f"{cache_dir.rstrip('/')}/dense_chunk_index_{safe_model}_{h}.pkl"


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

    def get_all_scored_chunks(self, question: str, doc_ids: List[str]) -> List[Tuple[str, str, int, float]]:
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
            (doc_id, text, pos, float(s)) for (doc_id, text, pos, _gi), s in zip(all_candidate_chunks, sims)
        ]

    @staticmethod
    def filter_scored_chunks_by_docs(
        scored_chunks_full: List[Tuple[str, str, int, float]], doc_id_subset: List[str]
    ) -> List[Tuple[str, str, int, float]]:
        doc_id_set = set(doc_id_subset)
        return [c for c in scored_chunks_full if c[0] in doc_id_set]

    def split_chunk_prefix_and_body(self, chunk_text: str) -> Tuple[str, str]:
        return split_chunk_prefix_and_body(chunk_text, has_prefix=self.has_prefix)
