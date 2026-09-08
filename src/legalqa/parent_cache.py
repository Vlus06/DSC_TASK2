import hashlib
import re
from pathlib import Path

MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
ENABLE_DOC_NAME_PREFIX = True
ENABLE_DIEU_TITLE_PREFIX = True
ENABLE_BOILERPLATE_FILTER = True
ENABLE_LINE_JOIN_FIX = True
CHUNK_LOGIC_VERSION = "v4_1_docname_dieutitle_boilerplatefilter_keepgiaithich_linejoinfix_listmarkerfix"
CHARS_PER_TOKEN_INIT = 2.94
SAFETY_MARGIN = 0.85

_VN_LOWER = "a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
_TRAILING_SPACE_BEFORE_NL = re.compile(r"[ \t]+\n")
_PARA_BREAK_PLACEHOLDER = "\uE000"
_SINGLE_NL_MIDWORD = re.compile(
    r"(?<=[^\.\:\;\)\-\uE000])\n"
    r"(?=[" + _VN_LOWER + r"])"
    r"(?!\s*[a-zđ]\)\s)"
)

DIEU_PATTERN = re.compile(r"(?=Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:])")
DIEU_TITLE_PATTERN = re.compile(
    r"^\s*(Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:]\s*[^\n]{0,120})"
)
KHOAN_PATTERN = re.compile(r"(?=(?:^|\n)\s*\d+[\.\)]\s)")
BOILERPLATE_TITLE_PATTERN = re.compile(
    r"^\s*Điều\s+\d+[a-zA-Z]?\s*[\.:]\s*"
    r"(Phạm\s+vi\s+điều\s+chỉnh|Đối\s+tượng\s+áp\s+dụng)",
    re.IGNORECASE,
)


def expected_cache_filename() -> str:
    raw = "|".join([
        MODEL_NAME,
        str(CHARS_PER_TOKEN_INIT),
        str(SAFETY_MARGIN),
        str(ENABLE_DOC_NAME_PREFIX),
        str(ENABLE_DIEU_TITLE_PREFIX),
        str(ENABLE_BOILERPLATE_FILTER),
        str(ENABLE_LINE_JOIN_FIX),
        CHUNK_LOGIC_VERSION,
    ])
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"dense_chunk_index_{MODEL_NAME.replace('/', '_')}_{h}.pkl"


def join_broken_lines(text: str) -> str:
    if not text:
        return text
    text = _TRAILING_SPACE_BEFORE_NL.sub("\n", text)
    text = text.replace("\n\n", _PARA_BREAK_PLACEHOLDER)
    text = _SINGLE_NL_MIDWORD.sub(" ", text)
    text = text.replace(_PARA_BREAK_PLACEHOLDER, "\n\n")
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ")
    text = re.sub(r"[ \t\x0b\x0c]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if ENABLE_LINE_JOIN_FIX:
        text = join_broken_lines(text)
    return text


def _split_by_khoan(text: str, max_chars: int):
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
                final_parts.append(p[i:i + max_chars])
    return final_parts


def _extract_dieu_title(dieu_part: str) -> str:
    m = DIEU_TITLE_PATTERN.match(dieu_part)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def chunk_by_dieu_khoan(passage: str, doc_id: str, doc_name: str = "", max_chars: int = 5117):
    passage = clean_text(passage)
    if not passage:
        return []

    dieu_parts = [p for p in DIEU_PATTERN.split(passage) if p.strip()]
    if len(dieu_parts) <= 1:
        paras = [p for p in passage.split("\n\n") if p.strip()]
        dieu_parts = _split_by_khoan(passage, max_chars) if len(paras) <= 1 else paras

    chunks_raw = []
    for part in dieu_parts:
        part = part.strip()
        if not part:
            continue
        dieu_title = _extract_dieu_title(part) if ENABLE_DIEU_TITLE_PREFIX else ""
        if ENABLE_BOILERPLATE_FILTER and BOILERPLATE_TITLE_PATTERN.match(part):
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
        if ENABLE_DOC_NAME_PREFIX and doc_name:
            prefix_parts.append(doc_name)
        if ENABLE_DIEU_TITLE_PREFIX and dieu_title and not text.startswith(dieu_title):
            prefix_parts.append(dieu_title)
        final_text = ". ".join(prefix_parts) + ".\n" + text if prefix_parts else text
        result.append({"doc_id": doc_id, "chunk_id": f"{doc_id}_{i}", "text": final_text})
    return result


def cache_meta(n_docs: int, chunk_max_chars: int = 5117):
    return {
        "model_name": MODEL_NAME,
        "chars_per_token_init": CHARS_PER_TOKEN_INIT,
        "safety_margin": SAFETY_MARGIN,
        "chunk_max_chars": chunk_max_chars,
        "enable_doc_name_prefix": ENABLE_DOC_NAME_PREFIX,
        "enable_dieu_title_prefix": ENABLE_DIEU_TITLE_PREFIX,
        "enable_boilerplate_filter": ENABLE_BOILERPLATE_FILTER,
        "enable_line_join_fix": ENABLE_LINE_JOIN_FIX,
        "chunk_logic_version": CHUNK_LOGIC_VERSION,
        "n_docs": n_docs,
    }
