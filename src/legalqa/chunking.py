import re

DIEU_PATTERN = re.compile(r"(?=Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:])")
DIEU_TITLE_PATTERN = re.compile(r"^\s*(Điều\s+\d+[a-zA-Z]?\s*(?:\([^)]*\))?\s*[\.:]\s*[^\n]{0,120})")
KHOAN_TOPLEVEL_PATTERN = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+")


def extract_dieu_title(dieu_part: str) -> str:
    m = DIEU_TITLE_PATTERN.match(dieu_part)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def split_dieu_into_khoan(dieu_body: str):
    matches = list(KHOAN_TOPLEVEL_PATTERN.finditer(dieu_body))
    if len(matches) < 2:
        return None
    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(dieu_body)
        parts.append((m.group(1), dieu_body[start:end].strip()))
    return parts


def build_hierarchical_chunks_for_doc(doc_id, passage, doc_name):
    if not passage:
        return []
    spans = [m.start() for m in DIEU_PATTERN.finditer(passage)]
    if not spans:
        return []
    spans.append(len(passage))
    children = []
    for i in range(len(spans) - 1):
        dieu_text = passage[spans[i]:spans[i + 1]].strip()
        if not dieu_text:
            continue
        m_num = re.match(r"Điều\s+(\d+[a-zA-Z]?)", dieu_text)
        dieu_num = m_num.group(1) if m_num else f"pos{i}"
        dieu_title = extract_dieu_title(dieu_text)
        khoan_parts = split_dieu_into_khoan(dieu_text)
        if khoan_parts is None:
            continue
        for khoan_num, khoan_text in khoan_parts:
            prefix_bits = []
            if doc_name:
                prefix_bits.append(doc_name)
            if dieu_title:
                prefix_bits.append(dieu_title)
            prefix_bits.append(f"Khoản {khoan_num}")
            final_text = ". ".join(prefix_bits) + ".\n" + khoan_text
            children.append({
                "doc_id": doc_id,
                "dieu_num": dieu_num,
                "khoan_num": khoan_num,
                "text": final_text,
                "raw_khoan_text": khoan_text,
            })
    return children
