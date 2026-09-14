"""Reference inference-only post-processing for the accepted V87 public submission.

IMPORTANT:
- Do NOT call this inside training target generation.
- Do NOT change retrieval/ranker/action selection.
- Call postprocess_v87_final_answer() only after the final action has been selected
  and build_answer() has produced the raw answer.
"""
from __future__ import annotations

import re

QH_CANONICAL_MAP = {
    '02/2016/QH14': 'Luật Tín ngưỡng, tôn giáo 2016',
    '05/2017/QH14': 'Luật Quản lý ngoại thương 2017',
    '09/2017/QH14': 'Luật Du lịch 2017',
    '10/2017/QH14': 'Luật Trách nhiệm bồi thường của Nhà nước 2017',
    '100/2015/QH13': 'Bộ luật Hình sự 2015',
    '101/2015/QH13': 'Bộ luật Tố tụng hình sự 2015',
    '105/2016/QH13': 'Luật Dược 2016',
    '107/2016/QH13': 'Luật Thuế xuất khẩu, thuế nhập khẩu 2016',
    '108/2016/QH13': 'Luật Điều ước quốc tế 2016',
    '14/2017/QH14': 'Luật Quản lý, sử dụng vũ khí, vật liệu nổ và công cụ hỗ trợ 2017',
    '15/2012/QH13': 'Luật Xử lý vi phạm hành chính 2012',
    '22/2008/QH12': 'Luật Cán bộ, công chức 2008',
    '23/2004/QH11': 'Luật Giao thông đường thủy nội địa 2004',
    '23/2008/QH12': 'Luật Giao thông đường bộ 2008',
    '28/2009/QH12': 'Luật Lý lịch tư pháp 2009',
    '29/2013/QH13': 'Luật Khoa học và công nghệ 2013',
    '38/2013/QH13': 'Luật Việc làm 2013',
    '45/2013/QH13': 'Luật Đất đai 2013',
    '45/2019/QH14': 'Bộ luật Lao động 2019',
    '46/2010/QH12': 'Luật Ngân hàng nhà nước Việt Nam 2010',
    '47/2014/QH13': 'Luật Nhập cảnh, xuất cảnh, quá cảnh, cư trú của người nước ngoài tại Việt Nam 2014',
    '49/2005/QH11': 'Luật Các công cụ chuyển nhượng 2005',
    '50/2005/QH11': 'Luật Sở hữu trí tuệ 2005',
    '50/2014/QH13': 'Luật Xây dựng 2014',
    '51/2014/QH13': 'Luật Phá sản 2014',
    '52/2014/QH13': 'Luật Hôn nhân và gia đình 2014',
    '53/2014/QH13': 'Luật Công chứng 2014',
    '54/2010/QH12': 'Luật Trọng tài thương mại 2010',
    '58/2010/QH12': 'Luật Viên chức 2010',
    '58/2014/QH13': 'Luật Bảo hiểm xã hội 2014',
    '59/2020/QH14': 'Luật Doanh nghiệp 2020',
    '60/2010/QH12': 'Luật Khoáng sản 2010',
    '60/2014/QH13': 'Luật Hộ tịch 2014',
    '61/2020/QH14': 'Luật Đầu tư 2020',
    '62/2014/QH13': 'Luật Tổ chức Tòa án nhân dân 2014',
    '63/2014/QH13': 'Luật Tổ chức Viện kiểm sát 2014',
    '64/2020/QH14': 'Luật Đầu tư theo phương thức đối tác công tư 2020',
    '65/2014/QH13': 'Luật Nhà ở 2014',
    '66/2006/QH11': 'Luật Hàng không dân dụng Việt Nam 2006',
    '66/2011/QH12': 'Luật Phòng, chống mua bán người 2011',
    '68/2020/QH14': 'Luật Cư trú 2020',
    '70/2020/QH14': 'Luật Thỏa thuận quốc tế 2020',
    '72/2020/QH14': 'Luật Bảo vệ môi trường 2020',
    '79/2015/QH13': 'Luật Thú y 2015',
    '88/2015/QH13': 'Luật Kế toán 2015',
    '91/2015/QH13': 'Bộ luật Dân sự 2015',
    '92/2015/QH13': 'Bộ luật Tố tụng dân sự 2015',
    '93/2015/QH13': 'Luật Tố tụng Hành chính 2015',
    '95/2015/QH13': 'Bộ luật Hàng hải Việt Nam 2015',
}

TAIL_PAT = re.compile(
    r"\s+(?:"
    r"là gì\??|"
    r"(?:được\s+)?(?:quy định\s+)?như thế nào\??|"
    r"(?:được\s+)?(?:quy định\s+)?ra sao\??|"
    r"(?:được\s+)?(?:quy định\s+)?thế nào\??|"
    r"(?:là\s+)?bao lâu\??|"
    r"(?:là\s+)?bao nhiêu\??|"
    r"(?:hay|không)\s+không\??"
    r")$",
    re.I,
)

SAFE_CONC_TAIL = re.compile(
    r"\s+(?:"
    r"là gì\??|"
    r"(?:được\s+)?(?:quy định\s+)?như thế nào\??|"
    r"(?:được\s+)?(?:quy định\s+)?ra sao\??|"
    r"(?:được\s+)?(?:quy định\s+)?thế nào\??"
    r")$",
    re.I,
)

LEGAL_TYPE_RE_POST = re.compile(
    r"\b(Bộ luật|Luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Nghị quyết|Pháp lệnh)\b",
    re.I,
)


def _split_answer(answer: str):
    lines = str(answer).splitlines()
    if not lines:
        return "", "", ""

    header = lines[0].strip()
    last_idx = max((i for i, x in enumerate(lines) if x.strip()), default=-1)

    conclusion = ""
    body_end = len(lines)

    if last_idx >= 0 and lines[last_idx].strip().lower().startswith("như vậy"):
        conclusion = lines[last_idx].strip()
        body_end = last_idx

    body = "\n".join(lines[1:body_end]).strip()
    return header, body, conclusion


def _rebuild_answer(header: str, body: str, conclusion: str) -> str:
    return "\n".join(
        x for x in [header.strip(), body.strip(), conclusion.strip()]
        if x.strip()
    )


# V80: recover a citation header from a clean conclusion reference.
def _v80_header(answer: str) -> str:
    h, b, c = _split_answer(answer)

    m = re.match(
        r"^Theo quy định của pháp luật về\s+(.+?)\s+như sau:$",
        h,
        re.I,
    )
    if not m:
        return answer

    topic = m.group(1).strip()

    cm = re.search(
        r"\bquy định tại\s+Điều\s+(\d+[A-Za-z]?)\s+(.+?)\.$",
        c,
        re.I,
    )
    if not cm:
        return answer

    dieu, doc = cm.groups()
    doc = doc.strip()

    # The successful V80 path did not promote dirty metadata blobs.
    if ". " in doc or len(doc) > 180:
        return answer

    return _rebuild_answer(
        f"Căn cứ Điều {dieu} {doc} quy định về {topic} như sau:",
        b,
        c,
    )


# V81: normalize legal document type and citation wording.
def _normalize_type(s: str) -> str:
    s = re.sub(
        r"(?<!Thông tư liên tịch )\b(\d+[A-Za-zĐđ]?/\d{4}/TTLT-[A-ZĐ\-]+)\b",
        r"Thông tư liên tịch \1",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"(?<!Nghị định )\b(\d+[A-Za-zĐđ]?/\d{4}/NĐ-CP)\b",
        r"Nghị định \1",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"(?<!Thông tư )(?<!Thông tư liên tịch )\b(\d+[A-Za-zĐđ]?/\d{4}/TT-[A-ZĐ\-]+)\b",
        r"Thông tư \1",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"(?<!Quyết định )\b(\d+[A-Za-zĐđ]?/\d{4}/QĐ-[A-ZĐ\-]*)\b",
        r"Quyết định \1",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"(?<!Luật )(?<!Nghị quyết )\b(\d+[A-Za-zĐđ]?/\d{4}/QH\d+)\b",
        r"Luật \1",
        s,
        flags=re.I,
    )
    return s


def _v81_header(answer: str) -> str:
    h, b, c = _split_answer(answer)
    h2 = _normalize_type(h)

    m = re.match(
        r"^Căn cứ Điều\s+(\d+[A-Za-z]?)\s+(.+?)\s+quy định về\s+(.+?)\s+như sau:$",
        h2,
        re.I,
    )
    if not m:
        return _rebuild_answer(h2, b, c)

    dieu, doc, topic = m.groups()

    # TT/TTLT and raw/unknown document metadata use "Căn cứ theo Điều".
    if "/TT-" in doc.upper() or "/TTLT-" in doc.upper() or not LEGAL_TYPE_RE_POST.search(doc):
        h2 = f"Căn cứ theo Điều {dieu} {doc} quy định về {topic} như sau:"
    else:
        h2 = f"Căn cứ theo quy định tại Điều {dieu} {doc} về {topic} như sau:"

    return _rebuild_answer(h2, b, c)


# V82: canonical QH law names.
def _canonicalize_qh_header(answer: str) -> str:
    h, b, c = _split_answer(answer)

    m = re.search(
        r"(\bĐiều\s+\d+[A-Za-z]?\s+)(.+?)(\s+(?:quy định\s+)?về\s+)",
        h,
        re.I,
    )
    if not m:
        return answer

    doc = m.group(2)
    code_match = re.search(r"\b\d+[A-Za-zĐđ]?/\d{4}/QH\d+\b", doc, re.I)

    if code_match:
        code = code_match.group(0)
        canonical = QH_CANONICAL_MAP.get(code)
        if canonical:
            h = h[:m.start(2)] + canonical + h[m.end(2):]

    return _rebuild_answer(h, b, c)


# V82: remove question-tail noise from header topic.
def _cleanup_topic(answer: str) -> str:
    h, b, c = _split_answer(answer)

    m = re.search(r"(\bvề\s+)(.+?)(\s+như sau:)$", h, re.I)
    if not m:
        return answer

    topic = m.group(2)
    new_topic = TAIL_PAT.sub("", topic).strip(" ?")

    if not new_topic or new_topic == topic:
        return answer

    h2 = h[:m.start(2)] + new_topic + h[m.end(2):]
    return _rebuild_answer(h2, b, c)


# V82: conservative khoản inference from selected evidence structure.
def _add_safe_khoan(answer: str) -> str:
    h, b, c = _split_answer(answer)

    if not re.search(r"\bĐiều\s+\d+", h, re.I):
        return answer

    if re.search(r"\bkhoản\s+\d+\s+Điều\b", h, re.I):
        return answer

    chunks = re.split(r"\n\s*\.\.\.\s*\n", b)

    k = None

    if len(chunks) >= 2:
        m = re.match(r"^\s*(\d{1,2})\.\s+", chunks[1])
        if m:
            k = m.group(1)

    if k is None:
        nums = re.findall(r"(?m)^\s*(\d{1,2})\.\s+", b)
        if len(nums) == 1:
            k = nums[0]

    if k is None:
        return answer

    h2 = re.sub(
        r"\bĐiều\s+",
        f"khoản {k} Điều ",
        h,
        count=1,
        flags=re.I,
    )
    return _rebuild_answer(h2, b, c)


# V82: remove uppercase boilerplate headings, but never remove Điều lines.
def _upper_heading(line: str) -> bool:
    s = (line or "").strip()

    if not s or len(s) < 8 or len(s) > 180:
        return False

    if re.match(r"^Điều\b", s, re.I):
        return False

    letters = [ch for ch in s if ch.isalpha()]
    if len(letters) < 10:
        return False

    return sum(ch.isupper() for ch in letters) / len(letters) >= 0.90


def _remove_upper_headings(answer: str) -> str:
    h, b, c = _split_answer(answer)
    lines = [x for x in b.splitlines() if not _upper_heading(x)]
    b2 = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return _rebuild_answer(h, b2, c)


# V83: conclusion cleanup used before V84. V87 will overwrite the final
# conclusion, but retaining this keeps the accepted sequence deterministic.
def _v83_conclusion(answer: str) -> str:
    h, b, c = _split_answer(answer)

    if not c:
        return answer

    m = re.match(
        r"^(Như vậy,\s*)(.+?)(\s+(?:được\s+)?quy định(?:\s+tại.*)?\.)$",
        c,
        re.I,
    )

    if m:
        topic = m.group(2)
        new_topic = SAFE_CONC_TAIL.sub("", topic).strip(" ?")
        if new_topic:
            c = m.group(1) + new_topic + m.group(3)

    c = re.sub(
        r"\s+được quy định tại\s+",
        " quy định tại ",
        c,
        count=1,
        flags=re.I,
    )

    return _rebuild_answer(h, b, c)


# V84:
# A) recover remaining fallback header from an article marker in the body;
# B) remove a redundant leading "Điều X." marker from body chunks when the
#    same article is already cited in the header.
def _v84_micro(answer: str) -> str:
    h, b, c = _split_answer(answer)

    m = re.match(
        r"^Theo quy định của pháp luật về\s+(.+?)\s+như sau:$",
        h,
        re.I,
    )

    bm = re.match(r"^\s*Điều\s+(\d+[A-Za-z]?)\s*[.:]", b, re.I)

    if m and bm:
        h = f"Căn cứ theo Điều {bm.group(1)} quy định về {m.group(1).strip()} như sau:"

    hm = re.search(r"\bĐiều\s+(\d+[A-Za-z]?)\b", h, re.I)

    if hm:
        art = hm.group(1).lower()
        parts = re.split(r"(\n\s*\.\.\.\s*\n)", b)
        new_parts = []

        for part in parts:
            if re.fullmatch(r"\n\s*\.\.\.\s*\n", part or ""):
                new_parts.append(part)
                continue

            pm = re.match(
                r"^\s*Điều\s+(\d+[A-Za-z]?)\s*[.:]\s*",
                part,
                re.I,
            )

            if pm and pm.group(1).lower() == art:
                part = part[pm.end():]

            new_parts.append(part)

        b = "".join(new_parts).strip()

    return _rebuild_answer(h, b, c)


# V87 = accepted public winner (0.6122):
# keep V84 header/body, replace conclusion globally with the winning template.
def _v87_conclusion(answer: str, question: str) -> str:
    h, b, _ = _split_answer(answer)
    q = str(question).strip().rstrip(" ?.").strip()
    c = f"Như vậy, theo quy định trên thì {q}."
    return _rebuild_answer(h, b, c)


def postprocess_v87_final_answer(answer: str, question: str) -> str:
    """Apply the accepted V80→V84→V87 inference-only generation path."""
    answer = _v80_header(answer)
    answer = _v81_header(answer)
    answer = _canonicalize_qh_header(answer)
    answer = _cleanup_topic(answer)
    answer = _add_safe_khoan(answer)
    answer = _remove_upper_headings(answer)
    answer = _v83_conclusion(answer)
    answer = _v84_micro(answer)
    answer = _v87_conclusion(answer, question)
    return answer
