from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional, Tuple

from ..legal_metadata import LegalMetadata, LegalMetadataExtractor
from ..retrieval.dense import split_chunk_prefix_and_body, strip_boilerplate_lines

_VN_DIACRITIC_CHARS = set(
    "àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
)


def has_vn_diacritics(text: str) -> bool:
    return any(c in _VN_DIACRITIC_CHARS for c in text)


class HeaderConclusionBuilder:
    """Builds the "Căn cứ ..." header and "Như vậy ..." conclusion sentences
    that wrap the retrieved legal text, based on extracted metadata.
    """

    @staticmethod
    def build_header(meta: LegalMetadata, question: str, doc_title_line: str = "") -> str:
        vbqppl = meta.vbqppl
        dieu_nums = meta.dieu_numbers
        khoan_nums = meta.khoan_numbers
        dieu_title = meta.dieu_title

        khoan_str = ""
        if khoan_nums and len(khoan_nums) <= 3:
            khoan_str = "khoản " + ", khoản ".join(khoan_nums[:3]) + " "

        dieu_str = f"Điều {dieu_nums[0]}" if dieu_nums else ""

        if dieu_str and vbqppl and dieu_title:
            return f"Căn cứ {khoan_str}{dieu_str} {vbqppl} quy định về {dieu_title} như sau:"
        if dieu_str and vbqppl:
            return f"Căn cứ {khoan_str}{dieu_str} {vbqppl} quy định như sau:"
        if vbqppl:
            return f"Căn cứ {vbqppl} quy định như sau:"
        if doc_title_line:
            return f"Căn cứ {doc_title_line} quy định như sau:"
        q_clean = question.rstrip("?").strip()
        return f"Theo quy định của pháp luật về {q_clean} như sau:"

    @staticmethod
    def build_conclusion(question: str, meta: LegalMetadata) -> str:
        q_clean = question.rstrip("?").strip()
        vbqppl = meta.vbqppl
        dieu_nums = meta.dieu_numbers

        ref_str = ""
        if dieu_nums and vbqppl:
            ref_str = f" tại Điều {dieu_nums[0]} {vbqppl}"
        elif vbqppl:
            ref_str = f" tại {vbqppl}"

        q_lower = question.lower()
        if "có được" in q_lower or "có phải" in q_lower or "có cần" in q_lower:
            return f"Như vậy, {q_clean} được quy định{ref_str}."
        if "bao nhiêu" in q_lower or "bao lâu" in q_lower or "mấy" in q_lower:
            return f"Như vậy, {q_clean} được quy định cụ thể{ref_str} như trên."
        if "là gì" in q_lower or "thế nào" in q_lower or "ra sao" in q_lower:
            return f"Theo đó, {q_clean} được quy định{ref_str} như trên."
        if "gồm" in q_lower or "bao gồm" in q_lower or "những gì" in q_lower:
            return f"Như vậy, {q_clean} được quy định{ref_str} như trên."
        if "khi nào" in q_lower or "trường hợp nào" in q_lower:
            return f"Theo đó, {q_clean} được quy định{ref_str} như trên."
        return f"Như vậy, {q_clean} được quy định{ref_str}."


def post_process(answer: str, max_chars: int) -> str:
    if not answer:
        return answer
    answer = re.sub(r'\(Note:.*?\)', '', answer, flags=re.DOTALL)
    answer = re.sub(
        r'\b(?:The|Here|However|Note|Vietnamese|Chinese|corrected|version|translated|back|sentence|'
        r'mistakenly|written|final|part|response|above|language|requested|provided)\b',
        '',
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r'\[\s*\.\.\.\s*\]', '\n...\n', answer)
    answer = re.sub(r'\.{4,}', '...', answer)
    answer = re.sub(r'([^\n]{4,80})\n\.\.\.\n\1(?=\n|\.\.\.|$)', r'\1', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = re.sub(r'[ \t]{2,}', ' ', answer)
    answer = re.sub(r'\n[ \t]+', '\n', answer)
    if len(answer) > max_chars:
        answer = answer[:max_chars]
        last_period = answer.rfind('.')
        if last_period > max_chars * 0.7:
            answer = answer[: last_period + 1]
    return answer.strip()


def dedupe_overlapping_lines(chunk_bodies: List[str]) -> List[str]:
    seen_lines = set()
    deduped_bodies = []
    for body in chunk_bodies:
        lines = body.split("\n")
        kept_lines = []
        for line in lines:
            norm = line.strip()
            if norm and norm in seen_lines:
                continue
            if norm:
                seen_lines.add(norm)
            kept_lines.append(line)
        deduped_bodies.append("\n".join(kept_lines).strip())
    return [b for b in deduped_bodies if b]


class AnswerBuilder:
    """Assembles the final answer text from a list of kept (doc_id, text,
    pos, score) chunks: header + deduped/boilerplate-stripped body +
    conclusion, then post-processes and truncates to `max_chars`.
    """

    def __init__(self, doc_id_to_passage: dict, has_chunk_prefix: bool = True):
        self.doc_id_to_passage = doc_id_to_passage
        self.has_chunk_prefix = has_chunk_prefix

    def _split_prefix(self, text: str):
        return split_chunk_prefix_and_body(text, has_prefix=self.has_chunk_prefix)

    def build(
        self,
        question: str,
        kept_chunks: List[Tuple[str, str, int, float]],
        top1_full_passage: Optional[str] = None,
        max_chars: int = 5800,
        use_dedupe: bool = False,
    ) -> str:
        if not kept_chunks:
            return ""

        by_doc: dict = defaultdict(list)
        doc_prefixes: dict = defaultdict(list)
        doc_best_score: dict = {}
        for doc_id, text, pos, score in kept_chunks:
            prefix, real_body = self._split_prefix(text)
            by_doc[doc_id].append((pos, real_body, score))
            if prefix and prefix not in doc_prefixes[doc_id]:
                doc_prefixes[doc_id].append(prefix)
            doc_best_score[doc_id] = max(doc_best_score.get(doc_id, -1e9), score)

        primary_doc_id = max(doc_best_score.items(), key=lambda x: x[1])[0]

        body_parts = []
        for doc_id in sorted(by_doc.keys(), key=lambda d: -doc_best_score[d]):
            chunks_sorted = sorted(by_doc[doc_id], key=lambda x: -x[2])
            stripped_bodies = [strip_boilerplate_lines(t) for _, t, _ in chunks_sorted if t.strip()]
            if use_dedupe:
                stripped_bodies = dedupe_overlapping_lines(stripped_bodies)
            body_parts.append("\n...\n".join(stripped_bodies))

        body = "\n\n".join(bp for bp in body_parts if bp.strip())
        body = strip_boilerplate_lines(body)

        primary_doc_real_body = "\n...\n".join(t for _, t, _ in sorted(by_doc[primary_doc_id], key=lambda x: -x[2]))
        primary_doc_prefix_text = " ".join(doc_prefixes.get(primary_doc_id, []))
        primary_doc_kept_text = (primary_doc_prefix_text + "\n" + primary_doc_real_body).strip()

        title_passage = self.doc_id_to_passage.get(primary_doc_id, "") if top1_full_passage is None else top1_full_passage
        meta = LegalMetadataExtractor.extract_legal_metadata_from_kept_text(
            primary_doc_kept_text, title_passage=title_passage
        )

        prefix_is_usable = bool(primary_doc_prefix_text) and has_vn_diacritics(primary_doc_prefix_text)
        if not meta.vbqppl and prefix_is_usable:
            meta.vbqppl = primary_doc_prefix_text.split(".")[0].strip()

        doc_title_line = LegalMetadataExtractor.extract_doc_title_line(title_passage) if title_passage else ""
        if not doc_title_line and prefix_is_usable:
            doc_title_line = primary_doc_prefix_text.split(".")[0].strip()

        header = HeaderConclusionBuilder.build_header(meta, question, doc_title_line=doc_title_line)
        conclusion = HeaderConclusionBuilder.build_conclusion(question, meta)

        answer = f"{header}\n{body}\n{conclusion}"
        return post_process(answer, max_chars=max_chars)