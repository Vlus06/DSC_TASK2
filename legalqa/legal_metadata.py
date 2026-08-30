"""
Regex-based extraction of Vietnamese legal-document metadata.

This is a faithful, class-wrapped port of the extraction logic used
throughout the original notebooks (extract_vbqppl / extract_dieu_info /
extract_doc_title_line / normalize_vbqppl_key), kept as pure functions
inside a stateless `LegalMetadataExtractor` so it can be unit tested and
reused identically by the mining stage and the QA pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


_VBQPPL_PATTERNS = [
    r'(Nghị\s+định\s+\d+/\d{4}/NĐ-CP)',
    r'(Thông\s+tư\s+liên\s+tịch\s+\d+/\d{4}/TTLT-[A-ZĐa-zđ\-]+)',
    r'(Thông\s+tư\s+\d+/\d{4}/TT-[A-ZĐa-zđ]+)',
    r'(Quyết\s+định\s+\d+/\d{4}/QĐ-[A-ZĐa-zđ]+)',
    r'(Nghị\s+quyết\s+\d+/\d{4}/(?:NQ|QH)[A-ZĐa-zđ-]*)',
    r'((?:Bộ\s+luật|Luật)\s+(?!này\b|đó\b|hiện\s+hành\b)[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ]'
    r'[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯa-zđàáâãèéêìíòóôõùúăđĩũơư,\s]{2,55}?\d{4}(?:\s*\(sửa\s*đổi[^)]*\))?)',
]
_VBQPPL_GENERIC_PATTERN = (
    r'((?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết)\s+(?:số\s+)?[\dA-Za-zĐđ/\-]{3,30})'
)
_VBQPPL_SELF_REF_PATTERN = re.compile(r'(?:Bộ\s+luật|Luật)\s+(?:này|đó|hiện\s+hành).*', re.IGNORECASE)

_DIEU_NUM_RE = re.compile(r'Điều\s+(\d+)')
_DIEU_TITLE_RE = re.compile(r'Điều\s+\d+[.:\s]+([^\n]{5,100})')
_KHOAN_RE = re.compile(r'[Kk]hoản\s+(\d+)')
_DIEM_RE = re.compile(r'[Đđ]iểm\s+([a-zđ])')

_DOC_TYPE_KEYWORDS = [
    ("THÔNG TƯ LIÊN TỊCH", "Thông tư liên tịch"),
    ("THÔNG TƯ", "Thông tư"),
    ("NGHỊ ĐỊNH", "Nghị định"),
    ("QUYẾT ĐỊNH", "Quyết định"),
    ("NGHỊ QUYẾT", "Nghị quyết"),
    ("BỘ LUẬT", "Bộ luật"),
    ("LUẬT", "Luật"),
]
_SO_HIEU_RE = re.compile(r'Số\s*[:\.]?\s*([\dA-Za-zĐđ/\-]{4,25})')
_SO_HIEU_IN_STR_RE = re.compile(r'(\d{1,4}/\d{4}/[A-ZĐa-zđ\-]+)')


@dataclass
class DieuInfo:
    dieu_numbers: List[str] = field(default_factory=list)
    dieu_title: str = ""
    khoan_numbers: List[str] = field(default_factory=list)
    diem_numbers: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dieu_numbers": self.dieu_numbers,
            "dieu_title": self.dieu_title,
            "khoan_numbers": self.khoan_numbers,
            "diem_numbers": self.diem_numbers,
        }


@dataclass
class LegalMetadata:
    vbqppl: str = ""
    dieu_numbers: List[str] = field(default_factory=list)
    dieu_title: str = ""
    khoan_numbers: List[str] = field(default_factory=list)
    diem_numbers: List[str] = field(default_factory=list)


class LegalMetadataExtractor:
    """Stateless helper bundling all legal-document regex extraction logic."""

    @staticmethod
    def extract_vbqppl(text: str) -> str:
        for pat in _VBQPPL_PATTERNS:
            match = re.search(pat, text)
            if match:
                candidate = match.group(1).strip()
                if _VBQPPL_SELF_REF_PATTERN.fullmatch(candidate):
                    continue
                return re.sub(r'\s+', ' ', candidate).strip()
        generic = re.search(_VBQPPL_GENERIC_PATTERN, text)
        if generic:
            return re.sub(r'\s+', ' ', generic.group(1).strip())
        return ""

    @staticmethod
    def extract_dieu_info(text: str) -> DieuInfo:
        result = DieuInfo()
        dieu_matches = _DIEU_NUM_RE.findall(text)
        if dieu_matches:
            result.dieu_numbers = list(dict.fromkeys(dieu_matches))
        title_match = _DIEU_TITLE_RE.search(text)
        if title_match:
            title = title_match.group(1).strip()
            if not title[0].isdigit():
                result.dieu_title = title
        khoan_matches = _KHOAN_RE.findall(text)
        if khoan_matches:
            result.khoan_numbers = list(dict.fromkeys(khoan_matches))
        diem_matches = _DIEM_RE.findall(text)
        if diem_matches:
            result.diem_numbers = list(dict.fromkeys(diem_matches))
        return result

    @staticmethod
    def extract_doc_title_line(full_passage: str) -> str:
        if not full_passage:
            return ""
        cut_pos = full_passage.find("Căn cứ")
        head = full_passage[:cut_pos] if cut_pos > 0 else full_passage[:500]
        head = head[:600]

        doc_type = ""
        for upper_kw, display_kw in _DOC_TYPE_KEYWORDS:
            if re.search(re.escape(upper_kw), head, re.IGNORECASE):
                doc_type = display_kw
                break

        so_hieu_match = _SO_HIEU_RE.search(head)
        so_hieu = so_hieu_match.group(1).strip().rstrip('.,;') if so_hieu_match else ""

        if doc_type and so_hieu:
            return f"{doc_type} {so_hieu}"
        if doc_type:
            return doc_type
        return ""

    @staticmethod
    def normalize_vbqppl_key(s: str) -> str:
        s = s.lower()
        s = re.sub(r'\s+', ' ', s).strip()
        return s.rstrip('.,;:')

    @classmethod
    def extract_legal_metadata_from_kept_text(
        cls, kept_text_for_doc: str, title_passage: Optional[str] = None
    ) -> LegalMetadata:
        primary = title_passage if title_passage else kept_text_for_doc
        vbqppl = ""
        if title_passage:
            self_title = cls.extract_doc_title_line(title_passage)
            if self_title:
                vbqppl = self_title
        if not vbqppl:
            vbqppl = cls.extract_vbqppl(primary[:1500])
        if not vbqppl:
            vbqppl = cls.extract_vbqppl(kept_text_for_doc)
        dieu_info = cls.extract_dieu_info(kept_text_for_doc)
        return LegalMetadata(
            vbqppl=vbqppl,
            dieu_numbers=dieu_info.dieu_numbers,
            dieu_title=dieu_info.dieu_title,
            khoan_numbers=dieu_info.khoan_numbers,
            diem_numbers=dieu_info.diem_numbers,
        )

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        return re.findall(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?", text)


class TitleIndex:
    """Reverse index: normalized document title -> list of doc_ids.

    Used to resolve a VBQPPL string extracted from a gold answer back to
    the concrete `doc_id` it refers to (positive mining, Stage 1).
    """

    def __init__(self) -> None:
        self.title_to_doc_ids: dict[str, list[str]] = {}
        self.doc_id_to_self_title: dict[str, str] = {}

    def build(self, doc_id_to_passage: dict[str, str]) -> "TitleIndex":
        from collections import defaultdict

        self.title_to_doc_ids = defaultdict(list)
        for did, passage in doc_id_to_passage.items():
            self_title = LegalMetadataExtractor.extract_doc_title_line(passage)
            if not self_title:
                self_title = LegalMetadataExtractor.extract_vbqppl(passage[:1500])
            self.doc_id_to_self_title[did] = self_title
            if self_title:
                key = LegalMetadataExtractor.normalize_vbqppl_key(self_title)
                self.title_to_doc_ids[key].append(did)
        return self

    def find_doc_id_by_vbqppl(self, vbqppl_from_gold: str):
        if not vbqppl_from_gold:
            return None, None
        key = LegalMetadataExtractor.normalize_vbqppl_key(vbqppl_from_gold)
        if key in self.title_to_doc_ids and len(self.title_to_doc_ids[key]) == 1:
            return self.title_to_doc_ids[key][0], "exact"

        so_hieu_match = _SO_HIEU_IN_STR_RE.search(vbqppl_from_gold)
        if so_hieu_match:
            so_hieu = so_hieu_match.group(1).lower()
            candidates = [did for did, t in self.doc_id_to_self_title.items() if so_hieu in t.lower()]
            if len(candidates) == 1:
                return candidates[0], "so_hieu_fallback"
            if len(candidates) > 1:
                for c in candidates:
                    if key.split()[0] in self.doc_id_to_self_title[c].lower():
                        return c, "so_hieu_fallback_multi"
                return candidates[0], "so_hieu_fallback_multi"
        return None, None