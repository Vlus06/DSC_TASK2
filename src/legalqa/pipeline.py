import re
from collections import defaultdict

import numpy as np

from .child_cache import load_or_build_child_cache

_CHUNK_PREFIX_SPLIT_RE = re.compile(r"^(.{0,200}?\.)\n(.*)$", re.DOTALL)

BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\s*$", re.IGNORECASE),
    re.compile(r"^\s*Độc\s*lập\s*[-–]\s*Tự\s*do\s*[-–]\s*Hạnh\s*phúc\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–_]{3,}\s*$"),
    re.compile(r"^\s*Chương\s+[IVXLCDM\d]+\s*[.:]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Mục\s+\d+\s*[.:]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:BỘ|CHÍNH\s+PHỦ|QUỐC\s+HỘI)[^\n]{0,60}$"),
    re.compile(r"^\s*Số\s*[:\.]?\s*[\dA-Za-zĐđ/\-]+\s*$"),
]


def split_chunk_prefix_and_body(chunk_text):
    m = _CHUNK_PREFIX_SPLIT_RE.match(chunk_text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", chunk_text


def strip_boilerplate_lines(text):
    lines = text.split("\n")
    kept = [l for l in lines if not (l.strip() and any(p.match(l.strip()) for p in BOILERPLATE_LINE_PATTERNS))]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_vbqppl(text):
    patterns = [
        r'(Nghị\s+định\s+\d+/\d{4}/NĐ-CP)',
        r'(Thông\s+tư\s+liên\s+tịch\s+\d+/\d{4}/TTLT-[A-ZĐa-zđ\-]+)',
        r'(Thông\s+tư\s+\d+/\d{4}/TT-[A-ZĐa-zđ]+)',
        r'(Quyết\s+định\s+\d+/\d{4}/QĐ-[A-ZĐa-zđ]+)',
        r'(Nghị\s+quyết\s+\d+/\d{4}/(?:NQ|QH)[A-ZĐa-zđ-]*)',
        r'((?:Bộ\s+luật|Luật)\s+(?!này\b|đó\b|hiện\s+hành\b)[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯa-zđàáâãèéêìíòóôõùúăđĩũơư,\s]{2,55}?\d{4})',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def extract_dieu_info(text):
    dieu_numbers = list(dict.fromkeys(re.findall(r"Điều\s+(\d+)", text)))
    khoan_numbers = list(dict.fromkeys(re.findall(r"[Kk]hoản\s+(\d+)", text)))
    title_m = re.search(r"Điều\s+\d+[.:\s]+([^\n]{5,100})", text)
    dieu_title = title_m.group(1).strip() if title_m and not title_m.group(1)[0].isdigit() else ""
    return {"dieu_numbers": dieu_numbers, "khoan_numbers": khoan_numbers, "dieu_title": dieu_title}


def build_header(meta, question):
    vbqppl = meta["vbqppl"]
    dieu_nums = meta["dieu_numbers"]
    khoan_nums = meta["khoan_numbers"]
    dieu_title = meta["dieu_title"]
    khoan_str = "khoản " + ", khoản ".join(khoan_nums[:3]) + " " if khoan_nums and len(khoan_nums) <= 3 else ""
    dieu_str = f"Điều {dieu_nums[0]}" if dieu_nums else ""
    if dieu_str and vbqppl and dieu_title:
        return f"Căn cứ {khoan_str}{dieu_str} {vbqppl} quy định về {dieu_title} như sau:"
    if dieu_str and vbqppl:
        return f"Căn cứ {khoan_str}{dieu_str} {vbqppl} quy định như sau:"
    if vbqppl:
        return f"Căn cứ {vbqppl} quy định như sau:"
    return f"Theo quy định của pháp luật về {question.rstrip('?').strip()} như sau:"


def build_conclusion(question, meta):
    q_clean = question.rstrip("?").strip()
    if meta["dieu_numbers"] and meta["vbqppl"]:
        ref = f" tại Điều {meta['dieu_numbers'][0]} {meta['vbqppl']}"
    elif meta["vbqppl"]:
        ref = f" tại {meta['vbqppl']}"
    else:
        ref = ""
    return f"Như vậy, {q_clean} được quy định{ref}."


def post_process(answer, max_chars=5800):
    if not answer:
        return answer
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\n[ \t]+", "\n", answer)
    if len(answer) > max_chars:
        answer = answer[:max_chars]
        lp = answer.rfind(".")
        if lp > max_chars * 0.7:
            answer = answer[:lp + 1]
    return answer.strip()


def build_answer_from_kept(question, kept, top1_passage="", max_chars=5800):
    if not kept:
        return ""
    by_doc = defaultdict(list)
    doc_best = {}
    for doc_id, text, pos, score in kept:
        prefix, body = split_chunk_prefix_and_body(text)
        by_doc[doc_id].append((pos, body, score, prefix))
        doc_best[doc_id] = max(doc_best.get(doc_id, -1e9), score)
    primary_doc = max(doc_best.items(), key=lambda x: x[1])[0]

    body_parts = []
    for doc_id in sorted(by_doc.keys(), key=lambda d: -doc_best[d]):
        chunks_sorted = sorted(by_doc[doc_id], key=lambda x: -x[2])
        bodies = [strip_boilerplate_lines(t) for _, t, _, _ in chunks_sorted if t.strip()]
        body_parts.append("\n...\n".join(bodies))
    body = strip_boilerplate_lines("\n\n".join(bp for bp in body_parts if bp.strip()))

    primary_text = "\n...\n".join(t for _, t, _, _ in sorted(by_doc[primary_doc], key=lambda x: -x[2]))
    vbqppl = extract_vbqppl(top1_passage[:1500]) or extract_vbqppl(primary_text)
    meta = {**extract_dieu_info(primary_text), "vbqppl": vbqppl}
    return post_process(
        f"{build_header(meta, question)}\n{body}\n{build_conclusion(question, meta)}",
        max_chars=max_chars,
    )


class LegalQAPipeline:
    """Frozen clean implementation of the full-corpus hierarchical run that reported METEOR 0.5780."""

    def __init__(self, cfg, retriever, doc_id_to_passage, doc_id_to_name, device):
        self.cfg = cfg
        self.retriever = retriever
        self.doc_id_to_passage = doc_id_to_passage
        self.doc_id_to_name = doc_id_to_name

        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder(
            cfg.reranker_model,
            device=device,
            trust_remote_code=True,
            max_length=cfg.reranker_max_length,
        )

        self.child_texts, self.child_vecs, self.child_meta = self._load_or_build_child_cache()
        self.doc_to_child_positions = defaultdict(list)
        for i, m in enumerate(self.child_meta):
            self.doc_to_child_positions[str(m["doc_id"])].append(i)

    def _load_or_build_child_cache(self):
        return load_or_build_child_cache(
            self.cfg, self.doc_id_to_passage, self.doc_id_to_name,
            self.retriever.bi_encoder,
        )

    def get_candidate_pool(self, question, top_docs):
        query_vec = self.retriever.get_query_vec(question)[0]
        pool = []
        for doc_id in top_docs:
            parent_positions = self.retriever.doc_to_chunk_positions.get(str(doc_id), [])
            if parent_positions:
                idxs = np.array(parent_positions)
                sims = self.retriever.chunk_vecs_all[idxs] @ query_vec
                for gi, s in zip(idxs, sims):
                    pool.append((str(doc_id), self.retriever.chunk_texts_all[gi], float(s), "parent", None))

            child_positions = self.doc_to_child_positions.get(str(doc_id), [])
            if child_positions:
                idxs = np.array(child_positions)
                sims = self.child_vecs[idxs] @ query_vec
                for gi, s in zip(idxs, sims):
                    m = self.child_meta[gi]
                    pool.append((str(doc_id), self.child_texts[gi], float(s), "child", (str(doc_id), m["dieu_num"])))
        return pool

    def ce_score_candidates(self, question, candidates):
        if not candidates:
            return []
        pairs = []
        for _doc_id, text, _score, _kind, _dieu_key in candidates:
            _, body = split_chunk_prefix_and_body(text)
            pairs.append((question, body or text))
        ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        return [
            (doc_id, text, score, float(ce), kind, dieu_key)
            for (doc_id, text, score, kind, dieu_key), ce in zip(candidates, ce_scores)
        ]

    def predict_one(self, question, top_docs_full, n_docs=None, ce_topk=None, margin=None, max_keep=None, max_chars=None, **_):
        n_docs = self.cfg.final_top_n_docs if n_docs is None else n_docs
        ce_topk = self.cfg.ce_top_k_candidates if ce_topk is None else ce_topk
        margin = self.cfg.ce_margin if margin is None else margin
        max_keep = self.cfg.ce_max_keep if max_keep is None else max_keep
        max_chars = self.cfg.post_process_max_chars if max_chars is None else max_chars

        top_docs = top_docs_full[:n_docs]
        if not top_docs:
            return ""

        pool = self.get_candidate_pool(question, top_docs)
        if not pool:
            return ""
        candidates = sorted(pool, key=lambda x: -x[2])[:ce_topk]
        reranked = sorted(self.ce_score_candidates(question, candidates), key=lambda x: -x[3])
        if not reranked:
            return ""

        top1_ce = reranked[0][3]
        kept = [c for c in reranked if (top1_ce - c[3]) <= margin]
        if len(kept) < 1:
            kept = reranked[:1]
        elif len(kept) > max_keep:
            kept = reranked[:max_keep]

        kept_final = [(doc_id, text, 0, ce_s) for doc_id, text, _emb_s, ce_s, _kind, _dieu_key in kept]
        top1_passage = self.doc_id_to_passage.get(kept_final[0][0], "") if kept_final else ""
        return build_answer_from_kept(question, kept_final, top1_passage=top1_passage, max_chars=max_chars)
