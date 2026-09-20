"""Retrieval, ranking features and answer generation for LegalQA."""
from __future__ import annotations

import itertools
import math
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .postprocess_v87 import postprocess_best_06252
from .settings import PipelineSettings


PAIR_FEATURES = [
    'c1_ce','c1_emb','c1_ce_rank','c1_len','c1_body_len',
    'c1_q_overlap','c1_q_jaccard','c1_is_parent','c1_is_child','c1_has_dieu',
    'c2_ce','c2_emb','c2_ce_rank','c2_len','c2_body_len',
    'c2_q_overlap','c2_q_jaccard','c2_is_parent','c2_is_child','c2_has_dieu',
    'ce_sum','ce_mean','ce_gap',
    'emb_sum','emb_mean','emb_gap',
    'same_doc','same_kind','same_dieu',
    'pair_text_jaccard','pair_query_coverage',
    'incremental_q_cov_c2','incremental_q_cov_c1',
    'combined_len','len_ratio',
    'rank_sum','rank_gap',
    'contains_rank1','contains_rank2',
]
ACTION_FEATURES = PAIR_FEATURES + ['action_size']
assert len(PAIR_FEATURES) == 39
assert len(ACTION_FEATURES) == 40

TOKEN_RE = re.compile(r'\w+', re.UNICODE)
_CHUNK_PREFIX_SPLIT_RE = re.compile(r'^(.{0,200}?\.)\n(.*)$', re.DOTALL)

BOILERPLATE_LINE_PATTERNS = [
    re.compile(r'^\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\s*$', re.I),
    re.compile(r'^\s*Độc\s*lập\s*[-–]\s*Tự\s*do\s*[-–]\s*Hạnh\s*phúc\s*$', re.I),
    re.compile(r'^\s*[-–_]{3,}\s*$'),
    re.compile(r'^\s*Chương\s+[IVXLCDM\d]+\s*[.:]?\s*$', re.I),
    re.compile(r'^\s*Mục\s+\d+\s*[.:]?\s*$', re.I),
    re.compile(r'^\s*(?:BỘ|CHÍNH\s+PHỦ|QUỐC\s+HỘI)[^\n]{0,60}$'),
    re.compile(r'^\s*Số\s*[:\.]?\s*[\dA-Za-zĐđ/\-]+\s*$'),
]

LEGAL_TYPE_RE = re.compile(
    r'\b(Bộ luật|Luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Nghị quyết|Pháp lệnh)\b',
    re.I,
)
DOC_CODE_RE = re.compile(
    r'\b\d{1,4}[A-Za-zĐđ]?/\d{4}/(?:NĐ-CP|TTLT-[A-ZĐ]+|TT-[A-ZĐ]+|QĐ-[A-ZĐ]+|NQ-[A-ZĐ]+|NQ|QH\d*|UBTVQH\d*|[A-ZĐ\-]{2,})\b',
    re.I,
)


def configure_cuda(seed: int = 42) -> None:
    """Configure deterministic CUDA behavior without changing model precision."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Use the same full-float path on every supported CUDA device.
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_nltk_resources() -> None:
    import nltk
    for resource, package in [
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def meteor_local(pred: str, gold: str) -> float:
    from nltk.translate.meteor_score import meteor_score
    try:
        return float(meteor_score([gold.split()], pred.split()))
    except Exception:
        return 0.0


def vi_tokenize_clean(text: str, stopwords: set, max_chars: Optional[int] = None) -> List[str]:
    if max_chars:
        text = text[:max_chars]
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text, format='text').split()
    except Exception:
        tokens = text.split()
    return [t.lower() for t in tokens if t.lower() not in stopwords]


def extract_numbers(text: str) -> List[str]:
    # Keep this regex stable because it affects retrieval scores.
    return re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?', text or '')


def minmax_normalize(items_scores):
    items = list(items_scores.items()) if isinstance(items_scores, dict) else list(items_scores)
    if not items:
        return {}
    scores = [s for _, s in items]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {did: 0.5 for did, _ in items}
    return {did: (s - lo) / (hi - lo) for did, s in items}


def weighted_score_fusion(ranked_lists, weights):
    fused = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for did, score in minmax_normalize(ranked).items():
            fused[did] += weight * score
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def split_chunk_prefix_and_body(chunk_text: str) -> Tuple[str, str]:
    m = _CHUNK_PREFIX_SPLIT_RE.match(chunk_text or '')
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return '', chunk_text or ''


def strip_boilerplate_lines(text: str) -> str:
    lines = (text or '').split('\n')
    kept = [
        line for line in lines
        if not (line.strip() and any(p.match(line.strip()) for p in BOILERPLATE_LINE_PATTERNS))
    ]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(kept)).strip()


def extract_dieu_info(text: str) -> dict:
    dieu_numbers = list(dict.fromkeys(re.findall(r'Điều\s+(\d+[A-Za-z]?)', text or '')))
    khoan_numbers = list(dict.fromkeys(re.findall(r'[Kk]hoản\s+(\d+)', text or '')))
    title_m = re.search(r'Điều\s+\d+[A-Za-z]?[.:\s]+([^\n]{5,100})', text or '')
    dieu_title = (
        title_m.group(1).strip()
        if title_m and title_m.group(1) and not title_m.group(1)[0].isdigit()
        else ''
    )
    return {'dieu_numbers': dieu_numbers, 'khoan_numbers': khoan_numbers, 'dieu_title': dieu_title}


def post_process(answer: str, max_chars: int = 5800) -> str:
    if not answer:
        return answer
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = re.sub(r'[ \t]{2,}', ' ', answer)
    answer = re.sub(r'\n[ \t]+', '\n', answer)
    if len(answer) > max_chars:
        answer = answer[:max_chars]
        last_period = answer.rfind('.')
        if last_period > max_chars * 0.7:
            answer = answer[:last_period + 1]
    return answer.strip()


def extract_legal_document_name(text: str) -> str:
    patterns = [
        r'(Nghị\s+định\s+\d+/\d{4}/NĐ-CP)',
        r'(Thông\s+tư\s+liên\s+tịch\s+\d+/\d{4}/TTLT-[A-ZĐa-zđ\-]+)',
        r'(Thông\s+tư\s+\d+/\d{4}/TT-[A-ZĐa-zđ]+)',
        r'(Quyết\s+định\s+\d+/\d{4}/QĐ-[A-ZĐa-zđ]+)',
        r'(Nghị\s+quyết\s+\d+/\d{4}/(?:NQ|QH)[A-ZĐa-zđ-]*)',
        r'((?:Bộ\s+luật|Luật)\s+(?!này\b|đó\b|hiện\s+hành\b)[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯa-zđàáâãèéêìíòóôõùúăđĩũơư,\s]{2,55}?\d{4})',
    ]
    for pat in patterns:
        m = re.search(pat, text or '')
        if m:
            return re.sub(r'\s+', ' ', m.group(1)).strip()
    return ''


def clean_name(name: str) -> str:
    s = re.sub(r'\s+', ' ', name or '').strip(' -–|')
    s = re.sub(r'\.(?:html?|aspx?)$', '', s, flags=re.I)
    return s


def name_from_link(link: str) -> str:
    if not link:
        return ''
    slug = link.rstrip('/').split('/')[-1]
    slug = re.sub(r'\.(?:html?|aspx?)$', '', slug, flags=re.I)
    slug = re.sub(r'-\d+$', '', slug)
    slug = slug.replace('-', ' ')
    return re.sub(r'\s+', ' ', slug).strip()


def normalize_doccode_from_passage(text: str) -> str:
    t = text or ''
    t = re.sub(r'Số\s*:\s*\n+\s*', 'Số: ', t, flags=re.I)
    t = re.sub(r'(\d+)\s*/\s*(\d{4})\s*/\s*', r'\1/\2/', t)
    m = DOC_CODE_RE.search(t)
    return m.group(0).strip() if m else ''


def extract_vbqppl_metadata(
    doc_id: str,
    doc_id_to_name: Mapping[str, str],
    doc_id_to_link: Mapping[str, str],
    doc_id_to_passage: Mapping[str, str],
    prefix: str = '',
    primary_text: str = '',
) -> str:
    vals = [
        clean_name(doc_id_to_name.get(str(doc_id), '')),
        name_from_link(doc_id_to_link.get(str(doc_id), '')),
        re.sub(r'\s+', ' ', prefix or '').strip(),
        re.sub(r'\s+', ' ', doc_id_to_passage.get(str(doc_id), '')[:2000]).strip(),
        re.sub(r'\s+', ' ', primary_text[:1000]).strip(),
    ]
    vals = [x for x in vals if x]

    for s in vals[:3]:
        if LEGAL_TYPE_RE.search(s):
            code = DOC_CODE_RE.search(s)
            if code:
                typ = LEGAL_TYPE_RE.search(s)
                start = typ.start()
                end = min(len(s), code.end() + 80)
                return re.sub(r'\s+', ' ', s[start:end].strip(' ,;-'))
            if len(s) <= 180:
                return s

    for s in vals:
        z = extract_legal_document_name(s)
        if z:
            return z

    passage = doc_id_to_passage.get(str(doc_id), '')
    code = normalize_doccode_from_passage(passage[:2500])
    if code:
        joined = ' '.join(vals[:3])
        typ = LEGAL_TYPE_RE.search(joined)
        if typ:
            return f'{typ.group(1)} {code}'
        return code
    return ''


def build_legal_header(meta: Mapping, question: str) -> str:
    vbqppl = meta['vbqppl']
    dieu_nums = meta['dieu_numbers']
    khoan_nums = meta['khoan_numbers']
    dieu_title = meta['dieu_title']
    khoan_str = 'khoản ' + ', khoản '.join(khoan_nums[:3]) + ' ' if khoan_nums and len(khoan_nums) <= 3 else ''
    dieu_str = f'Điều {dieu_nums[0]}' if dieu_nums else ''
    if dieu_str and vbqppl and dieu_title:
        return f'Căn cứ {khoan_str}{dieu_str} {vbqppl} quy định về {dieu_title} như sau:'
    if dieu_str and vbqppl:
        return f'Căn cứ {khoan_str}{dieu_str} {vbqppl} quy định như sau:'
    if vbqppl:
        return f'Căn cứ {vbqppl} quy định như sau:'
    return f"Theo quy định của pháp luật về {question.rstrip('?').strip()} như sau:"


def build_conclusion(question: str, meta: Mapping) -> str:
    q_clean = question.rstrip('?').strip()
    if meta['dieu_numbers'] and meta['vbqppl']:
        ref = f" tại Điều {meta['dieu_numbers'][0]} {meta['vbqppl']}"
    elif meta['vbqppl']:
        ref = f" tại {meta['vbqppl']}"
    else:
        ref = ''
    return f'Như vậy, {q_clean} được quy định{ref}.'


def build_answer_from_candidates(
    question: str,
    kept: Sequence[Tuple],
    doc_id_to_passage: Mapping[str, str],
    doc_id_to_name: Mapping[str, str],
    doc_id_to_link: Mapping[str, str],
    max_chars: int = 5800,
) -> str:
    if not kept:
        return ''

    by_doc = defaultdict(list)
    doc_best = {}
    for doc_id, text, _emb, ce_score, _kind, _dieu_key in kept:
        prefix, body = split_chunk_prefix_and_body(text)
        by_doc[str(doc_id)].append((body, float(ce_score), text, prefix))
        doc_best[str(doc_id)] = max(doc_best.get(str(doc_id), -1e9), float(ce_score))

    primary_doc = max(doc_best, key=doc_best.get)
    body = '\n\n'.join(
        '\n...\n'.join(
            strip_boilerplate_lines(x[0])
            for x in sorted(by_doc[d], key=lambda z: -z[1])
            if x[0].strip()
        )
        for d in sorted(by_doc, key=lambda z: -doc_best[z])
    )

    xs = sorted(by_doc[primary_doc], key=lambda z: -z[1])
    raw = '\n...\n'.join(x[2] for x in xs)
    prefix = ' '.join(x[3] for x in xs if x[3])
    info = extract_dieu_info(raw)
    passage = doc_id_to_passage.get(primary_doc, '')

    document_name = extract_legal_document_name(passage[:1500]) or extract_legal_document_name(raw)
    header = build_legal_header({**info, 'vbqppl': document_name}, question)

    vb_meta = extract_vbqppl_metadata(
        primary_doc,
        doc_id_to_name,
        doc_id_to_link,
        doc_id_to_passage,
        prefix,
        raw,
    )
    conclusion = build_conclusion(question, {**info, 'vbqppl': vb_meta})
    return post_process(f'{header}\n{body}\n{conclusion}', max_chars)


def toks(text: str) -> set:
    return set(TOKEN_RE.findall((text or '').lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def overlap_recall(qset: set, tset: set) -> float:
    return len(qset & tset) / max(1, len(qset))


def parse_first_dieu(text: str) -> str:
    m = re.search(r'Điều\s+(\d+[A-Za-z]?)', text or '')
    return m.group(1) if m else ''


def candidate_features(q: str, c: Mapping) -> dict:
    qset = toks(q)
    tset = toks(c['text'])
    _, body = split_chunk_prefix_and_body(c['text'])
    return {
        'ce': c['ce'], 'emb': c['emb'], 'ce_rank': c['ce_rank'],
        'len': len(c['text']), 'body_len': len(body),
        'q_overlap': overlap_recall(qset, tset),
        'q_jaccard': jaccard(qset, tset),
        'is_parent': int(c['kind'] == 'parent'),
        'is_child': int(c['kind'] == 'child'),
        'has_dieu': int(bool(parse_first_dieu(c['text']))),
    }


def pair_features(q: str, c1: Mapping, c2: Mapping) -> dict:
    f1 = candidate_features(q, c1)
    f2 = candidate_features(q, c2)
    qset = toks(q)
    t1, t2 = toks(c1['text']), toks(c2['text'])
    d1, d2 = parse_first_dieu(c1['text']), parse_first_dieu(c2['text'])
    feat = {}
    for k, v in f1.items():
        feat['c1_' + k] = v
    for k, v in f2.items():
        feat['c2_' + k] = v
    feat.update({
        'ce_sum': c1['ce'] + c2['ce'],
        'ce_mean': (c1['ce'] + c2['ce']) / 2,
        'ce_gap': abs(c1['ce'] - c2['ce']),
        'emb_sum': c1['emb'] + c2['emb'],
        'emb_mean': (c1['emb'] + c2['emb']) / 2,
        'emb_gap': abs(c1['emb'] - c2['emb']),
        'same_doc': int(c1['doc_id'] == c2['doc_id']),
        'same_kind': int(c1['kind'] == c2['kind']),
        'same_dieu': int(bool(d1 and d2 and d1 == d2)),
        'pair_text_jaccard': jaccard(t1, t2),
        'pair_query_coverage': overlap_recall(qset, t1 | t2),
        'incremental_q_cov_c2': len((qset & t2) - t1) / max(1, len(qset)),
        'incremental_q_cov_c1': len((qset & t1) - t2) / max(1, len(qset)),
        'combined_len': len(c1['text']) + len(c2['text']),
        'len_ratio': max(len(c1['text']), len(c2['text'])) / max(1, min(len(c1['text']), len(c2['text']))),
        'rank_sum': c1['ce_rank'] + c2['ce_rank'],
        'rank_gap': abs(c1['ce_rank'] - c2['ce_rank']),
        'contains_rank1': int(1 in [c1['ce_rank'], c2['ce_rank']]),
        'contains_rank2': int(2 in [c1['ce_rank'], c2['ce_rank']]),
    })
    assert set(feat) == set(PAIR_FEATURES)
    return feat


def singleton_action_features(q: str, c: Mapping) -> dict:
    f1 = candidate_features(q, c)
    out = {
        'c1_ce': f1['ce'], 'c1_emb': f1['emb'], 'c1_ce_rank': f1['ce_rank'],
        'c1_len': f1['len'], 'c1_body_len': f1['body_len'],
        'c1_q_overlap': f1['q_overlap'], 'c1_q_jaccard': f1['q_jaccard'],
        'c1_is_parent': f1['is_parent'], 'c1_is_child': f1['is_child'], 'c1_has_dieu': f1['has_dieu'],
        'c2_ce': np.nan, 'c2_emb': np.nan, 'c2_ce_rank': np.nan,
        'c2_len': np.nan, 'c2_body_len': np.nan,
        'c2_q_overlap': np.nan, 'c2_q_jaccard': np.nan,
        'c2_is_parent': np.nan, 'c2_is_child': np.nan, 'c2_has_dieu': np.nan,
        'ce_sum': f1['ce'], 'ce_mean': f1['ce'], 'ce_gap': 0.0,
        'emb_sum': f1['emb'], 'emb_mean': f1['emb'], 'emb_gap': 0.0,
        'same_doc': np.nan, 'same_kind': np.nan, 'same_dieu': np.nan,
        'pair_text_jaccard': np.nan,
        'pair_query_coverage': f1['q_overlap'],
        'incremental_q_cov_c2': 0.0,
        'incremental_q_cov_c1': f1['q_overlap'],
        'combined_len': f1['len'], 'len_ratio': np.nan,
        'rank_sum': f1['ce_rank'], 'rank_gap': np.nan,
        'contains_rank1': int(f1['ce_rank'] == 1),
        'contains_rank2': int(f1['ce_rank'] == 2),
    }
    assert set(out) == set(PAIR_FEATURES)
    return out


def candidate_tuple_from_dict(c: Mapping) -> tuple:
    return (
        str(c['doc_id']), c['text'], float(c['emb']), float(c['ce']),
        str(c['kind']), None,
    )


def canonical_action_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['qid', 'action_size', 'action_i', 'action_j'], kind='stable').reset_index(drop=True)


def action_metadata(candidate_dicts: Sequence[Mapping], ranks: Sequence[int]) -> dict:
    selected = [candidate_dicts[int(rank) - 1] for rank in ranks]
    primary = max(selected, key=lambda candidate: float(candidate['ce']))
    return {
        'primary_doc': str(primary['doc_id']),
        'first_article': parse_first_dieu(primary['text']),
    }


def build_action_frame(qid: str, question: str, candidate_dicts: Sequence[Mapping]) -> pd.DataFrame:
    if len(candidate_dicts) != 5:
        raise RuntimeError(f'Expected five candidates, got {len(candidate_dicts)}')
    rows = []
    for rank, candidate in enumerate(candidate_dicts, start=1):
        rows.append({
            'qid': str(qid), 'action_key': f'S{rank}',
            'action_type': 'single', 'action_size': 1,
            'action_i': rank, 'action_j': 0,
            **action_metadata(candidate_dicts, [rank]),
            **singleton_action_features(question, candidate),
        })
    for i, j in itertools.combinations(range(1, 6), 2):
        rows.append({
            'qid': str(qid), 'action_key': f'P{i}_{j}',
            'action_type': 'pair', 'action_size': 2,
            'action_i': i, 'action_j': j,
            **action_metadata(candidate_dicts, [i, j]),
            **pair_features(question, candidate_dicts[i - 1], candidate_dicts[j - 1]),
        })
    frame = canonical_action_sort(pd.DataFrame(rows))
    if len(frame) != 15:
        raise RuntimeError(f'Expected 15 actions, got {len(frame)}')
    return frame


def make_ranker(cfg: PipelineSettings, random_state: int):
    import xgboost as xgb
    return xgb.XGBRanker(
        objective='rank:pairwise',
        eval_metric='ndcg',
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        tree_method='hist',
        random_state=random_state,
        n_jobs=-1,
    )


def train_ranker_ensemble(cfg: PipelineSettings, train_df: pd.DataFrame, features: Sequence[str]):
    tr = canonical_action_sort(train_df)
    groups = tr.groupby('qid', sort=False).size().tolist()
    expected_actions = 10 if list(features) == PAIR_FEATURES else 15
    assert all(x == expected_actions for x in groups)
    models = []
    for seed in cfg.model_seeds:
        model = make_ranker(cfg, seed)
        model.fit(tr[list(features)], tr.target.values, group=groups, verbose=False)
        models.append(model)
    return models


class LegalQAEngine:
    """Hybrid retrieval, hierarchical candidates and CrossEncoder reranking."""

    def __init__(
        self,
        cfg: PipelineSettings,
        cache_dir: Path,
        doc_id_to_passage: Mapping[str, str],
        doc_id_to_name: Mapping[str, str],
        doc_id_to_link: Mapping[str, str],
        stopwords: set,
        device: str = 'cuda',
        ce_batch_size: Optional[int] = None,
        load_models: bool = True,
    ):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.doc_id_to_passage = dict(doc_id_to_passage)
        self.doc_id_to_name = dict(doc_id_to_name)
        self.doc_id_to_link = dict(doc_id_to_link)
        self.stopwords = stopwords
        self.device = device
        self.ce_batch_size = int(ce_batch_size or cfg.ce_batch_size)
        self.query_cache = {}

        self.bm25_path = self.cache_dir / cfg.bm25_cache_name
        self.dense_path = self.cache_dir / cfg.dense_parent_cache_name
        self.child_path = self._resolve_child_cache()
        self._load_base_caches()

        self.bi_encoder = None
        self.cross_encoder = None
        if load_models:
            self.load_models()

    def _resolve_child_cache(self) -> Path:
        for name in self.cfg.child_cache_names:
            p = self.cache_dir / name
            if p.exists():
                return p
        return self.cache_dir / self.cfg.child_cache_names[-1]

    def _load_base_caches(self) -> None:
        for p in (self.bm25_path, self.dense_path, self.child_path):
            if not p.is_file() or p.stat().st_size == 0:
                raise FileNotFoundError(f'Missing base cache: {p}')

        print(f'[cache] loading BM25: {self.bm25_path}', flush=True)
        with self.bm25_path.open('rb') as f:
            self.bm25_doc_ids, self.bm25_index = pickle.load(f)
        print(f'[cache] BM25 loaded: {len(self.bm25_doc_ids)} docs', flush=True)

        print(f'[cache] loading dense parent: {self.dense_path}', flush=True)
        with self.dense_path.open('rb') as f:
            dp = pickle.load(f)
        self.chunk_doc_ids_all = np.asarray(dp['chunk_doc_ids_all'])
        if self.chunk_doc_ids_all.dtype.kind not in ('U', 'S'):
            self.chunk_doc_ids_all = self.chunk_doc_ids_all.astype(str)
        self.chunk_texts_all = dp['chunk_texts_all']
        self.chunk_vecs_all = np.ascontiguousarray(dp['chunk_vecs_all'], dtype=np.float32)
        del dp
        print(
            f'[cache] dense parent loaded: {len(self.chunk_texts_all)} chunks, '
            f'vectors={self.chunk_vecs_all.shape}', flush=True,
        )

        print(f'[cache] loading child: {self.child_path}', flush=True)
        with self.child_path.open('rb') as f:
            cp = pickle.load(f)
        self.child_texts = cp['child_texts']
        self.child_vecs = np.ascontiguousarray(cp['child_vecs'], dtype=np.float32)
        self.child_meta = cp['child_meta']
        del cp
        print(
            f'[cache] child loaded: {len(self.child_texts)} chunks, '
            f'vectors={self.child_vecs.shape}', flush=True,
        )

        if len(self.bm25_doc_ids) != self.cfg.expected_docs:
            raise RuntimeError(f'BM25 doc count {len(self.bm25_doc_ids)} != {self.cfg.expected_docs}')
        if len(self.chunk_texts_all) != self.cfg.expected_parent_chunks:
            raise RuntimeError(f'Parent chunk count {len(self.chunk_texts_all)} != {self.cfg.expected_parent_chunks}')
        if len(self.chunk_doc_ids_all) != len(self.chunk_texts_all) or len(self.chunk_vecs_all) != len(self.chunk_texts_all):
            raise RuntimeError('Parent cache arrays have inconsistent lengths.')
        if len(self.child_texts) != self.cfg.expected_child_chunks:
            raise RuntimeError(f'Child chunk count {len(self.child_texts)} != {self.cfg.expected_child_chunks}')
        if len(self.child_vecs) != len(self.child_texts) or len(self.child_meta) != len(self.child_texts):
            raise RuntimeError('Child cache arrays have inconsistent lengths.')
        if self.chunk_vecs_all.ndim != 2 or self.child_vecs.ndim != 2:
            raise RuntimeError('Dense vectors must be 2-D matrices.')
        if self.chunk_vecs_all.shape[1] != self.child_vecs.shape[1]:
            raise RuntimeError(
                f'Parent/child embedding dimensions differ: '
                f'{self.chunk_vecs_all.shape[1]} vs {self.child_vecs.shape[1]}'
            )
        print('[cache] checking finite embedding values', flush=True)
        if not np.isfinite(self.chunk_vecs_all).all() or not np.isfinite(self.child_vecs).all():
            raise RuntimeError('Dense cache contains non-finite values.')

        corpus_ids = set(map(str, self.doc_id_to_passage))
        bm25_ids = set(map(str, self.bm25_doc_ids))
        if bm25_ids != corpus_ids:
            raise RuntimeError(
                f'BM25/corpus doc-id mismatch: bm25_only={len(bm25_ids-corpus_ids)}, '
                f'corpus_only={len(corpus_ids-bm25_ids)}'
            )

        self.doc_to_parent = defaultdict(list)
        for i, did in enumerate(self.chunk_doc_ids_all):
            self.doc_to_parent[str(did)].append(i)
        parent_ids = set(self.doc_to_parent)
        if len(parent_ids) != self.cfg.expected_parent_docs:
            raise RuntimeError(
                f'Parent doc count {len(parent_ids)} != {self.cfg.expected_parent_docs}'
            )
        extra_parent_ids = parent_ids - corpus_ids
        missing_parent_ids = corpus_ids - parent_ids
        expected_missing = self.cfg.expected_docs - self.cfg.expected_parent_docs
        if extra_parent_ids or len(missing_parent_ids) != expected_missing:
            raise RuntimeError(
                'Parent dense cache/corpus mismatch: '
                f'extra={len(extra_parent_ids)}, missing={len(missing_parent_ids)}, '
                f'expected_missing={expected_missing}'
            )
        print(
            f'[cache] dense parent covers {len(parent_ids)} docs; '
            f'{len(missing_parent_ids)} corpus docs produced no retained parent chunk',
            flush=True,
        )

        self.doc_to_child = defaultdict(list)
        for i, meta in enumerate(self.child_meta):
            self.doc_to_child[str(meta['doc_id'])].append(i)
        if not set(self.doc_to_child) <= corpus_ids:
            raise RuntimeError('Child cache contains doc ids not present in corpus.')
        if len(self.doc_to_child) != self.cfg.expected_child_docs:
            raise RuntimeError(f'Child doc count {len(self.doc_to_child)} != {self.cfg.expected_child_docs}')

        # Vectorized document reduction is mathematically equivalent to the original
        # first-seen dict loop and preserves the same stable tie order.
        self.dense_doc_ids = list(self.doc_to_parent)
        codes = {did: i for i, did in enumerate(self.dense_doc_ids)}
        self.dense_doc_codes = np.asarray([codes[str(d)] for d in self.chunk_doc_ids_all], dtype=np.int32)
        print(
            f'[cache] validation PASS: docs={len(self.dense_doc_ids)}, '
            f'parent={len(self.chunk_texts_all)}, child={len(self.child_texts)}',
            flush=True,
        )

    def load_models(self) -> None:
        if self.bi_encoder is not None and self.cross_encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        from .reranker import TransformerReranker
        print(f'[model] loading bi-encoder {self.cfg.bi_encoder_model}', flush=True)
        self.bi_encoder = SentenceTransformer(
            self.cfg.bi_encoder_model, device=self.device, trust_remote_code=True,
        )
        self.bi_encoder.max_seq_length = self.cfg.bi_encoder_max_seq_length
        print(f'[model] loading reranker {self.cfg.reranker_model}', flush=True)
        self.cross_encoder = TransformerReranker(
            self.cfg.reranker_model,
            device=self.device,
            max_length=self.cfg.reranker_max_length,
        )
        print('[model] bi-encoder and reranker ready', flush=True)

    def get_query_vec(self, query: str) -> np.ndarray:
        if query in self.query_cache:
            return self.query_cache[query]
        if self.bi_encoder is None:
            self.load_models()
        vec = self.bi_encoder.encode(
            [query], show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        if len(self.query_cache) >= 2000:
            self.query_cache.clear()
        self.query_cache[query] = vec
        return vec

    def bm25_retrieve(self, query: str, top_k: Optional[int] = None):
        top_k = int(top_k or self.cfg.bm25_top_k)
        tokens = vi_tokenize_clean(query, self.stopwords)
        scores = self.bm25_index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        raw = [(str(self.bm25_doc_ids[i]), float(scores[i])) for i in top_idx]
        q_numbers = set(extract_numbers(query))
        if not q_numbers:
            return sorted(raw, key=lambda x: x[1], reverse=True)
        boosted = []
        for did, score in raw:
            matches = q_numbers & set(extract_numbers(self.doc_id_to_passage.get(did, '')))
            if matches:
                score *= 1 + 0.4 * len(matches)
            boosted.append((did, score))
        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def dense_retrieve_full_corpus(self, query: str, top_k: Optional[int] = None):
        top_k = int(top_k or self.cfg.dense_doc_top_k)
        qv = self.get_query_vec(query)[0]
        sims = self.chunk_vecs_all @ qv
        best = np.full(len(self.dense_doc_ids), -np.inf, dtype=sims.dtype)
        np.maximum.at(best, self.dense_doc_codes, sims)
        order = np.argsort(-best, kind='stable')[:top_k]
        return [(self.dense_doc_ids[i], float(best[i])) for i in order]

    def fuse_from_raw(self, bm25, dense, top_n: Optional[int] = None):
        top_n = int(top_n or self.cfg.fusion_top_k)
        fused = weighted_score_fusion([bm25, dense], [0.5, 1.0])
        return [did for did, _ in fused[:top_n]]

    def top_docs(self, question: str) -> List[str]:
        bm = self.bm25_retrieve(question, self.cfg.bm25_top_k)
        de = self.dense_retrieve_full_corpus(question, self.cfg.dense_doc_top_k)
        return self.fuse_from_raw(bm, de, self.cfg.fusion_top_k)

    def get_candidate_pool(self, question: str, top_docs: Sequence[str]):
        qv = self.get_query_vec(question)[0]
        pool = []
        for doc_id in top_docs:
            parent_positions = self.doc_to_parent.get(str(doc_id), [])
            if parent_positions:
                idxs = np.asarray(parent_positions, dtype=np.int64)
                sims = self.chunk_vecs_all[idxs] @ qv
                for gi, score in zip(idxs, sims):
                    pool.append((str(doc_id), self.chunk_texts_all[gi], float(score), 'parent', None))
            child_positions = self.doc_to_child.get(str(doc_id), [])
            if child_positions:
                idxs = np.asarray(child_positions, dtype=np.int64)
                sims = self.child_vecs[idxs] @ qv
                for gi, score in zip(idxs, sims):
                    meta = self.child_meta[gi]
                    pool.append((
                        str(doc_id), self.child_texts[gi], float(score), 'child',
                        (str(doc_id), meta.get('dieu_num')),
                    ))
        return pool

    def get_audit_candidate_pool(self, question: str, top_docs: Sequence[str]):
        """Return `(doc_id, kind, global_idx, text, embedding_score)` tuples."""
        qv = self.get_query_vec(question)[0]
        pool = []
        for doc_id in top_docs:
            pp = self.doc_to_parent.get(str(doc_id), [])
            if pp:
                idxs = np.asarray(pp, dtype=np.int64)
                sims = self.chunk_vecs_all[idxs] @ qv
                for gi, score in zip(idxs, sims):
                    pool.append((str(doc_id), 'parent', int(gi), self.chunk_texts_all[gi], float(score)))
            cc = self.doc_to_child.get(str(doc_id), [])
            if cc:
                idxs = np.asarray(cc, dtype=np.int64)
                sims = self.child_vecs[idxs] @ qv
                for gi, score in zip(idxs, sims):
                    pool.append((str(doc_id), 'child', int(gi), self.child_texts[gi], float(score)))
        return sorted(pool, key=lambda x: -x[4])

    def ce_score_candidates(self, question: str, candidates: Sequence[tuple]):
        if not candidates:
            return []
        if self.cross_encoder is None:
            self.load_models()
        pairs = []
        for _doc_id, text, _score, _kind, _dieu_key in candidates:
            _, body = split_chunk_prefix_and_body(text)
            pairs.append((question, body or text))
        scores = self.cross_encoder.predict(
            pairs, show_progress_bar=False, batch_size=self.ce_batch_size,
        )
        return [
            (doc_id, text, score, float(ce), kind, dieu_key)
            for (doc_id, text, score, kind, dieu_key), ce in zip(candidates, scores)
        ]

    def score_audit_candidates(self, question: str, candidates: Sequence[tuple]):
        if not candidates:
            return []
        if self.cross_encoder is None:
            self.load_models()
        pairs = [(question, c[3]) for c in candidates]
        scores = np.asarray(self.cross_encoder.predict(
            pairs, show_progress_bar=False, batch_size=self.ce_batch_size,
        ), dtype=float).reshape(-1)
        return [
            (c[0], c[1], c[2], c[3], c[4], float(score))
            for c, score in zip(candidates, scores)
        ]

    def retrieve_top5(self, question: str):
        top_docs_full = self.top_docs(question)
        top_docs = top_docs_full[:self.cfg.final_top_docs]
        if len(top_docs) != self.cfg.final_top_docs:
            raise RuntimeError(f'Expected top3 docs, got {len(top_docs)}')
        pool = self.get_candidate_pool(question, top_docs)
        if not pool:
            raise RuntimeError('Retrieval produced an empty candidate pool')
        candidates = sorted(pool, key=lambda x: -x[2])[:self.cfg.pre_ce_top_k]
        if len(candidates) < self.cfg.ce_top_k:
            raise RuntimeError(
                f'Expected at least {self.cfg.ce_top_k} embedding candidates, '
                f'got {len(candidates)}'
            )
        reranked = sorted(self.ce_score_candidates(question, candidates), key=lambda x: -x[3])
        top5 = reranked[:self.cfg.ce_top_k]
        if len(top5) != self.cfg.ce_top_k:
            raise RuntimeError(f'Expected CE top5, got {len(top5)}')
        return top5, {
            'top_docs': [str(x) for x in top_docs],
            'pre_ce_count': len(candidates),
            'ce_top5_count': len(top5),
        }

    def candidate_dicts(self, top5: Sequence[tuple]) -> List[dict]:
        result = []
        for rank, c in enumerate(top5, start=1):
            doc_id, text, emb, ce, kind, _dieu_key = c
            result.append({
                'doc_id': str(doc_id), 'text': text,
                'emb': float(emb), 'ce': float(ce),
                'kind': str(kind), 'ce_rank': rank,
            })
        return result

    def build_answer(self, question: str, kept: Sequence[tuple]) -> str:
        return build_answer_from_candidates(
            question, kept,
            self.doc_id_to_passage,
            self.doc_id_to_name,
            self.doc_id_to_link,
            self.cfg.max_chars,
        )

    def build_pair_rows(self, qid: str, question: str, gold: str, top5: Sequence[tuple]) -> List[dict]:
        cand_dicts = self.candidate_dicts(top5)
        rows = []
        for i, j in itertools.combinations(range(5), 2):
            pred = self.build_answer(question, [top5[i], top5[j]])
            rows.append({
                'qid': str(qid), 'pair_i': i + 1, 'pair_j': j + 1,
                'target': meteor_local(pred, gold),
                **pair_features(question, cand_dicts[i], cand_dicts[j]),
            })
        assert len(rows) == 10
        return rows

    def build_singleton_targets(self, question: str, gold: str, cand_dicts: Sequence[Mapping]) -> List[dict]:
        targets = []
        for rank, c in enumerate(cand_dicts, start=1):
            pred = self.build_answer(question, [candidate_tuple_from_dict(c)])
            targets.append({'rank': rank, 'target': meteor_local(pred, gold), 'answer_len': len(pred)})
        return targets

    def predict(self, question: str, selector, qid: str, return_debug: bool = False):
        top5, ret_dbg = self.retrieve_top5(question)
        cand_dicts = self.candidate_dicts(top5)
        action_df = build_action_frame(str(qid), question, cand_dicts)
        selection = selector.select_action(action_df)
        size, i, j = selection.action_size, selection.action_i, selection.action_j
        if size == 1:
            kept = [top5[i - 1]]
        elif size == 2:
            kept = [top5[i - 1], top5[j - 1]]
        else:
            raise RuntimeError(f'Unexpected action_size={size}')
        raw_answer = self.build_answer(question, kept)
        answer = postprocess_best_06252(raw_answer, question=question, qid=str(qid))
        if not return_debug:
            return answer
        return answer, {
            **ret_dbg,
            'retarget_action': selection.retarget_action,
            'baseline_meta_action': selection.baseline_meta_action,
            'pairwise_action': selection.pairwise_action,
            'final_action': selection.final_action,
            'direct_prob': float(selection.direct_prob),
            'used_pairwise_gate': bool(selection.used_pairwise_gate),
            'allowed_action_count': int(selection.allowed_action_count),
            'selected_action_size': size,
            'selected_i': i,
            'selected_j': j,
        }
