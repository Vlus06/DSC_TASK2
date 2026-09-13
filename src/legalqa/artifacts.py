"""Build, validate and reuse training artifacts for the LegalQA pipeline."""
from __future__ import annotations

import gc
import itertools
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .engine import (
    ACTION_FEATURES,
    PAIR_FEATURES,
    LegalQAEngine,
    candidate_tuple_from_dict,
    pair_features,
    singleton_action_features,
)
from .settings import PipelineSettings, resolve_cache_file

TOL = 1e-12


def atomic_pickle_dump(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_json(path: Path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def load_corpus_metadata(task2_dir: Path):
    task2_dir = Path(task2_dir)
    context_dir = task2_dir / 'selected-contexts'
    if not context_dir.is_dir():
        # Some archives place the real directory one level deeper.
        candidates = [p for p in task2_dir.rglob('selected-contexts') if p.is_dir()]
        if len(candidates) == 1:
            context_dir = candidates[0]
        else:
            raise FileNotFoundError(f'Không tìm thấy selected-contexts dưới {task2_dir}')

    passages: Dict[str, str] = {}
    names: Dict[str, str] = {}
    links: Dict[str, str] = {}
    for fp in sorted(context_dir.rglob('*.json')):
        with fp.open(encoding='utf-8') as f:
            doc = json.load(f)
        did = str(doc.get('id', fp.stem.replace('context_', '')))
        passages[did] = doc.get('passage', '') or ''
        names[did] = doc.get('name', '') or ''
        links[did] = doc.get('link', '') or ''
    return passages, names, links


def load_task2(task2_dir: Path):
    task2_dir = Path(task2_dir)
    train_path = task2_dir / 'train.json'
    public_path = task2_dir / 'public-official.json'
    if not (train_path.is_file() and public_path.is_file()):
        candidates = []
        for tp in task2_dir.rglob('train.json'):
            d = tp.parent
            if (d / 'public-official.json').is_file() and (d / 'selected-contexts').is_dir():
                candidates.append(d)
        if len(candidates) == 1:
            task2_dir = candidates[0]
            train_path = task2_dir / 'train.json'
            public_path = task2_dir / 'public-official.json'
        else:
            raise FileNotFoundError(f'Không xác định được thư mục TASK2 hợp lệ từ {task2_dir}')

    train = load_json(train_path)
    public = load_json(public_path)
    packed_path = resolve_cache_file(task2_dir.parent / 'cache', 'corpus_metadata.pkl')
    if packed_path.is_file() and packed_path.stat().st_size > 0:
        print(f'[corpus] loading packed corpus: {packed_path}', flush=True)
        with packed_path.open('rb') as f:
            packed = pickle.load(f)
        required = {'passages', 'names', 'links'}
        if not isinstance(packed, dict) or not required <= set(packed):
            raise RuntimeError(f'Unexpected packed corpus schema: {packed_path}')
        passages = {str(k): v for k, v in packed['passages'].items()}
        names = {str(k): v for k, v in packed['names'].items()}
        links = {str(k): v for k, v in packed['links'].items()}
        if not (len(passages) == len(names) == len(links) == 8532):
            raise RuntimeError(
                f'Packed corpus size mismatch: passages={len(passages)}, '
                f'names={len(names)}, links={len(links)}'
            )
        print(f'[corpus] packed corpus loaded: {len(passages)} docs', flush=True)
    else:
        print('[corpus] packed corpus missing; reading individual JSON files', flush=True)
        passages, names, links = load_corpus_metadata(task2_dir)
    return task2_dir, train, public, passages, names, links


def split_training_data(train: Mapping, seed: int = 42):
    qids = list(train.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    qids = list(map(str, qids))
    if len(qids) != 7000:
        raise RuntimeError(f'Expected 7000 train qids, got {len(qids)}')
    base1000 = qids[:1000]
    holdout1000 = qids[1000:2000]
    training5000 = qids[2000:7000]
    return base1000, holdout1000, training5000


def _normalize_train_keys(train: Mapping) -> dict:
    return {str(k): v for k, v in train.items()}


def build_or_load_top_documents(
    cfg: PipelineSettings,
    engine: LegalQAEngine,
    train: Mapping,
    base1000: Sequence[str],
    cache_dir: Path,
    checkpoint_every: int = 25,
):
    """Build or load top-document candidates for the first 1,000 qids."""
    train = _normalize_train_keys(train)
    path = resolve_cache_file(cache_dir, cfg.top_documents_cache_name)
    rows = {}
    provenance = 'rebuilt'
    if path.exists():
        with path.open('rb') as f:
            raw_obj = pickle.load(f)
        rows = {str(k): v for k, v in raw_obj.items()}
        if not set(rows) <= set(base1000):
            raise RuntimeError(f'{path.name} chứa qid ngoài base1000 seed=42.')
        for qid, value in rows.items():
            if value is None or len(value) < 3 or len(value[2]) < cfg.final_top_docs:
                raise RuntimeError(f'{path.name} malformed at qid={qid}')
        if len(rows) == len(base1000):
            if list(rows.keys()) != list(base1000):
                raise RuntimeError(f'{path.name} đủ 1000 qid nhưng thứ tự split không khớp.')
            return rows, 'existing'
        provenance = 'resumed'
        print(f'Loaded partial {path.name}: {len(rows)}/1000 qids', flush=True)

    todo = [qid for qid in base1000 if qid not in rows]
    started = time.time()
    for i, qid in enumerate(todo, start=1):
        item = train[qid]
        q = item['question']
        gold = item['answer']
        docs = engine.top_docs(q)
        if len(docs) < cfg.final_top_docs:
            raise RuntimeError(f'Historical qid={qid}: expected >=3 docs, got {len(docs)}')
        rows[qid] = (q, gold, docs)
        if i % checkpoint_every == 0 or i == len(todo):
            # Keep deterministic split order when writing checkpoints.
            ordered = {k: rows[k] for k in base1000 if k in rows}
            atomic_pickle_dump(ordered, path)
            rows = ordered
            elapsed = time.time() - started
            print(f'[top documents] {i}/{len(todo)} new | total={len(rows)}/1000 elapsed={elapsed/60:.1f}m', flush=True)
    if list(rows.keys()) != list(base1000):
        raise RuntimeError('Top-document cache build incomplete')
    return rows, provenance if not todo else ('rebuilt' if provenance == 'rebuilt' else 'resumed')


def build_or_load_candidate_audit(
    cfg: PipelineSettings,
    engine: LegalQAEngine,
    train: Mapping,
    base1000: Sequence[str],
    cache_dir: Path,
    top_documents: Mapping,
    checkpoint_every: int = 25,
):
    """Build or load CrossEncoder scores for the first 1,000 qids."""
    train = _normalize_train_keys(train)
    path = resolve_cache_file(cache_dir, cfg.candidate_audit_cache_name)
    audit = {}
    provenance = 'rebuilt'
    if path.exists():
        with path.open('rb') as f:
            raw_obj = pickle.load(f)
        audit = {str(k): v for k, v in raw_obj.items()}
        if not set(audit) <= set(base1000):
            raise RuntimeError(f'{path.name} chứa qid ngoài base1000.')
        for qid, raw in audit.items():
            if raw is None or len(raw) < 10:
                raise RuntimeError(f'{path.name} thiếu top10 tại qid={qid}')
        if len(audit) == len(base1000):
            return audit, 'existing'
        provenance = 'resumed'
        print(f'Loaded partial {path.name}: {len(audit)}/1000 qids', flush=True)

    todo = [qid for qid in base1000 if qid not in audit]
    started = time.time()
    for i, qid in enumerate(todo, start=1):
        q, _gold, docs = top_documents[qid]
        pool = engine.get_audit_candidate_pool(q, list(docs)[:cfg.final_top_docs])[:50]
        if len(pool) < 10:
            raise RuntimeError(f'Candidate audit qid={qid}: expected >=10 candidates, got {len(pool)}')
        audit[qid] = engine.score_audit_candidates(q, pool)
        if i % checkpoint_every == 0 or i == len(todo):
            ordered = {k: audit[k] for k in base1000 if k in audit}
            atomic_pickle_dump(ordered, path)
            audit = ordered
            elapsed = time.time() - started
            print(f'[candidate audit] {i}/{len(todo)} new | total={len(audit)}/1000 elapsed={elapsed/60:.1f}m', flush=True)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    if set(audit) != set(base1000):
        raise RuntimeError('Candidate audit cache build incomplete')
    return audit, provenance if not todo else ('rebuilt' if provenance == 'rebuilt' else 'resumed')


def _pair_cache_path(cfg: PipelineSettings, cache_dir: Path) -> Path:
    return resolve_cache_file(cache_dir, cfg.pair_features_cache_name)


def build_or_load_pair_features(
    cfg: PipelineSettings,
    engine: LegalQAEngine,
    train: Mapping,
    base1000: Sequence[str],
    holdout1000: Sequence[str],
    training5000: Sequence[str],
    cache_dir: Path,
    checkpoint_every: int = 25,
):
    """Build or load pairwise feature rows for the remaining 6,000 questions."""
    train = _normalize_train_keys(train)
    remaining_qids = list(holdout1000) + list(training5000)
    path = _pair_cache_path(cfg, cache_dir)
    rows_by_qid = {}
    provenance = 'rebuilt'

    if path.exists():
        with path.open('rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict) or 'rows_by_qid' not in obj:
            raise RuntimeError(f'{path.name}: invalid pair-feature cache')
        cached_base = obj.get('base1000', obj.get('historical_dev1000', base1000))
        cached_holdout = obj.get('holdout1000', obj.get('fresh_holdout1000', holdout1000))
        cached_training = obj.get('training5000', obj.get('scale_train5000', training5000))
        if list(map(str, cached_base)) != list(base1000):
            raise RuntimeError(f'{path.name}: base split mismatch')
        if list(map(str, cached_holdout)) != list(holdout1000):
            raise RuntimeError(f'{path.name}: holdout split mismatch')
        if list(map(str, cached_training)) != list(training5000):
            raise RuntimeError(f'{path.name}: training split mismatch')
        rows_by_qid = {str(k): v for k, v in obj.get('rows_by_qid', {}).items()}
        provenance = 'existing' if len(rows_by_qid) == 6000 else 'resumed'
        print(f'Loaded {path.name}: {len(rows_by_qid)}/6000 qids', flush=True)

    done = set(rows_by_qid)
    if not done <= set(remaining_qids):
        raise RuntimeError(f'{path.name}: contains unexpected qids')
    for qid, rows in rows_by_qid.items():
        if len(rows) != 10:
            raise RuntimeError(f'{path.name}: qid={qid} has {len(rows)} rows, expected 10')

    def save():
        atomic_pickle_dump({
            'version': cfg.pair_features_cache_version,
            'base1000': list(base1000),
            'holdout1000': list(holdout1000),
            'training5000': list(training5000),
            'rows_by_qid': rows_by_qid,
        }, path)

    todo = [qid for qid in remaining_qids if qid not in done]
    started = time.time()
    for i, qid in enumerate(todo, start=1):
        item = train[qid]
        try:
            top5, _dbg = engine.retrieve_top5(item['question'])
            rows_by_qid[qid] = engine.build_pair_rows(
                qid, item['question'], item['answer'], top5,
            )
        except Exception:
            save()
            print(f'FAILED qid={qid}; progress saved to {path}', flush=True)
            raise
        if i % checkpoint_every == 0 or i == len(todo):
            save()
            elapsed = time.time() - started
            rate = i / max(elapsed, 1e-9)
            eta = (len(todo) - i) / max(rate, 1e-9)
            print(
                f'[pair features] {i}/{len(todo)} new | total={len(rows_by_qid)}/6000 '
                f'elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m', flush=True,
            )
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if set(rows_by_qid) != set(remaining_qids):
        raise RuntimeError('Pair-feature cache incomplete after build')
    save()
    return {
        'version': cfg.pair_features_cache_version,
        'base1000': list(base1000),
        'holdout1000': list(holdout1000),
        'training5000': list(training5000),
        'rows_by_qid': rows_by_qid,
    }, provenance if not todo else ('rebuilt' if provenance == 'rebuilt' else 'resumed')


def sorted_pair_rows(pair_cache: Mapping, qid: str) -> list:
    rows = list(pair_cache['rows_by_qid'][str(qid)])
    rows.sort(key=lambda r: (int(r['pair_i']), int(r['pair_j'])))
    expected = list(itertools.combinations(range(1, 6), 2))
    ids = [(int(r['pair_i']), int(r['pair_j'])) for r in rows]
    if ids != expected:
        raise RuntimeError(f'Pair-feature order/schema mismatch qid={qid}: {ids}')
    return rows


def reconstruct_candidate_scores(pair_cache: Mapping, qid: str) -> dict:
    rows = sorted_pair_rows(pair_cache, qid)
    slots = {r: {'ce': [], 'emb': [], 'is_parent': [], 'is_child': []} for r in range(1, 6)}
    for row in rows:
        i, j = int(row['pair_i']), int(row['pair_j'])
        slots[i]['ce'].append(float(row['c1_ce']))
        slots[i]['emb'].append(float(row['c1_emb']))
        slots[i]['is_parent'].append(int(row['c1_is_parent']))
        slots[i]['is_child'].append(int(row['c1_is_child']))
        slots[j]['ce'].append(float(row['c2_ce']))
        slots[j]['emb'].append(float(row['c2_emb']))
        slots[j]['is_parent'].append(int(row['c2_is_parent']))
        slots[j]['is_child'].append(int(row['c2_is_child']))
    result = {}
    for rank, data in slots.items():
        ce0, emb0 = data['ce'][0], data['emb'][0]
        p0, c0 = data['is_parent'][0], data['is_child'][0]
        if not all(abs(x - ce0) <= TOL for x in data['ce']):
            raise RuntimeError(f'cached CE inconsistent qid={qid} rank={rank}')
        if not all(abs(x - emb0) <= TOL for x in data['emb']):
            raise RuntimeError(f'cached EMB inconsistent qid={qid} rank={rank}')
        if not all(x == p0 for x in data['is_parent']) or not all(x == c0 for x in data['is_child']):
            raise RuntimeError(f'cached kind inconsistent qid={qid} rank={rank}')
        result[rank] = {'ce': ce0, 'emb': emb0, 'is_parent': p0, 'is_child': c0}
    return result


def verify_pair_feature_identity(pair_cache: Mapping, qid: str, question: str, candidate_dicts: Sequence[Mapping]):
    rows = sorted_pair_rows(pair_cache, qid)
    max_abs_diff = 0.0
    for cached, (i0, j0) in zip(rows, itertools.combinations(range(5), 2)):
        generated = pair_features(question, candidate_dicts[i0], candidate_dicts[j0])
        for feature in PAIR_FEATURES:
            a = float(generated[feature])
            b = float(cached[feature])
            diff = abs(a - b)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > TOL:
                raise RuntimeError(
                    'Pair-feature candidate identity mismatch: '
                    f'qid={qid} pair=({i0+1},{j0+1}) feature={feature} '
                    f'generated={a} cached={b} diff={diff}'
                )
    return max_abs_diff


def merge_current_and_cached_candidates(pair_cache: Mapping, qid: str, top5: Sequence[tuple]):
    cached = reconstruct_candidate_scores(pair_cache, qid)
    out = []
    score_diff = {'ce': 0.0, 'emb': 0.0}
    for rank, c in enumerate(top5, start=1):
        doc_id, text, current_emb, current_ce, kind, _dieu_key = c
        s = cached[rank]
        kind = str(kind)
        if int(kind == 'parent') != s['is_parent'] or int(kind == 'child') != s['is_child']:
            raise RuntimeError(f'candidate kind mismatch qid={qid} rank={rank}')
        score_diff['ce'] = max(score_diff['ce'], abs(float(current_ce) - float(s['ce'])))
        score_diff['emb'] = max(score_diff['emb'], abs(float(current_emb) - float(s['emb'])))
        out.append({
            'doc_id': str(doc_id), 'text': text,
            # Keep cached scores so all rows use the same feature values.
            'emb': float(s['emb']), 'ce': float(s['ce']),
            'kind': kind, 'ce_rank': rank,
        })
    return out, score_diff


def build_or_load_singleton_features(
    cfg: PipelineSettings,
    engine: LegalQAEngine,
    train: Mapping,
    holdout1000: Sequence[str],
    training5000: Sequence[str],
    cache_dir: Path,
    pair_cache: Mapping,
    checkpoint_every: int = 25,
):
    train = _normalize_train_keys(train)
    remaining_qids = list(holdout1000) + list(training5000)
    path = resolve_cache_file(cache_dir, cfg.singleton_features_cache_name)
    items = {}
    provenance = 'rebuilt'

    if path.exists():
        with path.open('rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict) or 'items_by_qid' not in obj:
            raise RuntimeError(f'{path.name}: invalid singleton-feature cache')
        cached_holdout = obj.get('holdout1000', obj.get('fresh1000', []))
        cached_training = obj.get('training5000', obj.get('scale5000', []))
        if list(map(str, cached_holdout)) != list(holdout1000):
            raise RuntimeError(f'{path.name}: holdout split mismatch')
        if list(map(str, cached_training)) != list(training5000):
            raise RuntimeError(f'{path.name}: training split mismatch')
        items = {str(k): v for k, v in obj.get('items_by_qid', {}).items()}
        provenance = 'existing' if len(items) == 6000 else 'resumed'
        print(f'Loaded {path.name}: {len(items)}/6000 qids', flush=True)

    def save():
        atomic_pickle_dump({
            'version': cfg.singleton_features_cache_version,
            'source_pair_features_version': pair_cache.get('version'),
            'holdout1000': list(holdout1000),
            'training5000': list(training5000),
            'items_by_qid': items,
        }, path)

    remaining_set = set(remaining_qids)
    if not set(items) <= remaining_set:
        raise RuntimeError(f'{path.name}: contains unexpected qids')

    # Verify reused items cheaply before trusting them.
    for qid, item in items.items():
        candidates = item['candidates']
        if len(candidates) != 5 or len(item['singleton_targets']) != 5:
            raise RuntimeError(f'{path.name}: malformed qid={qid}')
        cand_dicts = [{
            'doc_id': str(c['doc_id']), 'text': c['text'],
            'emb': float(c['emb']), 'ce': float(c['ce']),
            'kind': str(c['kind']), 'ce_rank': int(c['rank']),
        } for c in candidates]
        verify_pair_feature_identity(pair_cache, qid, item['question'], cand_dicts)

    todo = [qid for qid in remaining_qids if qid not in items]
    started = time.time()
    max_ce_diff = 0.0
    max_emb_diff = 0.0
    for i, qid in enumerate(todo, start=1):
        q = train[qid]['question']
        gold = train[qid]['answer']
        try:
            top5, ret_dbg = engine.retrieve_top5(q)
            cand_dicts, score_diff = merge_current_and_cached_candidates(pair_cache, qid, top5)
            verify_pair_feature_identity(pair_cache, qid, q, cand_dicts)
            max_ce_diff = max(max_ce_diff, score_diff['ce'])
            max_emb_diff = max(max_emb_diff, score_diff['emb'])
            singleton_targets = engine.build_singleton_targets(q, gold, cand_dicts)
            items[qid] = {
                'question': q,
                'candidates': [{
                    'rank': int(c['ce_rank']), 'doc_id': str(c['doc_id']),
                    'kind': str(c['kind']), 'text': c['text'],
                    'emb': float(c['emb']), 'ce': float(c['ce']),
                } for c in cand_dicts],
                'singleton_targets': singleton_targets,
                'retrieval_debug': ret_dbg,
                'current_vs_cached_max_ce_diff': float(score_diff['ce']),
                'current_vs_cached_max_emb_diff': float(score_diff['emb']),
            }
        except Exception:
            save()
            print(f'FAILED qid={qid}; progress saved to {path}', flush=True)
            raise

        if i % checkpoint_every == 0 or i == len(todo):
            save()
            elapsed = time.time() - started
            rate = i / max(elapsed, 1e-9)
            eta = (len(todo) - i) / max(rate, 1e-9)
            print(
                f'[singleton features] {i}/{len(todo)} new | total={len(items)}/6000 '
                f'elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m', flush=True,
            )
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if set(items) != remaining_set:
        raise RuntimeError('Singleton-feature cache incomplete after build')
    save()
    return {
        'version': cfg.singleton_features_cache_version,
        'source_pair_features_version': pair_cache.get('version'),
        'holdout1000': list(holdout1000),
        'training5000': list(training5000),
        'items_by_qid': items,
    }, provenance if not todo else ('rebuilt' if provenance == 'rebuilt' else 'resumed'), {
        'max_current_vs_cached_ce_diff': float(max_ce_diff),
        'max_current_vs_cached_emb_diff': float(max_emb_diff),
    }


def assemble_training_actions(
    cfg: PipelineSettings,
    engine: LegalQAEngine,
    train: Mapping,
    base1000: Sequence[str],
    holdout1000: Sequence[str],
    training5000: Sequence[str],
    audit: Mapping,
    pair_cache: Mapping,
    singleton_cache: Mapping,
):
    train = _normalize_train_keys(train)
    base_pair_rows = []
    base_single_rows = []
    base_items = {}

    for pos, qid in enumerate(base1000, start=1):
        raw = audit.get(qid, audit.get(int(qid) if qid.isdigit() else qid))
        if raw is None:
            raise RuntimeError(f'audit missing qid={qid}')
        ranked = [{
            'doc_id': str(x[0]), 'kind': str(x[1]), 'idx': x[2],
            'text': x[3], 'emb': float(x[4]), 'ce': float(x[5]),
        } for x in raw[:10]]
        ranked.sort(key=lambda z: -z['ce'])
        if len(ranked) < 5:
            raise RuntimeError(f'audit qid={qid} has <5 candidates')
        ranked = ranked[:5]
        for rank, c in enumerate(ranked, start=1):
            c['ce_rank'] = rank

        q = train[qid]['question']
        gold = train[qid]['answer']
        base_items[qid] = {
            'question': q,
            'candidates': [{
                'rank': int(c['ce_rank']), 'doc_id': str(c['doc_id']),
                'kind': str(c['kind']), 'text': c['text'],
                'emb': float(c['emb']), 'ce': float(c['ce']),
            } for c in ranked],
        }

        for i, j in itertools.combinations(range(5), 2):
            c1, c2 = ranked[i], ranked[j]
            pred = engine.build_answer(q, [candidate_tuple_from_dict(c1), candidate_tuple_from_dict(c2)])
            base_pair_rows.append({
                'qid': qid, 'action_type': 'pair', 'action_size': 2,
                'action_i': i + 1, 'action_j': j + 1,
                'target': compute_target_score(pred, gold),
                **pair_features(q, c1, c2),
            })

        for rank, c in enumerate(ranked, start=1):
            pred = engine.build_answer(q, [candidate_tuple_from_dict(c)])
            base_single_rows.append({
                'qid': qid, 'action_type': 'single', 'action_size': 1,
                'action_i': rank, 'action_j': 0,
                'target': compute_target_score(pred, gold),
                **singleton_action_features(q, c),
            })
        if pos % 250 == 0:
            print(f'base group assembled {pos}/1000', flush=True)

    base_pair_df = pd.DataFrame(base_pair_rows)
    base_single_df = pd.DataFrame(base_single_rows)
    if len(base_pair_df) != 10000 or len(base_single_df) != 5000:
        raise RuntimeError('Base-group assembly row-count mismatch')

    remaining_qids = list(holdout1000) + list(training5000)
    remaining_pair_rows = []
    remaining_single_rows = []
    items = {str(k): v for k, v in singleton_cache['items_by_qid'].items()}
    for pos, qid in enumerate(remaining_qids, start=1):
        for r in sorted_pair_rows(pair_cache, qid):
            row = {
                'qid': qid, 'action_type': 'pair', 'action_size': 2,
                'action_i': int(r['pair_i']), 'action_j': int(r['pair_j']),
                'target': float(r['target']),
            }
            for feature in PAIR_FEATURES:
                row[feature] = r[feature]
            remaining_pair_rows.append(row)

        item = items[qid]
        candidates = [{
            'doc_id': str(c['doc_id']), 'text': c['text'],
            'emb': float(c['emb']), 'ce': float(c['ce']),
            'kind': str(c['kind']), 'ce_rank': int(c['rank']),
        } for c in item['candidates']]
        targets = sorted(item['singleton_targets'], key=lambda x: int(x['rank']))
        if len(candidates) != 5 or [int(x['rank']) for x in targets] != [1, 2, 3, 4, 5]:
            raise RuntimeError(f'singleton cache malformed qid={qid}')
        for rank, c in enumerate(candidates, start=1):
            remaining_single_rows.append({
                'qid': qid, 'action_type': 'single', 'action_size': 1,
                'action_i': rank, 'action_j': 0,
                'target': float(targets[rank - 1]['target']),
                **singleton_action_features(item['question'], c),
            })
        if pos % 1000 == 0:
            print(f'remaining assembled {pos}/6000', flush=True)

    remaining_pair_df = pd.DataFrame(remaining_pair_rows)
    remaining_single_df = pd.DataFrame(remaining_single_rows)
    pair_rows = pd.concat([base_pair_df, remaining_pair_df], ignore_index=True)
    actions = pd.concat([
        base_pair_df, base_single_df,
        remaining_pair_df, remaining_single_df,
    ], ignore_index=True)

    if len(pair_rows) != 70000 or pair_rows.qid.nunique() != 7000:
        raise RuntimeError('Pairwise training-row assembly mismatch')
    if len(actions) != 105000 or actions.qid.nunique() != 7000:
        raise RuntimeError('Training-action assembly mismatch')
    if not (pair_rows.groupby('qid').size() == 10).all():
        raise RuntimeError('Pairwise training group-size mismatch')
    if not (actions.groupby('qid').size() == 15).all():
        raise RuntimeError('Training-action group-size mismatch')

    pair_from_action = actions[actions.action_size == 2].sort_values(
        ['qid', 'action_i', 'action_j'], kind='stable',
    ).reset_index(drop=True)
    pair_reference = pair_rows.sort_values(
        ['qid', 'action_i', 'action_j'], kind='stable',
    ).reset_index(drop=True)
    if not pair_from_action[['qid', 'action_i', 'action_j']].equals(
        pair_reference[['qid', 'action_i', 'action_j']]
    ):
        raise RuntimeError('Pair identity mismatch in assembled training actions')
    for col in ['target'] + PAIR_FEATURES:
        if not np.allclose(
            pair_from_action[col].to_numpy(dtype=np.float64),
            pair_reference[col].to_numpy(dtype=np.float64),
            rtol=0.0, atol=0.0, equal_nan=True,
        ):
            raise RuntimeError(f'pair value mismatch col={col}')

    single_df = actions[actions.action_size == 1]
    undef = [
        'c2_ce','c2_emb','c2_ce_rank','c2_len','c2_body_len',
        'c2_q_overlap','c2_q_jaccard','c2_is_parent','c2_is_child','c2_has_dieu',
        'same_doc','same_kind','same_dieu','pair_text_jaccard','len_ratio','rank_gap',
    ]
    if not (single_df[undef].isna().mean() == 1.0).all():
        raise RuntimeError('Singleton feature NaN mapping mismatch')

    return {
        'base_pair_df': base_pair_df,
        'base_single_df': base_single_df,
        'base_items_by_qid': base_items,
        'remaining_pair_df': remaining_pair_df,
        'remaining_single_df': remaining_single_df,
        'pair_rows': pair_rows,
        'actions': actions,
    }


def compute_target_score(pred: str, gold: str) -> float:
    # Local alias avoids a circular-looking dependency in the assembly code.
    from .engine import meteor_local
    return meteor_local(pred, gold)


def cache_manifest(cfg: PipelineSettings, cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    files = []
    names = [
        cfg.bm25_cache_name,
        cfg.dense_parent_cache_name,
        *cfg.child_cache_names,
        cfg.corpus_cache_name,
        cfg.top_documents_cache_name,
        cfg.candidate_audit_cache_name,
        cfg.pair_features_cache_name,
        cfg.singleton_features_cache_name,
    ]
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = cache_dir / name
        if p.exists():
            files.append({'name': name, 'size_bytes': p.stat().st_size})
    return {'cache_dir': str(cache_dir), 'files': files}
