"""Resumable training pipeline for the frozen LegalQA selector V2.1."""
from __future__ import annotations

import hashlib
import gc
import json
import os
import pickle
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .engine import (
    ACTION_FEATURES,
    build_action_frame,
    candidate_tuple_from_dict,
    canonical_action_sort,
    meteor_local,
)
from .meta_selector import add_meta_transforms, add_v21_features, fit_meta_model, fit_pairwise_model
from .model_bundle import BUNDLE_VERSION, MANIFEST_NAME, load_bundle, save_bundle
from .postprocess_v87 import postprocess_best_06252
from .selector_models import (
    MODEL_NAMES,
    RETARGET_SEEDS,
    add_level0_predictions,
    fit_heterogeneous_model,
    fit_retarget_models,
)


FINAL_TARGET_VERSION = "legalqa_final_target_best_06252_scale5000_v1"
OOF_VERSION = "legalqa_selector_v21_oof5_v1"
ACTION_KEYS = tuple(
    [f"S{i}" for i in range(1, 6)]
    + [f"P{i}_{j}" for i in range(1, 6) for j in range(i + 1, 6)]
)


def _atomic_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _signature(payload: Mapping) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _source_cache_digest(
    scale_qids: Sequence[str], pair_cache: Mapping, singleton_cache: Mapping,
) -> str:
    """Bind resumable targets/OOF to exact candidate text, scores and raw targets."""
    digest = hashlib.sha256()
    for qid in map(str, scale_qids):
        item = singleton_cache["items_by_qid"][qid]
        pairs = sorted(
            pair_cache["rows_by_qid"][qid],
            key=lambda row: (int(row["pair_i"]), int(row["pair_j"])),
        )
        source = {
            "qid": qid,
            "question": item["question"],
            "candidates": item["candidates"],
            "singleton_targets": item["singleton_targets"],
            "pairs": pairs,
        }
        digest.update(
            pickle.dumps(source, protocol=4)
        )
    return digest.hexdigest()


def training_signature(
    scale_qids: Sequence[str], pair_cache: Mapping, singleton_cache: Mapping,
) -> tuple[dict, str]:
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "final_target_version": FINAL_TARGET_VERSION,
        "oof_version": OOF_VERSION,
        "postprocess": "best_06252",
        "scale_qids": [str(qid) for qid in scale_qids],
        "action_keys": list(ACTION_KEYS),
        "action_features": list(ACTION_FEATURES),
        "pair_cache_version": pair_cache.get("version"),
        "singleton_cache_version": singleton_cache.get("version"),
        "singleton_source_pair_version": singleton_cache.get("source_pair_features_version"),
        "source_cache_sha256": _source_cache_digest(
            scale_qids, pair_cache, singleton_cache,
        ),
        "retarget_seeds": list(RETARGET_SEEDS),
        "model_names": list(MODEL_NAMES),
        "fold_rule": "position_mod_5_plus_1",
    }
    return payload, _signature(payload)


def _candidate_dicts(singleton_cache: Mapping, qid: str) -> tuple[str, list[dict]]:
    item = singleton_cache["items_by_qid"][str(qid)]
    candidates = [
        {
            "doc_id": str(candidate["doc_id"]),
            "text": candidate["text"],
            "emb": float(candidate["emb"]),
            "ce": float(candidate["ce"]),
            "kind": str(candidate["kind"]),
            "ce_rank": int(candidate["rank"]),
        }
        for candidate in item["candidates"]
    ]
    if [candidate["ce_rank"] for candidate in candidates] != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Invalid cached top-5 order for qid={qid}")
    return str(item["question"]), candidates


def _raw_target_map(pair_cache: Mapping, singleton_cache: Mapping, qid: str) -> dict[str, float]:
    item = singleton_cache["items_by_qid"][str(qid)]
    singletons = sorted(item["singleton_targets"], key=lambda row: int(row["rank"]))
    result = {f"S{int(row['rank'])}": float(row["target"]) for row in singletons}
    pairs = pair_cache["rows_by_qid"][str(qid)]
    result.update({
        f"P{int(row['pair_i'])}_{int(row['pair_j'])}": float(row["target"])
        for row in pairs
    })
    if tuple(sorted(result, key=lambda key: (key[0] == "P", key))) != tuple(
        sorted(ACTION_KEYS, key=lambda key: (key[0] == "P", key))
    ):
        raise RuntimeError(f"Historical target action schema mismatch for qid={qid}")
    return result


def _materialize_action(engine, question: str, candidates: Sequence[Mapping], row: Mapping) -> str:
    ranks = [int(row["action_i"])]
    if int(row["action_size"]) == 2:
        ranks.append(int(row["action_j"]))
    kept = [candidate_tuple_from_dict(candidates[rank - 1]) for rank in ranks]
    return engine.build_answer(question, kept)


def verify_raw_target_reproduction(
    engine,
    train: Mapping,
    scale_qids: Sequence[str],
    pair_cache: Mapping,
    singleton_cache: Mapping,
    sample_qids: int = 20,
    tolerance: float = 1e-12,
) -> dict:
    positions = np.linspace(0, len(scale_qids) - 1, min(sample_qids, len(scale_qids)), dtype=int)
    checked = 0
    maximum = 0.0
    for position in dict.fromkeys(positions.tolist()):
        qid = str(scale_qids[position])
        question, candidates = _candidate_dicts(singleton_cache, qid)
        frame = build_action_frame(qid, question, candidates)
        historical = _raw_target_map(pair_cache, singleton_cache, qid)
        gold = train[qid]["answer"]
        for row in frame.to_dict("records"):
            raw_answer = _materialize_action(engine, question, candidates, row)
            observed = meteor_local(raw_answer, gold)
            difference = abs(observed - historical[str(row["action_key"])])
            maximum = max(maximum, difference)
            checked += 1
            if difference > tolerance:
                raise RuntimeError(
                    "Raw answer builder no longer reproduces historical target: "
                    f"qid={qid}, action={row['action_key']}, diff={difference}"
                )
    return {"sample_actions": checked, "max_abs_difference": maximum}


def build_or_load_final_targets(
    engine,
    train: Mapping,
    scale_qids: Sequence[str],
    pair_cache: Mapping,
    singleton_cache: Mapping,
    cache_dir: Path,
    checkpoint_every: int = 25,
) -> tuple[pd.DataFrame, dict]:
    payload, signature = training_signature(scale_qids, pair_cache, singleton_cache)
    path = Path(cache_dir) / "selector_v21_final_targets.pkl"
    rows_by_qid: dict[str, list[dict]] = {}
    if path.is_file():
        with path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("version") != FINAL_TARGET_VERSION or cached.get("signature") != signature:
            raise RuntimeError(f"Refusing incompatible final-target cache: {path}")
        rows_by_qid = {str(key): value for key, value in cached.get("rows_by_qid", {}).items()}

    expected_qids = [str(qid) for qid in scale_qids]
    if not set(rows_by_qid) <= set(expected_qids):
        raise RuntimeError("Final-target cache contains qids outside frozen SCALE5000")

    def save() -> None:
        _atomic_pickle({
            "version": FINAL_TARGET_VERSION,
            "signature": signature,
            "signature_payload": payload,
            "ordered_qids": expected_qids,
            "action_keys": list(ACTION_KEYS),
            "postprocess": "best_06252",
            "rows_by_qid": rows_by_qid,
        }, path)

    todo = [qid for qid in expected_qids if qid not in rows_by_qid]
    started = time.time()
    for index, qid in enumerate(todo, start=1):
        try:
            question, candidates = _candidate_dicts(singleton_cache, qid)
            action_frame = build_action_frame(qid, question, candidates)
            gold = train[qid]["answer"]
            records = []
            for row in action_frame.to_dict("records"):
                raw_answer = _materialize_action(engine, question, candidates, row)
                final_answer = postprocess_best_06252(raw_answer, question=question, qid=qid)
                row["final_target"] = meteor_local(final_answer, gold)
                records.append(row)
            if tuple(row["action_key"] for row in records) != ACTION_KEYS:
                raise RuntimeError(f"Final-target action order mismatch for qid={qid}")
            rows_by_qid[qid] = records
        except Exception:
            save()
            raise
        if index % checkpoint_every == 0 or index == len(todo):
            save()
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-9)
            eta = (len(todo) - index) / max(rate, 1e-9)
            print(
                f"[final targets] {index}/{len(todo)} new | total={len(rows_by_qid)}/5000 "
                f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True,
            )
    if set(rows_by_qid) != set(expected_qids):
        raise RuntimeError("Final-target cache is incomplete")
    save()
    frame = canonical_action_sort(pd.DataFrame([
        row for qid in expected_qids for row in rows_by_qid[qid]
    ]))
    validate_action_table(frame, expected_qids, require_target=True)
    return frame, {"path": str(path), "signature": signature, "rows": len(frame)}


def validate_action_table(frame: pd.DataFrame, ordered_qids: Sequence[str], require_target: bool) -> None:
    if len(frame) != len(ordered_qids) * 15 or frame["qid"].nunique() != len(ordered_qids):
        raise RuntimeError("Selector table row/qid count mismatch")
    expected_canonical_qids = sorted(map(str, ordered_qids))
    if frame["qid"].drop_duplicates().astype(str).tolist() != expected_canonical_qids:
        raise RuntimeError("Selector qid order mismatch")
    if not (frame.groupby("qid", sort=False).size() == 15).all():
        raise RuntimeError("Selector table does not contain 15 actions per qid")
    observed = frame.groupby("qid", sort=False)["action_key"].apply(tuple)
    if not all(keys == ACTION_KEYS for keys in observed):
        raise RuntimeError("Selector action key/order mismatch")
    missing = [column for column in ACTION_FEATURES if column not in frame]
    if missing or (require_target and "final_target" not in frame):
        raise RuntimeError(f"Selector table schema mismatch; missing={missing}")


def build_or_load_oof(
    actions: pd.DataFrame,
    scale_qids: Sequence[str],
    cache_dir: Path,
    signature: str,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    all_blocks = []
    qids = [str(qid) for qid in scale_qids]
    for fold in range(1, 6):
        held_qids = [qid for position, qid in enumerate(qids) if position % 5 + 1 == fold]
        training_qids = [qid for position, qid in enumerate(qids) if position % 5 + 1 != fold]
        path = cache_dir / f"selector_v21_oof_fold_{fold}.pkl"
        block = None
        if path.is_file():
            with path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                cached.get("version") == OOF_VERSION
                and cached.get("signature") == signature
                and cached.get("held_qids") == held_qids
            ):
                block = cached.get("frame")
        if block is None:
            print(f"[OOF fold {fold}] fitting on 4000 qids", flush=True)
            training = actions[actions["qid"].isin(training_qids)].copy()
            held = actions[actions["qid"].isin(held_qids)].copy()
            retarget = fit_retarget_models(training)
            heterogeneous = {
                name: fit_heterogeneous_model(name, training) for name in MODEL_NAMES
            }
            block = add_level0_predictions(held, retarget, heterogeneous)
            _atomic_pickle({
                "version": OOF_VERSION,
                "signature": signature,
                "fold": fold,
                "training_qids": training_qids,
                "held_qids": held_qids,
                "frame": block,
            }, path)
            del retarget, heterogeneous
            gc.collect()
        block = canonical_action_sort(block)
        validate_action_table(block, held_qids, require_target=True)
        all_blocks.append(block)
        print(f"[OOF fold {fold}] ready: {len(block)} rows", flush=True)
    oof = canonical_action_sort(pd.concat(all_blocks, ignore_index=True))
    validate_action_table(oof, qids, require_target=True)
    if len(oof) != 75000:
        raise RuntimeError(f"OOF row count must be 75000, got {len(oof)}")
    return oof


def train_selector_v21(
    engine,
    train: Mapping,
    scale_qids: Sequence[str],
    pair_cache: Mapping,
    singleton_cache: Mapping,
    cache_dir: Path,
    bundle_dir: Path,
    checkpoint_every: int = 25,
) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    scale_qids = [str(qid) for qid in scale_qids]
    if len(scale_qids) != 5000 or len(set(scale_qids)) != 5000:
        raise RuntimeError(f"Frozen selector requires exactly 5000 SCALE qids, got {len(scale_qids)}")
    if not set(scale_qids) <= set(map(str, train)):
        raise RuntimeError("SCALE5000 includes qids absent from train.json")
    _payload, signature = training_signature(scale_qids, pair_cache, singleton_cache)
    if manifest_path.is_file():
        _selector, manifest = load_bundle(bundle_dir, verify_hashes=True)
        if manifest.get("training_signature") != signature:
            raise RuntimeError("Existing selector bundle was trained from different cache inputs")
        print(f"[selector V2.1] complete validated bundle found: {bundle_dir}", flush=True)
        return manifest

    reproduction = verify_raw_target_reproduction(
        engine, train, scale_qids, pair_cache, singleton_cache,
    )
    print(f"[selector V2.1] raw target reproduction PASS: {reproduction}", flush=True)
    actions, target_info = build_or_load_final_targets(
        engine, train, scale_qids, pair_cache, singleton_cache, cache_dir,
        checkpoint_every=checkpoint_every,
    )
    oof = build_or_load_oof(actions, scale_qids, cache_dir, signature)
    transformed = add_v21_features(add_meta_transforms(oof), require_target=True)
    print("[selector V2.1] fitting final meta and pairwise regret models", flush=True)
    meta_model = fit_meta_model(transformed)
    pairwise_model = fit_pairwise_model(transformed)
    print("[selector V2.1] fitting 15 full-SCALE level-0 models", flush=True)
    started = time.time()
    retarget_models = fit_retarget_models(actions)
    heterogeneous_models = {
        name: fit_heterogeneous_model(name, actions) for name in MODEL_NAMES
    }
    manifest = save_bundle(
        bundle_dir,
        retarget_models,
        heterogeneous_models,
        meta_model,
        pairwise_model,
        oof_rows=len(oof),
        training_seconds=time.time() - started,
        training_signature=signature,
        target_cache=target_info,
        raw_reproduction=reproduction,
    )
    load_bundle(bundle_dir, verify_hashes=True)
    print(f"[selector V2.1] complete bundle saved: {bundle_dir}", flush=True)
    return manifest
