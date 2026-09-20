"""Frozen level-0 model factories for selector V2.1."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .engine import ACTION_FEATURES, canonical_action_sort


RETARGET_SEEDS = (42, 10042, 20042, 30042, 40042)
MODEL_NAMES = (
    "xgb_pair_s1",
    "lgb_rank_s1",
    "cat_yeti_s1",
    "xgb_pair_s2",
    "xgb_ndcg",
    "lgb_deep",
    "cat_pair",
    "xgb_reg",
    "lgb_reg",
    "cat_reg",
)
REGRESSION_MODELS = frozenset({"xgb_reg", "lgb_reg", "cat_reg"})
CATBOOST_RANK_MODELS = frozenset({"cat_yeti_s1", "cat_pair"})


def make_retarget_xgb(seed: int):
    from xgboost import XGBRanker

    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_estimators=350,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def make_heterogeneous_model(name: str):
    if name == "xgb_pair_s1":
        return make_retarget_xgb(42)
    if name == "xgb_pair_s2":
        return make_retarget_xgb(10042)
    if name == "xgb_ndcg":
        from xgboost import XGBRanker

        return XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            n_estimators=350,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgb_reg":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=450,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    if name in {"lgb_rank_s1", "lgb_deep", "lgb_reg"}:
        from lightgbm import LGBMRanker, LGBMRegressor

        if name == "lgb_rank_s1":
            return LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=350,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                reg_alpha=0.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                label_gain=list(range(32)),
            )
        if name == "lgb_deep":
            return LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=450,
                learning_rate=0.025,
                max_depth=6,
                num_leaves=31,
                min_child_samples=8,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                random_state=10042,
                n_jobs=-1,
                verbosity=-1,
                label_gain=list(range(32)),
            )
        return LGBMRegressor(
            objective="regression",
            n_estimators=450,
            learning_rate=0.025,
            max_depth=6,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if name in {"cat_yeti_s1", "cat_pair", "cat_reg"}:
        from catboost import CatBoostRanker, CatBoostRegressor

        if name == "cat_yeti_s1":
            return CatBoostRanker(
                loss_function="YetiRank",
                eval_metric="NDCG",
                iterations=350,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=False,
                thread_count=-1,
                allow_writing_files=False,
            )
        if name == "cat_pair":
            return CatBoostRanker(
                loss_function="PairLogitPairwise",
                eval_metric="NDCG",
                iterations=350,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=10042,
                verbose=False,
                thread_count=-1,
                allow_writing_files=False,
            )
        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=450,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        )
    raise KeyError(f"Unknown heterogeneous model: {name}")


def dense_relevance_target(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.groupby("qid", sort=False)["final_target"]
        .rank(method="dense", ascending=True)
        .astype(int)
        .sub(1)
        .to_numpy(dtype=np.int64)
    )


def _groups(frame: pd.DataFrame) -> list[int]:
    groups = frame.groupby("qid", sort=False).size().tolist()
    if not groups or not all(size == 15 for size in groups):
        raise RuntimeError("Every selector training qid must contain exactly 15 actions")
    return groups


def fit_retarget_models(frame: pd.DataFrame) -> list:
    frame = canonical_action_sort(frame.copy())
    X = frame[ACTION_FEATURES]
    y = frame["final_target"].to_numpy(dtype=float)
    groups = _groups(frame)
    models = []
    for seed in RETARGET_SEEDS:
        model = make_retarget_xgb(seed)
        model.fit(X, y, group=groups, verbose=False)
        models.append(model)
    return models


def fit_heterogeneous_model(name: str, frame: pd.DataFrame):
    frame = canonical_action_sort(frame.copy())
    X = frame[ACTION_FEATURES]
    groups = _groups(frame)
    model = make_heterogeneous_model(name)
    if name in REGRESSION_MODELS:
        model.fit(X, frame["final_target"].to_numpy(dtype=float))
    elif name in CATBOOST_RANK_MODELS:
        model.fit(
            X,
            dense_relevance_target(frame),
            group_id=frame["qid"].astype(str).to_numpy(),
        )
    elif name.startswith("xgb"):
        model.fit(X, dense_relevance_target(frame), group=groups, verbose=False)
    else:
        model.fit(X, dense_relevance_target(frame), group=groups)
    return model


def add_level0_predictions(
    frame: pd.DataFrame,
    retarget_models: Sequence,
    heterogeneous_models: Mapping[str, object],
) -> pd.DataFrame:
    if len(retarget_models) != len(RETARGET_SEEDS):
        raise RuntimeError("Selector requires five retarget models")
    if tuple(heterogeneous_models) != MODEL_NAMES:
        raise RuntimeError("Heterogeneous model order does not match frozen MODEL_NAMES")

    out = canonical_action_sort(frame.copy())
    X = out[ACTION_FEATURES]
    for seed, model in zip(RETARGET_SEEDS, retarget_models):
        out[f"base_final_{seed}"] = np.asarray(model.predict(X), dtype=float)
    for name in MODEL_NAMES:
        out[f"pred__{name}"] = np.asarray(
            heterogeneous_models[name].predict(X), dtype=float,
        )
    return out
