"""Frozen level-2 selector, safety mask and pairwise gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .engine import ACTION_FEATURES, canonical_action_sort
from .selector_models import MODEL_NAMES, RETARGET_SEEDS, add_level0_predictions


GATE_THRESHOLD = 0.55
BASE_FINAL_COLS = tuple(f"base_final_{seed}" for seed in RETARGET_SEEDS)
META_FEATURES = tuple(
    [f"invrank__{name}" for name in MODEL_NAMES]
    + [f"z__{name}" for name in MODEL_NAMES]
    + [f"prob__{name}" for name in MODEL_NAMES]
    + [f"base_invrank__{index}" for index in range(len(RETARGET_SEEDS))]
    + ["action_size"]
)
CONSENSUS_FEATURES = (
    "het_rank_mean",
    "het_rank_std",
    "het_rank_min",
    "het_rank_max",
    "het_top1_votes",
    "het_top3_votes",
    "het_invrank_mean",
    "het_invrank_std",
    "het_z_mean",
    "het_z_std",
    "het_z_max",
    "het_prob_mean",
    "het_prob_std",
    "het_prob_max",
    "base_rank_mean",
    "base_rank_std",
    "base_rank_min",
    "base_rank_max",
    "base_top1_votes",
    "base_invrank_mean",
    "base_invrank_std",
    "_is_current",
)
CURRENT_REF_SOURCE_COLS = (
    "het_rank_mean",
    "het_z_mean",
    "het_prob_mean",
    "base_rank_mean",
    "action_size",
    "ce_mean",
    "emb_mean",
    "pair_query_coverage",
    "c1_q_overlap",
    "c1_q_jaccard",
    "combined_len",
    "rank_sum",
)
CURRENT_DELTA_FEATURES = tuple(
    f"delta_current__{column}" for column in CURRENT_REF_SOURCE_COLS
)


def _deduplicate(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


V21_FEATURES = _deduplicate(
    META_FEATURES + tuple(ACTION_FEATURES) + CONSENSUS_FEATURES + CURRENT_DELTA_FEATURES
)

assert len(META_FEATURES) == 36
assert len(V21_FEATURES) == 109


@dataclass(frozen=True)
class SelectionResult:
    retarget_action: str
    baseline_meta_action: str
    pairwise_action: str
    final_action: str
    direct_prob: float
    used_pairwise_gate: bool
    allowed_action_count: int
    action_size: int
    action_i: int
    action_j: int


def add_meta_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    frame = canonical_action_sort(frame.copy())
    required = list(BASE_FINAL_COLS) + [f"pred__{name}" for name in MODEL_NAMES]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Missing level-0 prediction columns: {missing}")

    base_rank_cols = []
    for index, column in enumerate(BASE_FINAL_COLS):
        rank_column = f"_base_rank_{index}"
        frame[rank_column] = frame.groupby("qid", sort=False)[column].rank(
            method="first", ascending=False,
        )
        frame[f"base_invrank__{index}"] = 1.0 / frame[rank_column]
        base_rank_cols.append(rank_column)
    frame["_base_mean_rank"] = frame[base_rank_cols].mean(axis=1)

    for name in MODEL_NAMES:
        prediction_column = f"pred__{name}"
        rank_column = f"rank__{name}"
        z_column = f"z__{name}"
        frame[rank_column] = frame.groupby("qid", sort=False)[prediction_column].rank(
            method="first", ascending=False,
        )
        frame[f"invrank__{name}"] = 1.0 / frame[rank_column]
        mean = frame.groupby("qid", sort=False)[prediction_column].transform("mean")
        std = (
            frame.groupby("qid", sort=False)[prediction_column]
            .transform("std")
            .replace(0, np.nan)
        )
        frame[z_column] = ((frame[prediction_column] - mean) / std).fillna(0.0)
        maximum = frame.groupby("qid", sort=False)[z_column].transform("max")
        exponent = np.exp(frame[z_column] - maximum)
        denominator = exponent.groupby(frame["qid"], sort=False).transform("sum")
        frame[f"prob__{name}"] = exponent / denominator
    return frame


def add_v21_features(frame: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    frame = canonical_action_sort(frame.copy())
    rank_columns = [f"rank__{name}" for name in MODEL_NAMES]
    inverse_columns = [f"invrank__{name}" for name in MODEL_NAMES]
    z_columns = [f"z__{name}" for name in MODEL_NAMES]
    probability_columns = [f"prob__{name}" for name in MODEL_NAMES]
    base_rank_columns = [f"_base_rank_{index}" for index in range(len(RETARGET_SEEDS))]
    base_inverse_columns = [
        f"base_invrank__{index}" for index in range(len(RETARGET_SEEDS))
    ]

    frame["het_rank_mean"] = frame[rank_columns].mean(axis=1)
    frame["het_rank_std"] = frame[rank_columns].std(axis=1).fillna(0.0)
    frame["het_rank_min"] = frame[rank_columns].min(axis=1)
    frame["het_rank_max"] = frame[rank_columns].max(axis=1)
    frame["het_top1_votes"] = (frame[rank_columns] == 1.0).sum(axis=1)
    frame["het_top3_votes"] = (frame[rank_columns] <= 3.0).sum(axis=1)
    frame["het_invrank_mean"] = frame[inverse_columns].mean(axis=1)
    frame["het_invrank_std"] = frame[inverse_columns].std(axis=1).fillna(0.0)
    frame["het_z_mean"] = frame[z_columns].mean(axis=1)
    frame["het_z_std"] = frame[z_columns].std(axis=1).fillna(0.0)
    frame["het_z_max"] = frame[z_columns].max(axis=1)
    frame["het_prob_mean"] = frame[probability_columns].mean(axis=1)
    frame["het_prob_std"] = frame[probability_columns].std(axis=1).fillna(0.0)
    frame["het_prob_max"] = frame[probability_columns].max(axis=1)
    frame["base_rank_mean"] = frame[base_rank_columns].mean(axis=1)
    frame["base_rank_std"] = frame[base_rank_columns].std(axis=1).fillna(0.0)
    frame["base_rank_min"] = frame[base_rank_columns].min(axis=1)
    frame["base_rank_max"] = frame[base_rank_columns].max(axis=1)
    frame["base_top1_votes"] = (frame[base_rank_columns] == 1.0).sum(axis=1)
    frame["base_invrank_mean"] = frame[base_inverse_columns].mean(axis=1)
    frame["base_invrank_std"] = frame[base_inverse_columns].std(axis=1).fillna(0.0)

    qids = frame["qid"].drop_duplicates().tolist()
    qid_count = len(qids)
    if len(frame) != qid_count * 15:
        raise RuntimeError("Selector frame must contain exactly 15 actions per qid")
    positions = np.arange(qid_count)
    base_metric = frame["_base_mean_rank"].to_numpy(float).reshape(qid_count, 15)
    current_index = np.argmin(base_metric, axis=1)
    is_current = np.zeros((qid_count, 15), dtype=np.int8)
    is_current[positions, current_index] = 1
    frame["_is_current"] = is_current.reshape(-1)

    documents = frame["primary_doc"].astype(str).to_numpy().reshape(qid_count, 15)
    articles = (
        frame["first_article"].fillna("").astype(str).to_numpy().reshape(qid_count, 15)
    )
    current_documents = documents[positions, current_index]
    current_articles = articles[positions, current_index]
    allowed = (
        (documents == current_documents[:, None])
        & (articles == current_articles[:, None])
        & (articles != "")
    )
    allowed[positions, current_index] = True
    frame["_allowed"] = allowed.reshape(-1)

    if require_target and "final_target" not in frame:
        raise RuntimeError("final_target is required for pairwise training")
    for column in CURRENT_REF_SOURCE_COLS:
        values = frame[column].to_numpy(float).reshape(qid_count, 15)
        current = values[positions, current_index]
        frame[f"delta_current__{column}"] = (values - current[:, None]).reshape(-1)

    missing = [column for column in V21_FEATURES if column not in frame]
    if missing:
        raise RuntimeError(f"Missing V2.1 features: {missing}")
    return frame


def make_meta_model():
    from xgboost import XGBRanker

    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_estimators=240,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        reg_alpha=0.1,
        tree_method="hist",
        random_state=2026,
        n_jobs=-1,
    )


def fit_meta_model(frame: pd.DataFrame):
    frame = canonical_action_sort(frame.copy())
    groups = frame.groupby("qid", sort=False).size().tolist()
    if not all(size == 15 for size in groups):
        raise RuntimeError("Meta training requires 15 actions per qid")
    target = (
        frame.groupby("qid", sort=False)["final_target"]
        .rank(method="dense", ascending=True)
        .astype(int)
        .sub(1)
        .to_numpy(dtype=np.int64)
    )
    model = make_meta_model()
    model.fit(frame[list(META_FEATURES)].to_numpy(float), target, group=groups, verbose=False)
    return model


def make_pairwise_model():
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=220,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        reg_alpha=0.1,
        tree_method="hist",
        random_state=2026,
        n_jobs=-1,
    )


def build_regret_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = frame[frame["_allowed"]].copy()
    rows, labels, weights = [], [], []
    for _qid, group in safe.groupby("qid", sort=False):
        group = canonical_action_sort(group)
        features = group[list(V21_FEATURES)].to_numpy(float)
        target = group["final_target"].to_numpy(float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                regret = float(target[left] - target[right])
                if abs(regret) <= 1e-12:
                    continue
                difference = features[left] - features[right]
                label = int(regret > 0)
                weight = abs(regret)
                rows.extend((difference, -difference))
                labels.extend((label, 1 - label))
                weights.extend((weight, weight))
    if not labels:
        raise RuntimeError("No V2.1 regret pairs generated")
    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=np.int8)
    sample_weight = np.asarray(weights, dtype=float)
    sample_weight /= max(1e-12, float(sample_weight.mean()))
    return X, y, sample_weight


def fit_pairwise_model(frame: pd.DataFrame):
    X, y, sample_weight = build_regret_pairs(frame)
    model = make_pairwise_model()
    model.fit(X, y, sample_weight=sample_weight)
    return model


def pairwise_borda_index(
    pairwise_model,
    feature_matrix: np.ndarray,
    allowed: np.ndarray,
    current_index: int,
) -> int:
    indices = np.flatnonzero(allowed)
    if len(indices) == 1:
        return int(current_index)
    differences, pairs = [], []
    for left_pos in range(len(indices)):
        for right_pos in range(left_pos + 1, len(indices)):
            left, right = int(indices[left_pos]), int(indices[right_pos])
            differences.append(feature_matrix[left] - feature_matrix[right])
            pairs.append((left, right))
    probabilities = pairwise_model.predict_proba(np.asarray(differences, dtype=float))[:, 1]
    scores = np.full(len(feature_matrix), -np.inf, dtype=float)
    scores[indices] = 0.0
    for probability, (left, right) in zip(probabilities, pairs):
        scores[left] += float(probability)
        scores[right] += float(1.0 - probability)
    scores[int(current_index)] += 1e-12
    return int(np.argmax(scores))


def gate_choice(
    baseline_index: int,
    pairwise_index: int,
    direct_probability: float,
    threshold: float = GATE_THRESHOLD,
) -> tuple[int, bool]:
    if pairwise_index == baseline_index:
        return int(baseline_index), False
    if direct_probability >= threshold:
        return int(pairwise_index), True
    return int(baseline_index), False


class FrozenSelectorV21:
    """Inference interface for the complete frozen selector bundle."""

    def __init__(
        self,
        retarget_models: Sequence,
        heterogeneous_models: Mapping[str, object],
        meta_model,
        pairwise_model,
    ) -> None:
        self.retarget_models = list(retarget_models)
        self.heterogeneous_models = dict(heterogeneous_models)
        self.meta_model = meta_model
        self.pairwise_model = pairwise_model

    def select_action(self, action_frame: pd.DataFrame) -> SelectionResult:
        frame = add_level0_predictions(
            action_frame, self.retarget_models, self.heterogeneous_models,
        )
        frame = add_v21_features(add_meta_transforms(frame), require_target=False)
        if frame["qid"].nunique() != 1 or len(frame) != 15:
            raise RuntimeError("select_action expects exactly one qid with 15 actions")

        current_index = int(np.argmax(frame["_is_current"].to_numpy(int)))
        allowed = frame["_allowed"].to_numpy(bool)
        meta_strength = np.asarray(
            self.meta_model.predict(frame[list(META_FEATURES)].to_numpy(float)),
            dtype=float,
        )
        baseline_index = int(np.argmax(np.where(allowed, meta_strength, -np.inf)))
        v21_matrix = frame[list(V21_FEATURES)].to_numpy(float)
        pairwise_index = pairwise_borda_index(
            self.pairwise_model, v21_matrix, allowed, current_index,
        )
        direct_probability = 1.0
        if pairwise_index != baseline_index:
            difference = v21_matrix[pairwise_index] - v21_matrix[baseline_index]
            direct_probability = float(
                self.pairwise_model.predict_proba(difference.reshape(1, -1))[0, 1]
            )
        final_index, used_gate = gate_choice(
            baseline_index, pairwise_index, direct_probability,
        )
        rows = frame.reset_index(drop=True)

        def action(index: int) -> str:
            return str(rows.iloc[index]["action_key"])

        chosen = rows.iloc[final_index]
        return SelectionResult(
            retarget_action=action(current_index),
            baseline_meta_action=action(baseline_index),
            pairwise_action=action(pairwise_index),
            final_action=action(final_index),
            direct_prob=direct_probability,
            used_pairwise_gate=used_gate,
            allowed_action_count=int(allowed.sum()),
            action_size=int(chosen["action_size"]),
            action_i=int(chosen["action_i"]),
            action_j=int(chosen["action_j"]),
        )
