import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.engine import ACTION_FEATURES, build_action_frame
from legalqa.meta_selector import (
    GATE_THRESHOLD,
    META_FEATURES,
    V21_FEATURES,
    add_meta_transforms,
    add_v21_features,
    build_regret_pairs,
    gate_choice,
    pairwise_borda_index,
)
from legalqa.model_bundle import (
    BUNDLE_VERSION,
    MANIFEST_NAME,
    expected_model_files,
    load_bundle,
    save_bundle,
)
from legalqa.selector_models import MODEL_NAMES, RETARGET_SEEDS
from legalqa.selector_models import fit_heterogeneous_model, fit_retarget_models


def candidate(rank, doc=None, article=1):
    return {
        "doc_id": str(doc if doc is not None else rank),
        "text": f"Điều {article}. Nội dung chứng cứ {rank}",
        "emb": 1.0 - rank / 100.0,
        "ce": 1.0 - rank / 10.0,
        "kind": "parent",
        "ce_rank": rank,
    }


def prediction_frame(qids=("q1",)):
    frames = []
    for qid in qids:
        frame = build_action_frame(qid, "Câu hỏi pháp luật?", [candidate(i) for i in range(1, 6)])
        for index, seed in enumerate(RETARGET_SEEDS):
            frame[f"base_final_{seed}"] = np.linspace(0, 1, 15) + index / 100
        for index, name in enumerate(MODEL_NAMES):
            frame[f"pred__{name}"] = np.linspace(1, 0, 15) + index / 100
        frame["final_target"] = np.linspace(0.0, 1.0, 15)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class ConstantPairwise:
    def predict_proba(self, values):
        values = np.asarray(values)
        probability = np.where(values[:, 0] >= 0, 0.75, 0.25)
        return np.column_stack([1.0 - probability, probability])


class NativeFakeModel:
    def save_model(self, path):
        Path(path).write_bytes(b"native-model")


class TinyEstimator:
    def fit(self, X, y, **_kwargs):
        self.offset = float(np.nanmean(y))
        return self

    def predict(self, X):
        values = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        return values[:, -1] + self.offset


class SelectorV21Tests(unittest.TestCase):
    def test_action_schema_is_exact_and_deterministic(self):
        frame = build_action_frame("q", "Hỏi gì?", [candidate(i) for i in range(1, 6)])
        self.assertEqual(len(ACTION_FEATURES), 40)
        self.assertEqual(len(frame), 15)
        self.assertEqual(
            frame.action_key.tolist(),
            ["S1", "S2", "S3", "S4", "S5"]
            + [f"P{i}_{j}" for i in range(1, 6) for j in range(i + 1, 6)],
        )

    def test_meta_transforms_and_v21_counts(self):
        transformed = add_meta_transforms(prediction_frame())
        self.assertEqual(len(META_FEATURES), 36)
        self.assertEqual(len(V21_FEATURES), 109)
        for name in MODEL_NAMES:
            self.assertEqual(transformed[f"rank__{name}"].tolist(), list(range(1, 16)))
            self.assertAlmostEqual(transformed[f"prob__{name}"].sum(), 1.0)
            self.assertAlmostEqual(transformed[f"z__{name}"].mean(), 0.0, places=12)
        v21 = add_v21_features(transformed, require_target=True)
        self.assertTrue(all(column in v21 for column in V21_FEATURES))

    def test_safety_requires_same_nonempty_doc_article_and_keeps_current(self):
        frame = prediction_frame()
        # Current is S15 by base mean rank; force controlled metadata around it.
        frame.loc[:, "primary_doc"] = "other"
        frame.loc[:, "first_article"] = ""
        frame.loc[14, ["primary_doc", "first_article"]] = ["D", "9"]
        frame.loc[0, ["primary_doc", "first_article"]] = ["D", "9"]
        frame.loc[1, ["primary_doc", "first_article"]] = ["D", ""]
        result = add_v21_features(add_meta_transforms(frame), require_target=True)
        allowed = result["_allowed"].to_numpy(bool)
        self.assertTrue(allowed[14])
        self.assertTrue(allowed[0])
        self.assertFalse(allowed[1])

    def test_pairwise_rows_are_symmetric(self):
        frame = add_v21_features(add_meta_transforms(prediction_frame()), require_target=True)
        X, y, weight = build_regret_pairs(frame)
        self.assertGreater(len(X), 0)
        np.testing.assert_allclose(X[0], -X[1])
        self.assertEqual(int(y[0]), 1 - int(y[1]))
        self.assertEqual(float(weight[0]), float(weight[1]))

    def test_borda_and_gate_are_deterministic(self):
        features = np.array([[3.0], [2.0], [1.0]])
        chosen = pairwise_borda_index(ConstantPairwise(), features, np.ones(3, bool), 2)
        self.assertEqual(chosen, 0)
        self.assertEqual(gate_choice(0, 1, GATE_THRESHOLD - 1e-9), (0, False))
        self.assertEqual(gate_choice(0, 1, GATE_THRESHOLD), (1, True))
        self.assertEqual(gate_choice(0, 1, GATE_THRESHOLD + 1e-9), (1, True))
        self.assertEqual(gate_choice(1, 1, 0.0), (1, False))

    def test_native_bundle_roundtrip_manifest(self):
        retarget = [NativeFakeModel() for _ in RETARGET_SEEDS]
        heterogeneous = {name: NativeFakeModel() for name in MODEL_NAMES}
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            manifest = save_bundle(
                bundle, retarget, heterogeneous, NativeFakeModel(), NativeFakeModel(),
                oof_rows=75000, training_seconds=1.0,
                training_signature="a" * 64,
                target_cache={"signature": "a" * 64, "rows": 75000},
                raw_reproduction={"sample_actions": 300, "max_abs_difference": 0.0},
            )
            self.assertEqual(manifest["version"], BUNDLE_VERSION)
            self.assertEqual(manifest["model_count"], 17)
            self.assertEqual(set(path.name for path in bundle.iterdir()), set(expected_model_files()) | {MANIFEST_NAME})
            sentinel = object()
            with patch("legalqa.model_bundle.load_native_model", return_value=sentinel):
                selector, loaded = load_bundle(bundle, verify_hashes=True)
            self.assertEqual(loaded["version"], BUNDLE_VERSION)
            self.assertEqual(len(selector.retarget_models), 5)
            self.assertEqual(tuple(selector.heterogeneous_models), MODEL_NAMES)

    def test_tiny_level0_training_smoke(self):
        frame = prediction_frame(("q1", "q2")).drop(
            columns=[column for column in prediction_frame().columns if column.startswith("base_final_") or column.startswith("pred__")]
        )
        with patch("legalqa.selector_models.make_retarget_xgb", side_effect=lambda _seed: TinyEstimator()):
            retarget = fit_retarget_models(frame)
        with patch("legalqa.selector_models.make_heterogeneous_model", return_value=TinyEstimator()):
            heterogeneous = {name: fit_heterogeneous_model(name, frame) for name in MODEL_NAMES}
        self.assertEqual(len(retarget), 5)
        self.assertEqual(tuple(heterogeneous), MODEL_NAMES)


if __name__ == "__main__":
    unittest.main()
