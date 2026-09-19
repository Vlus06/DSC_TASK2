import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from legalqa.artifacts import load_evaluation_questions
from legalqa.engine import LegalQAEngine
from pipeline import inference_artifact_names, run_public_inference


class PrivateInferenceTests(unittest.TestCase):
    def test_loads_private_questions_without_replacing_public(self):
        with tempfile.TemporaryDirectory() as directory:
            task2 = Path(directory)
            (task2 / "public-official.json").write_text(
                json.dumps({"public": {"question": "Public?"}}), encoding="utf-8"
            )
            (task2 / "private-official.json").write_text(
                json.dumps({"private": {"question": "Private?"}}), encoding="utf-8"
            )

            public = load_evaluation_questions(task2, "public")
            private = load_evaluation_questions(task2, "private")

        self.assertEqual(list(public), ["public"])
        self.assertEqual(list(private), ["private"])

    def test_private_artifact_names_are_isolated(self):
        self.assertEqual(
            inference_artifact_names("private", True, 1918),
            (
                "inference_private_progress.pkl",
                "submission_private.json",
                "inference_log_private.json",
            ),
        )

    def test_sparse_pool_with_six_candidates_still_returns_top_five(self):
        engine = LegalQAEngine.__new__(LegalQAEngine)
        engine.cfg = SimpleNamespace(
            final_top_docs=3,
            pre_ce_top_k=10,
            ce_top_k=5,
        )
        pool = [
            (str(index), f"chunk {index}", float(10 - index), "parent", None)
            for index in range(6)
        ]
        engine.top_docs = Mock(return_value=["1", "2", "3"])
        engine.get_candidate_pool = Mock(return_value=pool)
        engine.ce_score_candidates = Mock(
            side_effect=lambda _question, candidates: [
                (doc_id, text, emb, float(emb), kind, dieu)
                for doc_id, text, emb, kind, dieu in candidates
            ]
        )

        top5, debug = engine.retrieve_top5("question")

        self.assertEqual(len(top5), 5)
        self.assertEqual(debug["pre_ce_count"], 6)

    def test_resume_retries_only_failed_or_empty_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            progress = output / "inference_private_progress.pkl"
            with progress.open("wb") as handle:
                pickle.dump(
                    {
                        "version": "legalqa_inference_progress_v1",
                        "predictions": {
                            "ok": {"answer": "existing"},
                            "retry": {"answer": ""},
                        },
                        "inference_log": [
                            {"qid": "ok", "pred_len": 8, "error": None},
                            {"qid": "retry", "pred_len": 0, "error": "boom"},
                        ],
                    },
                    handle,
                )
            engine = Mock()
            engine.predict.return_value = (
                "recovered",
                {
                    "member_actions": [],
                    "ensemble_action": [1, 1, 0],
                    "ensemble_action_size": 1,
                    "ensemble_i": 1,
                    "ensemble_j": 0,
                    "selected_singleton": True,
                    "seed_unique_actions": 1,
                    "ensemble_differs_from_member0": False,
                    "ensemble_mean_rank": 1.0,
                },
            )
            evaluation = {
                "ok": {"question": "already complete"},
                "retry": {"question": "retry me"},
            }

            submission, _ = run_public_inference(
                SimpleNamespace(),
                engine,
                [],
                evaluation,
                output,
                dataset_name="private",
                checkpoint_every=1,
            )
            predictions = json.loads(submission.read_text(encoding="utf-8"))

            engine.predict.assert_called_once_with(
                "retry me", [], qid="retry", return_debug=True,
            )
        self.assertEqual(predictions["ok"]["answer"], "existing")
        self.assertEqual(predictions["retry"]["answer"], "recovered")
        self.assertEqual(
            inference_artifact_names("private", False, 10),
            (
                "inference_private_smoke_progress.pkl",
                "submission_private_smoke_10.json",
                "inference_log_private_smoke_10.json",
            ),
        )


if __name__ == "__main__":
    unittest.main()
