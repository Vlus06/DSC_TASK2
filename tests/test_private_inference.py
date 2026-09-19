import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from legalqa.artifacts import load_evaluation_questions
from pipeline import inference_artifact_names


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
