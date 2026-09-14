import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.engine import LegalQAEngine
from legalqa.postprocess_v87 import postprocess_v87_final_answer


def _candidate(rank: int) -> dict:
    return {
        "doc_id": str(rank),
        "text": f"Văn bản {rank}.\nNội dung chứng cứ {rank}.",
        "emb": 1.0 - rank / 100,
        "ce": 1.0 - rank / 10,
        "kind": "parent",
        "ce_rank": rank,
    }


def _candidate_tuple(rank: int) -> tuple:
    candidate = _candidate(rank)
    return (
        candidate["doc_id"], candidate["text"], candidate["emb"],
        candidate["ce"], candidate["kind"], None,
    )


class _ConstantRanker:
    def predict(self, rows):
        return np.zeros(len(rows), dtype=np.float64)


class V87PostprocessTests(unittest.TestCase):
    def test_final_answer_has_one_exact_v87_conclusion(self):
        raw = (
            "Căn cứ theo quy định tại Điều 12 Luật Ví dụ về quyền lợi như sau:\n"
            "Nội dung chứng cứ được giữ lại.\n"
            "Như vậy, kết luận cũ được quy định tại Điều 12 Luật Ví dụ."
        )
        question = "  Người lao động có quyền lợi gì?  "

        answer = postprocess_v87_final_answer(raw, question)
        conclusions = [
            line for line in answer.splitlines()
            if line.strip().lower().startswith("như vậy")
        ]

        self.assertEqual(len(conclusions), 1)
        self.assertEqual(
            conclusions[0],
            "Như vậy, theo quy định trên thì Người lao động có quyền lợi gì.",
        )
        self.assertTrue(conclusions[0].startswith("Như vậy, theo quy định trên thì"))

    def test_v84_removes_only_redundant_leading_article_marker(self):
        raw = (
            "Căn cứ theo quy định tại Điều 12 Luật Ví dụ về quyền lợi như sau:\n"
            "Điều 12. Nội dung chứng cứ thứ nhất.\n"
            "Dòng giải thích phải được giữ nguyên.\n"
            "Như vậy, kết luận cũ."
        )

        answer = postprocess_v87_final_answer(raw, "Quyền lợi được quy định thế nào?")

        self.assertNotIn("\nĐiều 12. Nội dung chứng cứ", answer)
        self.assertIn("\nNội dung chứng cứ thứ nhất.", answer)
        self.assertIn("Dòng giải thích phải được giữ nguyên.", answer)

    def test_training_target_builders_keep_using_raw_answer(self):
        engine = LegalQAEngine.__new__(LegalQAEngine)
        candidates = [_candidate(rank) for rank in range(1, 6)]
        top5 = [_candidate_tuple(rank) for rank in range(1, 6)]
        engine.candidate_dicts = Mock(return_value=candidates)
        engine.build_answer = Mock(return_value="RAW ANSWER")

        with (
            patch("legalqa.engine.meteor_local", return_value=0.25) as meteor,
            patch("legalqa.engine.postprocess_v87_final_answer") as postprocess,
        ):
            pair_rows = engine.build_pair_rows("qid", "question", "gold", top5)
            singleton_rows = engine.build_singleton_targets("question", "gold", candidates)

        self.assertEqual(len(pair_rows), 10)
        self.assertEqual(len(singleton_rows), 5)
        self.assertEqual(engine.build_answer.call_count, 15)
        self.assertTrue(all(call.args[0] == "RAW ANSWER" for call in meteor.call_args_list))
        postprocess.assert_not_called()

    def test_predict_postprocesses_only_after_action_selection(self):
        engine = LegalQAEngine.__new__(LegalQAEngine)
        candidates = [_candidate(rank) for rank in range(1, 6)]
        top5 = [_candidate_tuple(rank) for rank in range(1, 6)]
        engine.retrieve_top5 = Mock(return_value=(top5, {"top_docs": ["1", "2", "3"]}))
        engine.candidate_dicts = Mock(return_value=candidates)
        engine.build_answer = Mock(return_value="RAW ANSWER")
        models = [_ConstantRanker() for _ in range(5)]

        with patch(
            "legalqa.engine.postprocess_v87_final_answer",
            return_value="V87 ANSWER",
        ) as postprocess:
            answer = engine.predict("Câu hỏi?", models)

        self.assertEqual(answer, "V87 ANSWER")
        engine.build_answer.assert_called_once_with("Câu hỏi?", [top5[0]])
        postprocess.assert_called_once_with("RAW ANSWER", "Câu hỏi?")


if __name__ == "__main__":
    unittest.main()
