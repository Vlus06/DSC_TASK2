import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalqa.child_cache import load_or_build_child_cache
from legalqa.chunking import build_hierarchical_chunks_for_doc


class ChildCacheTests(unittest.TestCase):
    def test_build_format_and_reuse_without_encoding(self):
        passages = {"1": "Điều 1. Quy định\n1. Nội dung thứ nhất.\n2. Nội dung thứ hai."}
        names = {"1": "Văn bản thử"}
        expected = build_hierarchical_chunks_for_doc("1", passages["1"], names["1"])
        self.assertTrue(expected)
        encoder = Mock()
        encoder.encode.return_value = np.ones((len(expected), 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(child_cache=Path(directory) / "child.pkl")
            texts, vectors, meta = load_or_build_child_cache(cfg, passages, names, encoder, 2)
            self.assertEqual(texts, [c["text"] for c in expected])
            self.assertEqual(meta, [{k: c[k] for k in (
                "doc_id", "dieu_num", "khoan_num", "raw_khoan_text",
            )} for c in expected])
            self.assertEqual(encoder.encode.call_args.kwargs["batch_size"], 2)
            self.assertFalse(cfg.child_cache.with_suffix(".pkl.tmp").exists())
            with cfg.child_cache.open("rb") as f:
                self.assertEqual(set(pickle.load(f)), {"child_texts", "child_vecs", "child_meta"})
            encoder.reset_mock()
            restored = load_or_build_child_cache(cfg, {}, {}, encoder)
            encoder.encode.assert_not_called()
            np.testing.assert_array_equal(vectors, restored[1])
            self.assertEqual(meta, restored[2])

    def test_failed_save_does_not_publish_cache(self):
        encoder = Mock()
        encoder.encode.return_value = np.empty((0, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(child_cache=Path(directory) / "child.pkl")
            with patch("legalqa.child_cache.pickle.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    load_or_build_child_cache(cfg, {}, {}, encoder)
            self.assertFalse(cfg.child_cache.exists())


if __name__ == "__main__":
    unittest.main()
