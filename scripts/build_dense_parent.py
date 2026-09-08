#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.config import PipelineConfig
from legalqa.corpus import load_corpus
from legalqa.parent_cache import chunk_by_dieu_khoan, cache_meta, expected_cache_filename


def main():
    cfg = PipelineConfig()
    if cfg.dense_parent_cache.name != expected_cache_filename():
        raise RuntimeError(
            f"Dense cache filename mismatch: config={cfg.dense_parent_cache.name}, "
            f"v4.1={expected_cache_filename()}"
        )

    passages, names = load_corpus()
    chunk_doc_ids_all, chunk_texts_all = [], []
    for doc_id, passage in passages.items():
        for chunk in chunk_by_dieu_khoan(passage, doc_id, names.get(doc_id, ""), max_chars=5117):
            chunk_doc_ids_all.append(chunk["doc_id"])
            chunk_texts_all.append(chunk["text"])

    print(f"Parent chunks: {len(chunk_texts_all)} (expected notebook value: 738506)")
    if len(passages) == 8532 and len(chunk_texts_all) != 738506:
        raise RuntimeError(
            "Parent chunk count does not match the original v4.1 notebook (738506). "
            "Stop here: data/order/text preprocessing is not identical."
        )

    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg.bi_encoder_model, device=device, trust_remote_code=True)
    model.max_seq_length = cfg.bi_encoder_max_seq_length
    vecs = model.encode(
        chunk_texts_all,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cfg.dense_parent_cache.parent.mkdir(parents=True, exist_ok=True)
    with cfg.dense_parent_cache.open("wb") as f:
        pickle.dump({
            "chunk_doc_ids_all": chunk_doc_ids_all,
            "chunk_texts_all": chunk_texts_all,
            "chunk_vecs_all": vecs,
            "meta": cache_meta(len(passages)),
        }, f)
    print(f"Saved: {cfg.dense_parent_cache}")


if __name__ == "__main__":
    main()
