"""Build and load hierarchical child-chunk embeddings."""
import pickle

import numpy as np

from .chunking import build_hierarchical_chunks_for_doc


def load_or_build_child_cache(cfg, passages, names, encoder, batch_size=64):
    path = cfg.child_cache
    if path.exists():
        print(f"Using existing child cache: {path}", flush=True)
        with path.open("rb") as f:
            payload = pickle.load(f)
        return (
            payload["child_texts"],
            np.ascontiguousarray(payload["child_vecs"], dtype=np.float32),
            payload["child_meta"],
        )

    chunks = []
    print(f"Building child chunks from {len(passages)} documents", flush=True)
    for i, doc_id in enumerate(passages, 1):
        chunks.extend(build_hierarchical_chunks_for_doc(
            doc_id, passages[doc_id], names.get(doc_id, ""),
        ))
        if i % 500 == 0:
            print(f"Child chunking: {i}/{len(passages)} documents", flush=True)

    print(f"Child chunks: {len(chunks)} (expected: 595597)", flush=True)
    if len(passages) == 8532 and len(chunks) != 595597:
        raise RuntimeError(
            f"Child chunk count mismatch: {len(chunks)} != 595597. "
            "Check the corpus and chunking settings."
        )

    texts = [c["text"] for c in chunks]
    meta = [{
        "doc_id": c["doc_id"],
        "dieu_num": c["dieu_num"],
        "khoan_num": c["khoan_num"],
        "raw_khoan_text": c["raw_khoan_text"],
    } for c in chunks]
    del chunks
    vecs = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completely written cache; interrupted writes can be rebuilt.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as f:
        pickle.dump({"child_texts": texts, "child_vecs": vecs, "child_meta": meta}, f)
    temporary_path.replace(path)
    print(f"Saved child cache: {path}", flush=True)
    return texts, vecs, meta
