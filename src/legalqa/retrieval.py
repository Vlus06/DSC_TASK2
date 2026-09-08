import pickle
from collections import defaultdict
import numpy as np
from .text_utils import vi_tokenize_clean, extract_numbers


def minmax_normalize(items_scores):
    items = list(items_scores.items()) if isinstance(items_scores, dict) else list(items_scores)
    if not items:
        return {}
    scores = [s for _, s in items]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {did: 0.5 for did, _ in items}
    return {did: (s - lo) / (hi - lo) for did, s in items}


def weighted_score_fusion(ranked_lists, weights=None):
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for did, score in minmax_normalize(ranked).items():
            fused[did] += w * score
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(self, cfg, stopwords, doc_id_to_passage, device):
        self.cfg = cfg
        self.stopwords = stopwords
        self.doc_id_to_passage = doc_id_to_passage
        with cfg.bm25_cache.open("rb") as f:
            self.bm25_doc_ids, self.bm25_index = pickle.load(f)
        with cfg.dense_parent_cache.open("rb") as f:
            payload = pickle.load(f)
        self.chunk_doc_ids_all = np.array(payload["chunk_doc_ids_all"])
        self.chunk_texts_all = payload["chunk_texts_all"]
        self.chunk_vecs_all = np.ascontiguousarray(payload["chunk_vecs_all"], dtype=np.float32)
        self.cache_meta = payload.get("meta", {})
        from sentence_transformers import SentenceTransformer
        self.bi_encoder = SentenceTransformer(cfg.bi_encoder_model, device=device, trust_remote_code=True)
        self.bi_encoder.max_seq_length = cfg.bi_encoder_max_seq_length
        self.query_cache = {}
        self.doc_to_chunk_positions = defaultdict(list)
        for i, did in enumerate(self.chunk_doc_ids_all):
            self.doc_to_chunk_positions[str(did)].append(i)

    def get_query_vec(self, query):
        if query in self.query_cache:
            return self.query_cache[query]
        vec = self.bi_encoder.encode([query], show_progress_bar=False, convert_to_numpy=True,
                                     normalize_embeddings=True).astype(np.float32)
        if len(self.query_cache) >= 2000:
            self.query_cache.clear()
        self.query_cache[query] = vec
        return vec

    def bm25_retrieve(self, query, top_k=None):
        top_k = top_k or self.cfg.bm25_top_k
        toks = vi_tokenize_clean(query, self.stopwords)
        scores = self.bm25_index.get_scores(toks)
        top_idx = np.argsort(scores)[::-1][:top_k]
        raw = [(str(self.bm25_doc_ids[i]), float(scores[i])) for i in top_idx]
        q_numbers = set(extract_numbers(query))
        if not q_numbers:
            return sorted(raw, key=lambda x: x[1], reverse=True)
        boosted = []
        for doc_id, score in raw:
            matches = q_numbers & set(extract_numbers(self.doc_id_to_passage.get(doc_id, "")))
            if matches:
                score *= 1 + 0.4 * len(matches)
            boosted.append((doc_id, score))
        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def dense_retrieve_full_corpus(self, query, top_k=None):
        top_k = top_k or self.cfg.dense_top_k
        qv = self.get_query_vec(query)[0]
        sims = self.chunk_vecs_all @ qv
        doc_best = {}
        for pos, sim in enumerate(sims):
            did = str(self.chunk_doc_ids_all[pos])
            if did not in doc_best or sim > doc_best[did]:
                doc_best[did] = float(sim)
        return sorted(doc_best.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def fuse_from_raw(self, bm25_cands, dense_cands, top_n=None):
        top_n = top_n or self.cfg.n_docs_sweep_max
        fused = weighted_score_fusion([bm25_cands, dense_cands], [self.cfg.best_w_bm25, self.cfg.best_w_dense])
        return [did for did, _ in fused[:top_n]]
