from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..answer.answer_builder import AnswerBuilder
from ..answer.chunk_selector import ChunkSelector
from ..config import PipelineConfig
from ..corpus import Corpus, load_train_json
from ..eval.meteor import MeteorScorer
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseChunkCache, DenseRetriever
from ..retrieval.fusion import RecencyBooster, ScoreFusion
from ..retrieval.reranker import CrossEncoderReranker
from ..tokenization import VietnameseTokenizer
from ..utils import free_memory, get_device, logger, seed_everything


class PipelineTuner:
    """Reproduces the staged grid-search from the v9 notebook:

      1. sweep W_BM25 / W_DENSE (embedding-only eval)
      2. sweep TOP_N_DOCS
      3. sweep cross-encoder TOP_K_CANDIDATES, then KEEP_TOP_N / dynamic
         margin, then alpha-blend
      4. sweep POST_PROCESS_MAX_CHARS
      5. dedupe on/off

    Each stage's winner is written back into `config` in place, so after
    `.run()` the passed-in `PipelineConfig` holds the tuned values and can
    be handed straight to `QAPipeline` for inference / submission.
    """

    def __init__(
        self,
        config: PipelineConfig,
        corpus: Corpus,
        bm25: BM25Retriever,
        dense_chunk_cache: DenseChunkCache,
        bi_encoder,
    ):
        self.config = config
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = DenseRetriever(bi_encoder, dense_chunk_cache, max_seq_length=config.models.encode_max_seq_length)
        self.tokenizer = VietnameseTokenizer(stopwords_file=config.paths.stopwords_file)
        self.recency_booster = RecencyBooster(
            corpus.doc_id_to_name,
            max_boost=config.retrieval.recency_max_boost,
            base_year=config.retrieval.recency_base_year,
        )
        self.fusion = ScoreFusion(self.recency_booster)
        self.chunk_selector = ChunkSelector(self.tokenizer, lex_tiebreak_margin=config.answer.lex_tiebreak_margin)
        self.answer_builder = AnswerBuilder(corpus.doc_id_to_passage, has_chunk_prefix=self.dense.has_prefix)
        self.meteor = MeteorScorer()
        self.device = get_device(config.models.device)
        self.log: Dict = {}

    # ---------------------------------------------------------------
    def _build_val_split(self) -> List[Tuple[str, str, str]]:
        train_raw = load_train_json(self.config.paths.train_path)
        all_qids = list(train_raw.keys())
        rng = random.Random(self.config.tuning.seed)
        rng.shuffle(all_qids)
        val_qids = all_qids[: self.config.tuning.val_size]
        out = []
        for qid in val_qids:
            item = train_raw[qid]
            gold = item.get("answer", "") or item.get("gold_answer", "") or ""
            out.append((qid, item["question"], gold))
        logger.info(f"Validation split: {len(out)} questions (seed={self.config.tuning.seed}).")
        return out

    def _compute_raw_candidates(self, val_split):
        rcfg = self.config.retrieval
        raw = []
        t0 = time.time()
        for i, (qid, question, gold) in enumerate(tqdm(val_split, desc="BM25+Dense candidates")):
            bm25_cands = self.bm25.retrieve(question, top_k=rcfg.bm25_top_k)
            dense_cands = self.dense.retrieve_doc_level(question, top_k=rcfg.dense_top_k)
            raw.append((qid, question, gold, bm25_cands, dense_cands))
            if (i + 1) % 100 == 0:
                free_memory()
        logger.info(f"Computed raw candidates for {len(raw)} questions in {time.time() - t0:.1f}s.")
        return raw

    # ---------------------------------------------------------------
    def _eval_weight_pair(self, w_bm25, w_dense, raw_cache, n_docs) -> float:
        rcfg = self.config.retrieval
        acfg = self.config.answer
        scores = []
        for qid, question, gold, bm25_cands, dense_cands in raw_cache:
            top_docs = self.fusion.fuse(bm25_cands, dense_cands, question, w_bm25, w_dense, top_n=n_docs)
            if not top_docs:
                pred = ""
            else:
                scored_chunks = self.dense.get_all_scored_chunks(question, top_docs)
                kept = self.chunk_selector.apply_threshold(
                    scored_chunks,
                    acfg.emb_threshold,
                    max_total_chunks=acfg.emb_max_total_chunks,
                    min_kept_fallback=acfg.emb_min_kept_fallback,
                    question=question,
                    use_lexical_tiebreak=True,
                )
                top1_passage = self.corpus.get_passage(top_docs[0])
                pred = self.answer_builder.build(
                    question, kept, top1_full_passage=top1_passage, max_chars=acfg.post_process_max_chars, use_dedupe=True
                )
            scores.append(self.meteor.score(pred, gold))
        return float(np.mean(scores))

    def _sweep_fusion_weights(self, raw_cache):
        tcfg = self.config.tuning
        results = []
        for w in tcfg.w_bm25_candidates:
            m = self._eval_weight_pair(w, tcfg.w_dense_fixed, raw_cache, self.config.retrieval.n_docs_sweep_max)
            results.append({"w_bm25": w, "w_dense": tcfg.w_dense_fixed, "meteor": m})
            logger.info(f"  W_BM25={w:.2f} W_DENSE={tcfg.w_dense_fixed:.2f} | METEOR={m:.4f}")
        best = max(results, key=lambda r: r["meteor"])
        self.config.retrieval.w_bm25 = best["w_bm25"]
        self.config.retrieval.w_dense = best["w_dense"]
        self.log["weight_sweep_results"] = results
        logger.info(f">>> BEST W_BM25={best['w_bm25']} W_DENSE={best['w_dense']} (METEOR={best['meteor']:.4f})")
        return results

    def _build_val_cache(self, raw_cache):
        rcfg = self.config.retrieval
        val_cache = []
        for qid, question, gold, bm25_cands, dense_cands in raw_cache:
            top_docs_full = self.fusion.fuse(
                bm25_cands, dense_cands, question, rcfg.w_bm25, rcfg.w_dense, top_n=rcfg.n_docs_sweep_max
            )
            scored_chunks_full = self.dense.get_all_scored_chunks(question, top_docs_full) if top_docs_full else []
            val_cache.append((qid, question, gold, scored_chunks_full, top_docs_full))
        return val_cache

    def _eval_embedding_only(self, val_cache, n_docs=None) -> float:
        acfg = self.config.answer
        scores = []
        for qid, question, gold, scored_chunks_full, top_docs_full in val_cache:
            if not top_docs_full:
                pred = ""
            else:
                top_docs = top_docs_full if n_docs is None else top_docs_full[:n_docs]
                scored_chunks = self.dense.filter_scored_chunks_by_docs(scored_chunks_full, top_docs)
                kept = self.chunk_selector.apply_threshold(
                    scored_chunks,
                    acfg.emb_threshold,
                    max_total_chunks=acfg.emb_max_total_chunks,
                    min_kept_fallback=acfg.emb_min_kept_fallback,
                    question=question,
                    use_lexical_tiebreak=True,
                )
                top1_passage = self.corpus.get_passage(top_docs[0])
                pred = self.answer_builder.build(
                    question, kept, top1_full_passage=top1_passage, max_chars=acfg.post_process_max_chars, use_dedupe=True
                )
            scores.append(self.meteor.score(pred, gold))
        return float(np.mean(scores))

    def _sweep_n_docs(self, val_cache):
        results = []
        for n_docs in self.config.tuning.n_docs_candidates:
            m = self._eval_embedding_only(val_cache, n_docs=n_docs)
            results.append({"n_docs": n_docs, "meteor": m})
            logger.info(f"TOP_N_DOCS={n_docs:3d} | METEOR={m:.4f}")
        best = max(results, key=lambda r: r["meteor"])
        self.config.retrieval.top_n_docs = best["n_docs"]
        self.log["n_docs_sweep_results"] = results
        logger.info(f">>> BEST TOP_N_DOCS={best['n_docs']} (METEOR={best['meteor']:.4f})")
        return results

    def _narrow_val_cache(self, val_cache):
        n_docs = self.config.retrieval.top_n_docs
        out = []
        for qid, question, gold, scored_chunks_full, top_docs_full in val_cache:
            top_docs = top_docs_full[:n_docs]
            scored_chunks = self.dense.filter_scored_chunks_by_docs(scored_chunks_full, top_docs)
            out.append((qid, question, gold, scored_chunks, top_docs))
        return out

    # -- cross-encoder sweeps --------------------------------------------
    def _compute_ce_cache(self, val_cache_ndocs, reranker: CrossEncoderReranker, k_max: int):
        ce_cache = []
        for qid, question, gold, scored_chunks, top_docs in tqdm(val_cache_ndocs, desc="Cross-encoder scoring"):
            reranked = reranker.rerank(question, scored_chunks, k_max, split_prefix_fn=self.dense.split_chunk_prefix_and_body) if top_docs else []
            ce_cache.append((qid, question, gold, top_docs, reranked))
        return ce_cache

    def _eval_ce_fixed_keep_top_n(self, ce_cache, top_k_candidates, keep_top_n, alpha=1.0) -> float:
        acfg = self.config.answer
        scores = []
        for qid, question, gold, top_docs, reranked_kmax in ce_cache:
            if not reranked_kmax:
                pred = ""
            else:
                candidates_k = reranked_kmax[:top_k_candidates]
                scored_for_keep = [
                    (d, t, p, alpha * ce_s + (1 - alpha) * emb_s) for d, t, p, emb_s, ce_s in candidates_k
                ]
                kept = sorted(scored_for_keep, key=lambda x: -x[3])[:keep_top_n]
                top1_passage = self.corpus.get_passage(kept[0][0]) if kept else ""
                pred = self.answer_builder.build(
                    question, kept, top1_full_passage=top1_passage, max_chars=acfg.post_process_max_chars, use_dedupe=True
                )
            scores.append(self.meteor.score(pred, gold))
        return float(np.mean(scores))

    def _eval_ce_dynamic_margin(self, ce_cache, top_k_candidates, margin, min_keep=1, max_keep=4) -> float:
        acfg = self.config.answer
        scores = []
        for qid, question, gold, top_docs, reranked_kmax in ce_cache:
            if not reranked_kmax:
                pred = ""
            else:
                candidates_k = sorted(reranked_kmax[:top_k_candidates], key=lambda x: -x[4])
                top1_ce = candidates_k[0][4]
                kept_dyn = [c for c in candidates_k if (top1_ce - c[4]) <= margin]
                if len(kept_dyn) < min_keep:
                    kept_dyn = candidates_k[:min_keep]
                elif len(kept_dyn) > max_keep:
                    kept_dyn = candidates_k[:max_keep]
                kept = [(d, t, p, ce_s) for d, t, p, emb_s, ce_s in kept_dyn]
                top1_passage = self.corpus.get_passage(kept[0][0]) if kept else ""
                pred = self.answer_builder.build(
                    question, kept, top1_full_passage=top1_passage, max_chars=acfg.post_process_max_chars, use_dedupe=True
                )
            scores.append(self.meteor.score(pred, gold))
        return float(np.mean(scores))

    def _sweep_cross_encoder(self, val_cache_ndocs, embedding_only_meteor: float):
        tcfg = self.config.tuning
        rcfg = self.config.rerank
        reranker = CrossEncoderReranker(
            self.config.models.resolved_cross_encoder_name(), device=self.device, max_length=self.config.ce_train.max_length
        )
        if not reranker.is_available:
            logger.warning("Cross-encoder unavailable -- skipping CE sweep, USE_CROSS_ENCODER=False.")
            self.config.rerank.use_cross_encoder = False
            return

        k_max = max(tcfg.ce_top_k_candidates)
        ce_cache = self._compute_ce_cache(val_cache_ndocs, reranker, k_max)

        topk_results = []
        for k in tcfg.ce_top_k_candidates:
            m = self._eval_ce_fixed_keep_top_n(ce_cache, top_k_candidates=k, keep_top_n=2)
            topk_results.append({"top_k_candidates": k, "meteor": m})
            logger.info(f"  CE_TOP_K_CANDIDATES={k:3d} | METEOR={m:.4f}")
        best_topk = max(topk_results, key=lambda r: r["meteor"])
        rcfg.top_k_candidates = best_topk["top_k_candidates"]
        self.log["ce_topk_sweep_results"] = topk_results

        keepn_candidates = list(range(1, min(rcfg.top_k_candidates, 8) + 1))
        keepn_results = []
        for n in keepn_candidates:
            m = self._eval_ce_fixed_keep_top_n(ce_cache, top_k_candidates=rcfg.top_k_candidates, keep_top_n=n)
            keepn_results.append({"keep_top_n": n, "meteor": m})
            logger.info(f"  CE_KEEP_TOP_N={n} | METEOR={m:.4f}")
        best_fixed = max(keepn_results, key=lambda r: r["meteor"])
        self.log["ce_keepn_sweep_results"] = keepn_results

        margin_results = []
        for margin in tcfg.ce_margin_candidates:
            for max_keep in tcfg.ce_max_keep_candidates:
                m = self._eval_ce_dynamic_margin(ce_cache, top_k_candidates=rcfg.top_k_candidates, margin=margin, max_keep=max_keep)
                margin_results.append({"margin": margin, "max_keep": max_keep, "meteor": m})
                logger.info(f"  margin={margin:.2f} max_keep={max_keep} | METEOR={m:.4f}")
        best_margin = max(margin_results, key=lambda r: r["meteor"])
        self.log["ce_margin_sweep_results"] = margin_results

        use_dynamic = best_margin["meteor"] > best_fixed["meteor"]
        rcfg.use_dynamic_margin = use_dynamic
        if use_dynamic:
            rcfg.margin = best_margin["margin"]
            rcfg.max_keep = best_margin["max_keep"]
            ce_best = best_margin["meteor"]
        else:
            rcfg.keep_top_n = best_fixed["keep_top_n"]
            ce_best = best_fixed["meteor"]

        alpha_results = []
        keep_n_for_alpha = rcfg.keep_top_n if not use_dynamic else 2
        for alpha in tcfg.ce_alpha_candidates:
            m = self._eval_ce_fixed_keep_top_n(ce_cache, top_k_candidates=rcfg.top_k_candidates, keep_top_n=keep_n_for_alpha, alpha=alpha)
            alpha_results.append({"alpha": alpha, "meteor": m})
            logger.info(f"  alpha={alpha:.2f} | METEOR={m:.4f}")
        best_alpha = max(alpha_results, key=lambda r: r["meteor"])
        self.log["alpha_sweep_results"] = alpha_results

        if not use_dynamic and best_alpha["meteor"] > ce_best:
            rcfg.alpha = best_alpha["alpha"]
            ce_best = best_alpha["meteor"]
        else:
            rcfg.alpha = 1.0

        rcfg.use_cross_encoder = ce_best > embedding_only_meteor
        self._ce_cache = ce_cache
        logger.info(f"Embedding-only={embedding_only_meteor:.4f} | Cross-encoder(best)={ce_best:.4f} -> USE_CROSS_ENCODER={rcfg.use_cross_encoder}")

    # -- final kept-chunk cache + max_chars + dedupe ----------------------
    def _build_final_kept_cache(self, val_cache_ndocs):
        rcfg = self.config.rerank
        acfg = self.config.answer
        result = []
        if rcfg.use_cross_encoder and hasattr(self, "_ce_cache"):
            for qid, question, gold, top_docs, reranked_kmax in self._ce_cache:
                if not reranked_kmax or not top_docs:
                    result.append((qid, question, gold, [], ""))
                    continue
                candidates_k = reranked_kmax[: rcfg.top_k_candidates]
                if rcfg.use_dynamic_margin:
                    cs = sorted(candidates_k, key=lambda x: -x[4])
                    top1_ce = cs[0][4]
                    kept_dyn = [c for c in cs if (top1_ce - c[4]) <= rcfg.margin]
                    if len(kept_dyn) < 1:
                        kept_dyn = cs[:1]
                    elif len(kept_dyn) > rcfg.max_keep:
                        kept_dyn = cs[: rcfg.max_keep]
                    kept = [(d, t, p, ce_s) for d, t, p, emb_s, ce_s in kept_dyn]
                else:
                    scored_for_keep = [
                        (d, t, p, rcfg.alpha * ce_s + (1 - rcfg.alpha) * emb_s) for d, t, p, emb_s, ce_s in candidates_k
                    ]
                    kept = sorted(scored_for_keep, key=lambda x: -x[3])[: rcfg.keep_top_n]
                top1_passage = self.corpus.get_passage(kept[0][0]) if kept else ""
                result.append((qid, question, gold, kept, top1_passage))
        else:
            for qid, question, gold, scored_chunks, top_docs in val_cache_ndocs:
                if not top_docs:
                    result.append((qid, question, gold, [], ""))
                    continue
                kept = self.chunk_selector.apply_threshold(
                    scored_chunks,
                    acfg.emb_threshold,
                    max_total_chunks=acfg.emb_max_total_chunks,
                    min_kept_fallback=acfg.emb_min_kept_fallback,
                    question=question,
                    use_lexical_tiebreak=True,
                )
                top1_passage = self.corpus.get_passage(top_docs[0])
                result.append((qid, question, gold, kept, top1_passage))
        return result

    def _eval_max_chars(self, final_kept_cache, max_chars, use_dedupe=True) -> float:
        scores = []
        for qid, question, gold, kept, top1_passage in final_kept_cache:
            pred = (
                self.answer_builder.build(question, kept, top1_full_passage=top1_passage, max_chars=max_chars, use_dedupe=use_dedupe)
                if kept
                else ""
            )
            scores.append(self.meteor.score(pred, gold))
        return float(np.mean(scores))

    def _sweep_max_chars_and_dedupe(self, final_kept_cache):
        results = []
        for mc in self.config.tuning.max_chars_candidates:
            m = self._eval_max_chars(final_kept_cache, mc)
            results.append({"max_chars": mc, "meteor": m})
            logger.info(f"  max_chars={mc:5d} | METEOR={m:.4f}")
        best = max(results, key=lambda r: r["meteor"])
        self.config.answer.post_process_max_chars = best["max_chars"]
        self.log["max_chars_sweep_results"] = results

        m_on = self._eval_max_chars(final_kept_cache, best["max_chars"], use_dedupe=True)
        m_off = self._eval_max_chars(final_kept_cache, best["max_chars"], use_dedupe=False)
        self.config.answer.use_dedupe = m_on >= m_off
        self.log["dedupe_meteor_on"] = m_on
        self.log["dedupe_meteor_off"] = m_off
        final_meteor = self._eval_max_chars(final_kept_cache, best["max_chars"], use_dedupe=self.config.answer.use_dedupe)
        logger.info(f">>> POST_PROCESS_MAX_CHARS={best['max_chars']} USE_DEDUPE={self.config.answer.use_dedupe} FINAL METEOR={final_meteor:.4f}")
        return final_meteor

    # ---------------------------------------------------------------
    def run(self) -> float:
        """Runs the full staged grid-search, mutating `self.config` in
        place with the best hyperparameters found at each stage. Returns
        the final validation METEOR score.
        """
        seed_everything(self.config.tuning.seed)
        val_split = self._build_val_split()
        raw_cache = self._compute_raw_candidates(val_split)

        logger.info("Stage A: sweeping W_BM25 / W_DENSE ...")
        self._sweep_fusion_weights(raw_cache)

        val_cache = self._build_val_cache(raw_cache)
        embedding_only_baseline = self._eval_embedding_only(val_cache, n_docs=self.config.retrieval.n_docs_sweep_max)
        logger.info(f"Embedding-only baseline: METEOR={embedding_only_baseline:.4f}")

        logger.info("Stage B: sweeping TOP_N_DOCS ...")
        self._sweep_n_docs(val_cache)
        val_cache_ndocs = self._narrow_val_cache(val_cache)
        embedding_only_meteor = self._eval_embedding_only(val_cache_ndocs)
        self.log["embedding_only_meteor"] = embedding_only_meteor

        if self.config.rerank.use_cross_encoder:
            logger.info("Stage C: sweeping cross-encoder rerank strategy ...")
            self._sweep_cross_encoder(val_cache_ndocs, embedding_only_meteor)

        final_kept_cache = self._build_final_kept_cache(val_cache_ndocs)

        logger.info("Stage D: sweeping POST_PROCESS_MAX_CHARS + dedupe ...")
        final_meteor = self._sweep_max_chars_and_dedupe(final_kept_cache)

        self.log["final_meteor_on_val"] = final_meteor
        logger.info("=" * 70)
        logger.info(f"TUNING COMPLETE. Final METEOR on {self.config.tuning.val_size}-question val split: {final_meteor:.4f}")
        logger.info("=" * 70)
        return final_meteor