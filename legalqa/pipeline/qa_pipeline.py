from __future__ import annotations

import time
from typing import List, Optional, Tuple

from ..answer.answer_builder import AnswerBuilder
from ..answer.chunk_selector import ChunkSelector
from ..config import PipelineConfig
from ..corpus import Corpus
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseChunkCache, DenseRetriever
from ..retrieval.fusion import RecencyBooster, ScoreFusion
from ..retrieval.reranker import CrossEncoderReranker
from ..tokenization import VietnameseTokenizer
from ..utils import get_device, logger


class QAPipeline:
    """Full inference-time QA pipeline: BM25 + Dense retrieval -> weighted
    fusion -> (optional) cross-encoder rerank -> answer construction.

    This is the class-based equivalent of `predict_qa_final()` from the
    original notebook. Build it once (loads models + caches), then call
    `.predict(qid, question)` per question.
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
        self.device = get_device(config.models.device)

        self.tokenizer = VietnameseTokenizer(stopwords_file=config.paths.stopwords_file)
        self.dense = DenseRetriever(
            bi_encoder, dense_chunk_cache, max_seq_length=config.models.encode_max_seq_length
        )
        self.recency_booster = RecencyBooster(
            corpus.doc_id_to_name,
            max_boost=config.retrieval.recency_max_boost,
            base_year=config.retrieval.recency_base_year,
        )
        self.fusion = ScoreFusion(self.recency_booster)
        self.chunk_selector = ChunkSelector(self.tokenizer, lex_tiebreak_margin=config.answer.lex_tiebreak_margin)
        self.answer_builder = AnswerBuilder(corpus.doc_id_to_passage, has_chunk_prefix=self.dense.has_prefix)

        self.reranker: Optional[CrossEncoderReranker] = None
        if config.rerank.use_cross_encoder:
            self.reranker = CrossEncoderReranker(
                config.models.resolved_cross_encoder_name(), device=self.device, max_length=config.ce_train.max_length
            )

    # -- retrieval -----------------------------------------------------
    def retrieve_top_docs(self, question: str) -> Tuple[List[str], List[str]]:
        rcfg = self.config.retrieval
        bm25_cands = self.bm25.retrieve(question, top_k=rcfg.bm25_top_k)
        dense_cands = self.dense.retrieve_doc_level(question, top_k=rcfg.dense_top_k)
        top_docs_full = self.fusion.fuse(
            bm25_cands, dense_cands, question, rcfg.w_bm25, rcfg.w_dense, top_n=rcfg.n_docs_sweep_max
        )
        top_docs = top_docs_full[: rcfg.top_n_docs]
        return top_docs, top_docs_full

    # -- chunk keep decision --------------------------------------------
    def _select_kept_chunks(self, question: str, top_docs: List[str], top_docs_full: List[str]):
        scored_chunks_full = self.dense.get_all_scored_chunks(question, top_docs_full)
        scored_chunks = self.dense.filter_scored_chunks_by_docs(scored_chunks_full, top_docs)

        rcfg = self.config.rerank
        if self.reranker is not None and self.reranker.is_available and rcfg.use_cross_encoder:
            reranked = self.reranker.rerank(
                question, scored_chunks, rcfg.top_k_candidates, split_prefix_fn=self.dense.split_chunk_prefix_and_body
            )
            if not reranked:
                return []
            if rcfg.use_dynamic_margin:
                reranked_sorted = sorted(reranked, key=lambda x: -x[4])
                top1_ce = reranked_sorted[0][4]
                kept_dyn = [c for c in reranked_sorted if (top1_ce - c[4]) <= rcfg.margin]
                if len(kept_dyn) < 1:
                    kept_dyn = reranked_sorted[:1]
                elif len(kept_dyn) > rcfg.max_keep:
                    kept_dyn = reranked_sorted[: rcfg.max_keep]
                return [(d, t, p, ce_s) for d, t, p, emb_s, ce_s in kept_dyn]
            scored_for_keep = [
                (d, t, p, rcfg.alpha * ce_s + (1 - rcfg.alpha) * emb_s) for d, t, p, emb_s, ce_s in reranked
            ]
            return sorted(scored_for_keep, key=lambda x: -x[3])[: rcfg.keep_top_n]

        acfg = self.config.answer
        return self.chunk_selector.apply_threshold(
            scored_chunks,
            acfg.emb_threshold,
            max_total_chunks=acfg.emb_max_total_chunks,
            min_kept_fallback=acfg.emb_min_kept_fallback,
            question=question,
            use_lexical_tiebreak=True,
        )

    # -- full predict -----------------------------------------------------
    def predict(self, question: str, log_list: Optional[list] = None, qid: str = "") -> str:
        t_retrieve0 = time.time()
        top_docs, top_docs_full = self.retrieve_top_docs(question)
        t_retrieve = time.time() - t_retrieve0

        if not top_docs:
            if log_list is not None:
                log_list.append((qid, 0, t_retrieve, 0.0, 0))
            return ""

        t_gen0 = time.time()
        kept = self._select_kept_chunks(question, top_docs, top_docs_full)
        top1_full_passage = self.corpus.get_passage(top_docs[0])
        answer = self.answer_builder.build(
            question,
            kept,
            top1_full_passage=top1_full_passage,
            max_chars=self.config.answer.post_process_max_chars,
            use_dedupe=self.config.answer.use_dedupe,
        )
        t_gen = time.time() - t_gen0

        if log_list is not None:
            log_list.append((qid, len(top_docs), t_retrieve, t_gen, len(kept)))
        return answer