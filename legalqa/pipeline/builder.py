from __future__ import annotations

import os
from typing import Tuple

from ..config import PipelineConfig
from ..corpus import Corpus
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseChunkCache
from ..tokenization import VietnameseTokenizer
from ..utils import get_device, logger


def load_bi_encoder(config: PipelineConfig):
    from sentence_transformers import SentenceTransformer

    device = get_device(config.models.device)
    name = config.models.resolved_bi_encoder_name()
    model = SentenceTransformer(name, device=device, trust_remote_code=True)
    model.max_seq_length = config.models.encode_max_seq_length
    logger.info(f"Loaded bi-encoder for inference: {name} (max_seq_length={model.max_seq_length})")
    return model


def build_corpus_and_bm25(config: PipelineConfig) -> Tuple[Corpus, BM25Retriever]:
    corpus = Corpus.load(config.paths.context_dir)
    tokenizer = VietnameseTokenizer(stopwords_file=config.paths.stopwords_file)

    if config.paths.bm25_cache_file and os.path.exists(config.paths.bm25_cache_file):
        bm25 = BM25Retriever.from_cache(config.paths.bm25_cache_file, tokenizer, corpus.doc_id_to_passage)
    else:
        logger.info("No BM25 cache found -- building BM25 index from scratch.")
        bm25 = BM25Retriever(tokenizer, corpus.doc_id_to_passage).build()
        if config.paths.bm25_cache_file:
            bm25.save(config.paths.bm25_cache_file)
    return corpus, bm25


def load_dense_chunk_cache(config: PipelineConfig) -> DenseChunkCache:
    path = config.paths.dense_cache_file
    assert path is not None, "config.paths.dense_cache_file must be set (build it in Stage 2 first)."
    return DenseChunkCache.load(path)