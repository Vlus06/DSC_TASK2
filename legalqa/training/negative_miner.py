from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from tqdm.auto import tqdm

from ..config import MiningConfig
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseChunkCache, DenseRetriever
from ..tokenization import VietnameseTokenizer
from ..training.positive_miner import MinedPositive
from ..utils import free_memory, logger


@dataclass
class TrainingExample:
    question: str
    positive: str
    negatives: List[str]

    def to_dict(self) -> dict:
        return {"question": self.question, "positive": self.positive, "negatives": self.negatives}


class HardNegativeMiner:
    """Combines three complementary hard-negative sources for each mined
    positive, matching the "3-source" strategy from the fine-tuning
    notebook:

      1. BM25  -- lexically-similar chunks from OTHER documents (catches
         term-overlap confusions).
      2. Dense (base, pre-finetune model) -- semantically-similar chunks
         from OTHER documents (catches meaning-level confusions that BM25
         misses -- a different failure mode than #1).
      3. Intra-doc -- other Điều/Khoản from the SAME correct document
         (catches "wrong clause, right law" confusions -- old/new
         amendment mix-ups, adjacent Khoản confusions).
    """

    def __init__(
        self,
        chunk_cache: DenseChunkCache,
        bm25: BM25Retriever,
        dense_base: Optional[DenseRetriever],
        tokenizer: VietnameseTokenizer,
        config: MiningConfig,
    ):
        self.chunk_cache = chunk_cache
        self.bm25 = bm25
        self.dense_base = dense_base
        self.tokenizer = tokenizer
        self.config = config
        self._q_tokens_cache: Dict[str, Set[str]] = {}

    def _q_tokens(self, question: str) -> Set[str]:
        tokens = self._q_tokens_cache.get(question)
        if tokens is None:
            tokens = set(self.tokenizer.tokenize_clean(question))
            self._q_tokens_cache[question] = tokens
        return tokens

    def _best_chunk_in_doc(self, doc_id: str, q_tokens: Set[str], limit: int = 20) -> Optional[int]:
        positions = self.chunk_cache.doc_to_chunk_positions.get(doc_id, [])
        if not positions:
            return None
        return max(
            positions[:limit],
            key=lambda pos: self.tokenizer.token_overlap_score(q_tokens, self.chunk_cache.chunk_texts_all[pos]),
            default=None,
        )

    def _truncate(self, text: str) -> str:
        return text[: self.config.max_train_chars]

    def mine_for_example(self, ex: MinedPositive) -> List[str]:
        cfg = self.config
        question = ex.question
        positive_doc_id = ex.doc_id
        positive_pos = ex.positive_chunk_pos

        q_tokens = self._q_tokens(question)
        neg_texts: List[str] = []
        seen_positions = {positive_pos}

        # SOURCE 1: BM25 -- other documents
        bm25_docs = self.bm25.retrieve_doc_ids_only(question, top_k=cfg.bm25_top_k_for_mining)
        bm25_neg_docs = [d for d in bm25_docs if d != positive_doc_id][: cfg.n_bm25_neg_docs]
        for nd in bm25_neg_docs:
            best_pos = self._best_chunk_in_doc(nd, q_tokens)
            if best_pos is not None and best_pos not in seen_positions:
                neg_texts.append(self._truncate(self.chunk_cache.chunk_texts_all[best_pos]))
                seen_positions.add(best_pos)
            if len(neg_texts) >= cfg.n_hard_neg_per_query:
                return neg_texts

        # SOURCE 2: Dense (base/pre-finetune model) -- other documents,
        # catches meaning-level confusions BM25 misses.
        if self.dense_base is not None:
            dense_docs = self.dense_base.retrieve_doc_ids_only(question, top_k=cfg.dense_top_k_for_mining)
            dense_neg_docs = [d for d in dense_docs if d != positive_doc_id][: cfg.n_dense_neg_docs]
            for nd in dense_neg_docs:
                best_pos = self._best_chunk_in_doc(nd, q_tokens)
                if best_pos is not None and best_pos not in seen_positions:
                    neg_texts.append(self._truncate(self.chunk_cache.chunk_texts_all[best_pos]))
                    seen_positions.add(best_pos)
                if len(neg_texts) >= cfg.n_hard_neg_per_query:
                    return neg_texts

        # SOURCE 3: Intra-doc -- same correct document, different Điều/Khoản.
        intra_positions = [p for p in self.chunk_cache.doc_to_chunk_positions.get(positive_doc_id, []) if p != positive_pos]
        if intra_positions:
            rng = random.Random(hash(question) & 0xFFFFFFFF)
            rng.shuffle(intra_positions)
            for pos in intra_positions[: cfg.n_intra_doc_candidates]:
                if len(neg_texts) >= cfg.n_hard_neg_per_query:
                    break
                if pos not in seen_positions:
                    neg_texts.append(self._truncate(self.chunk_cache.chunk_texts_all[pos]))
                    seen_positions.add(pos)

        return neg_texts

    def mine_all(self, mined_positives: List[MinedPositive], show_progress: bool = True) -> List[TrainingExample]:
        results: List[TrainingExample] = []
        t0 = time.time()
        iterator = tqdm(mined_positives, desc="Mining hard negatives (3 sources)") if show_progress else mined_positives
        for i, ex in enumerate(iterator):
            negs = self.mine_for_example(ex)
            if not negs:
                continue
            results.append(
                TrainingExample(
                    question=ex.question,
                    positive=self._truncate(self.chunk_cache.chunk_texts_all[ex.positive_chunk_pos]),
                    negatives=negs,
                )
            )
            if (i + 1) % 500 == 0:
                free_memory()

        logger.info(f"Hard-negative mining done in {time.time() - t0:.1f}s.")
        logger.info(f"Complete training examples (positive + >=1 hard negative): {len(results)}")
        return results