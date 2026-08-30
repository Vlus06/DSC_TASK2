from __future__ import annotations

import json
import os
import random
from typing import List, Optional

from ..config import PipelineConfig
from ..corpus import Corpus, load_train_json
from ..legal_metadata import TitleIndex
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseChunkCache, DenseRetriever
from ..tokenization import VietnameseTokenizer
from ..training.bi_encoder_trainer import BiEncoderTrainer
from ..training.cross_encoder_trainer import CrossEncoderTrainer
from ..training.negative_miner import HardNegativeMiner, TrainingExample
from ..training.positive_miner import PositiveMiner
from ..utils import free_memory, get_device, logger, seed_everything


class RetrieverFinetuner:
    """End-to-end Stage 1 orchestrator: mine (positive, hard-negatives)
    training pairs from `train.json`, then fine-tune both the bi-encoder
    (retriever) and the cross-encoder (reranker) on them.

    Requires a pre-built `DenseChunkCache` (see
    `legalqa.retrieval.dense.DenseChunkCacheBuilder`, run once with the
    *base* bi-encoder) so that mining can look up candidate chunks without
    re-embedding the whole corpus.
    """

    def __init__(self, config: PipelineConfig, chunk_cache: DenseChunkCache):
        self.config = config
        self.chunk_cache = chunk_cache
        self.device = get_device(config.models.device)
        seed_everything(config.mining.seed)

        self.tokenizer = VietnameseTokenizer(stopwords_file=config.paths.stopwords_file)
        self.corpus: Optional[Corpus] = None
        self.title_index: Optional[TitleIndex] = None
        self.bm25: Optional[BM25Retriever] = None

    def load_corpus_and_indexes(self) -> None:
        self.corpus = Corpus.load(self.config.paths.context_dir)
        self.title_index = TitleIndex().build(self.corpus.doc_id_to_passage)

        if self.config.paths.bm25_cache_file and os.path.exists(self.config.paths.bm25_cache_file):
            self.bm25 = BM25Retriever.from_cache(
                self.config.paths.bm25_cache_file, self.tokenizer, self.corpus.doc_id_to_passage
            )
        else:
            self.bm25 = BM25Retriever(self.tokenizer, self.corpus.doc_id_to_passage).build()

    def mine_training_pairs(self) -> List[TrainingExample]:
        assert self.corpus is not None and self.title_index is not None and self.bm25 is not None
        train_raw = load_train_json(self.config.paths.train_path)
        logger.info(f"Total training questions: {len(train_raw)}")

        positive_miner = PositiveMiner(
            self.chunk_cache,
            self.tokenizer,
            self.title_index,
            overlap_fallback_threshold=self.config.mining.overlap_fallback_threshold,
        )
        mined_positives = positive_miner.mine_all(train_raw)

        # Base (pre-finetune) bi-encoder used ONLY for Dense hard-negative
        # mining -- it must NOT be the model we're about to fine-tune.
        dense_base = None
        try:
            from sentence_transformers import SentenceTransformer

            base_model = SentenceTransformer(
                self.config.models.base_bi_encoder_name, device=self.device, trust_remote_code=True
            )
            base_model.max_seq_length = 256
            dense_base = DenseRetriever(base_model, self.chunk_cache, max_seq_length=256)
        except Exception as e:
            logger.warning(f"Could not load base bi-encoder for dense mining ({e}) -- skipping dense negative source.")

        neg_miner = HardNegativeMiner(self.chunk_cache, self.bm25, dense_base, self.tokenizer, self.config.mining)
        training_examples = neg_miner.mine_all(mined_positives)

        if dense_base is not None:
            del dense_base
            free_memory()

        self.config.paths.ensure_dirs()
        with open(self.config.paths.mined_pairs_path, "w", encoding="utf-8") as f:
            json.dump([ex.to_dict() for ex in training_examples], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved mined training pairs to: {self.config.paths.mined_pairs_path}")

        return training_examples

    @staticmethod
    def load_mined_training_pairs(path: str) -> List[TrainingExample]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [TrainingExample(question=r["question"], positive=r["positive"], negatives=r["negatives"]) for r in raw]

    def split_holdout(self, training_examples: List[TrainingExample]):
        rng = random.Random(self.config.mining.seed)
        shuffled = list(training_examples)
        rng.shuffle(shuffled)
        n_holdout = min(
            self.config.bi_train.eval_holdout_max,
            max(1, int(len(shuffled) * self.config.bi_train.eval_holdout_fraction)),
        )
        eval_examples = shuffled[:n_holdout]
        train_examples = shuffled[n_holdout:]
        logger.info(f"Train/holdout split: {len(train_examples)} train, {len(eval_examples)} holdout.")
        return train_examples, eval_examples

    def finetune_bi_encoder(self, train_examples: List[TrainingExample], eval_examples: List[TrainingExample]):
        trainer = BiEncoderTrainer(
            self.config.models.base_bi_encoder_name,
            self.device,
            self.config.bi_train,
            train_max_seq_length=self.config.models.train_max_seq_length,
        )
        model = trainer.train(train_examples, output_path=self.config.paths.finetuned_bi_encoder_dir)

        logger.info("Quick pairwise sanity-check AFTER fine-tuning:")
        trainer.quick_pairwise_eval(model, eval_examples, tag="AFTER finetune")

        logger.info("Quick pairwise sanity-check for the BASE model (for comparison):")
        from sentence_transformers import SentenceTransformer

        base_model = SentenceTransformer(
            self.config.models.base_bi_encoder_name, device=self.device, trust_remote_code=True
        )
        trainer.quick_pairwise_eval(base_model, eval_examples, tag="BEFORE finetune (base model)")
        del base_model
        free_memory()
        return model

    def finetune_cross_encoder(self, train_examples: List[TrainingExample], eval_examples: List[TrainingExample]):
        trainer = CrossEncoderTrainer(self.config.models.base_cross_encoder_name, self.device, self.config.ce_train)
        return trainer.train(train_examples, eval_examples, output_path=self.config.paths.finetuned_cross_encoder_dir)

    def run(self):
        """Full Stage 1: mine -> split -> fine-tune bi-encoder -> fine-tune cross-encoder."""
        self.load_corpus_and_indexes()
        training_examples = self.mine_training_pairs()
        train_examples, eval_examples = self.split_holdout(training_examples)

        bi_encoder = self.finetune_bi_encoder(train_examples, eval_examples)
        free_memory()
        cross_encoder = self.finetune_cross_encoder(train_examples, eval_examples)

        logger.info("=" * 70)
        logger.info("STAGE 1 COMPLETE.")
        logger.info(f"  Bi-encoder (retriever) fine-tuned : {self.config.paths.finetuned_bi_encoder_dir}")
        logger.info(f"  Cross-encoder (reranker) fine-tuned: {self.config.paths.finetuned_cross_encoder_dir}")
        logger.info(f"  Training pairs (for audit)          : {self.config.paths.mined_pairs_path}")
        logger.info("=" * 70)
        return bi_encoder, cross_encoder