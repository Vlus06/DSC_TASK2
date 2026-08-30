from __future__ import annotations

from typing import List

from ..config import BiEncoderTrainConfig
from ..training.negative_miner import TrainingExample
from ..utils import free_memory, logger


class BiEncoderTrainer:
    """Fine-tunes a bi-encoder (SentenceTransformer) with
    `CachedMultipleNegativesRankingLoss` (GradCache), which lets us use a
    large *effective* batch size (many negatives per batch, good for
    contrastive learning) while keeping the *peak* activation memory tied
    to a small `mini_batch_size` -- this is what avoids the OOM the
    original notebook hit with plain `MultipleNegativesRankingLoss` +
    `DataParallel`.

    One `InputExample` per question: `texts=[question, positive, neg1,
    neg2, ...]` -- avoids re-encoding (question, positive) once per
    negative.
    """

    def __init__(self, model_name: str, device: str, config: BiEncoderTrainConfig, train_max_seq_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.config = config
        self.train_max_seq_length = train_max_seq_length
        self.model = None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
        model.max_seq_length = self.train_max_seq_length
        if self.config.use_gradient_checkpointing:
            try:
                model[0].auto_model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing on bi-encoder backbone.")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing ({e}) -- continuing without it.")
        return model

    @staticmethod
    def _build_input_examples(examples: List[TrainingExample]):
        from sentence_transformers import InputExample

        return [InputExample(texts=[ex.question, ex.positive] + ex.negatives) for ex in examples]

    def _fit_with_oom_retry(self, model, train_dataloader, train_loss, **fit_kwargs):
        import torch
        from torch.utils.data import DataLoader

        current_bs = train_dataloader.batch_size
        dataset = train_dataloader.dataset
        while current_bs >= 1:
            try:
                model.fit(train_objectives=[(train_dataloader, train_loss)], **fit_kwargs)
                return
            except torch.cuda.OutOfMemoryError:
                free_memory()
                current_bs = max(1, current_bs // 2)
                logger.warning(f"[OOM RETRY] Reducing batch_size to {current_bs} and retrying...")
                train_dataloader = DataLoader(dataset, shuffle=True, batch_size=current_bs)
        raise RuntimeError(
            "Could not fit even at batch_size=1 -- reduce max_train_chars/train_max_seq_length further."
        )

    def train(
        self,
        train_examples: List[TrainingExample],
        output_path: str,
    ):
        from sentence_transformers import losses
        from torch.utils.data import DataLoader

        model = self._load_model()
        bi_train_examples = self._build_input_examples(train_examples)
        logger.info(f"Total InputExamples (1 row/example, negatives folded in): {len(bi_train_examples)}")

        cfg = self.config
        train_dataloader = DataLoader(bi_train_examples, shuffle=True, batch_size=cfg.batch_size)
        train_loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=cfg.mini_batch_size)

        n_steps = len(train_dataloader) * cfg.epochs
        warmup_steps = int(n_steps * cfg.warmup_ratio)
        logger.info(
            f"Fine-tuning bi-encoder: {n_steps} steps, warmup={warmup_steps}, "
            f"batch_size={cfg.batch_size} (mini_batch_size={cfg.mini_batch_size}, "
            f"max_seq_length={self.train_max_seq_length})."
        )

        self._fit_with_oom_retry(
            model,
            train_dataloader,
            train_loss,
            epochs=cfg.epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            optimizer_params={"lr": cfg.lr},
            use_amp=(self.device == "cuda"),
        )
        logger.info(f"Saved fine-tuned bi-encoder to: {output_path}")
        self.model = model
        return model

    @staticmethod
    def quick_pairwise_eval(model, eval_examples: List[TrainingExample], tag: str = "") -> float:
        """Cheap sanity metric: fraction of (positive, negative) pairs where
        cos_sim(q, positive) > cos_sim(q, negative). NOT the official METEOR
        metric -- only use this to sanity-check training didn't diverge.
        """
        correct, total = 0, 0
        for ex in eval_examples:
            q_vec = model.encode([ex.question], normalize_embeddings=True)[0]
            pos_vec = model.encode([ex.positive], normalize_embeddings=True)[0]
            pos_sim = float(q_vec @ pos_vec)
            for neg in ex.negatives:
                neg_vec = model.encode([neg], normalize_embeddings=True)[0]
                neg_sim = float(q_vec @ neg_vec)
                total += 1
                if pos_sim > neg_sim:
                    correct += 1
        acc = correct / max(1, total)
        logger.info(f"[{tag}] pairwise accuracy (pos_sim > neg_sim): {acc:.4f} ({correct}/{total})")
        return acc