from __future__ import annotations

from typing import List

from ..config import CrossEncoderTrainConfig
from ..training.negative_miner import TrainingExample
from ..utils import free_memory, logger


class CrossEncoderTrainer:
    """Fine-tunes a `sentence_transformers.CrossEncoder` (binary
    relevance) on (question, positive)=1.0 / (question, negative)=0.0
    pairs built from the same mined training examples used for the
    bi-encoder.
    """

    def __init__(self, model_name: str, device: str, config: CrossEncoderTrainConfig):
        self.model_name = model_name
        self.device = device
        self.config = config
        self.model = None

    @staticmethod
    def _build_pairs(examples: List[TrainingExample]):
        from sentence_transformers import InputExample

        pairs = []
        for ex in examples:
            pairs.append(InputExample(texts=[ex.question, ex.positive], label=1.0))
            for neg in ex.negatives:
                pairs.append(InputExample(texts=[ex.question, neg], label=0.0))
        return pairs

    @staticmethod
    def _build_eval_lists(examples: List[TrainingExample]):
        sent1, sent2, labels = [], [], []
        for ex in examples:
            sent1.append(ex.question)
            sent2.append(ex.positive)
            labels.append(1)
            for neg in ex.negatives:
                sent1.append(ex.question)
                sent2.append(neg)
                labels.append(0)
        return sent1, sent2, labels

    def _make_evaluator(self, eval_examples: List[TrainingExample]):
        sent1, sent2, labels = self._build_eval_lists(eval_examples)
        try:
            # Newer sentence-transformers API.
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator

            pairs = list(zip(sent1, sent2))
            return CrossEncoderClassificationEvaluator(pairs, labels, name="holdout")
        except Exception:
            from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

            return CEBinaryClassificationEvaluator(sent1, sent2, labels, name="holdout")

    def _fit_with_oom_retry(self, model, dataloader, **fit_kwargs):
        import torch
        from torch.utils.data import DataLoader

        current_bs = dataloader.batch_size
        dataset = dataloader.dataset
        while current_bs >= 1:
            try:
                model.fit(train_dataloader=dataloader, **fit_kwargs)
                return
            except torch.cuda.OutOfMemoryError:
                free_memory()
                current_bs = max(1, current_bs // 2)
                logger.warning(f"[OOM RETRY] Cross-encoder reducing batch_size to {current_bs}...")
                dataloader = DataLoader(dataset, shuffle=True, batch_size=current_bs)
        raise RuntimeError("Cross-encoder could not fit even at batch_size=1.")

    def train(
        self,
        train_examples: List[TrainingExample],
        eval_examples: List[TrainingExample],
        output_path: str,
    ):
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch.utils.data import DataLoader

        cfg = self.config
        model = CrossEncoder(
            self.model_name, num_labels=1, device=self.device, trust_remote_code=True, max_length=cfg.max_length
        )

        train_pairs = self._build_pairs(train_examples)
        logger.info(f"Total (question, chunk, label) training pairs: {len(train_pairs)}")

        dataloader = DataLoader(train_pairs, shuffle=True, batch_size=cfg.batch_size)
        n_steps = len(dataloader) * cfg.epochs
        warmup_steps = int(n_steps * cfg.warmup_ratio)

        evaluator = self._make_evaluator(eval_examples) if eval_examples else None

        logger.info(f"Fine-tuning cross-encoder: {n_steps} steps, warmup={warmup_steps}, batch_size={cfg.batch_size}.")
        self._fit_with_oom_retry(
            model,
            dataloader,
            evaluator=evaluator,
            epochs=cfg.epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            optimizer_params={"lr": cfg.lr},
            use_amp=(self.device == "cuda"),
        )
        logger.info(f"Saved fine-tuned cross-encoder to: {output_path}")
        self.model = model
        return model