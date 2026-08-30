from __future__ import annotations

import glob
import os
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
        except Exception as e1:
            logger.info(f"Falling back to CEBinaryClassificationEvaluator ({e1})...")
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

    @staticmethod
    def _verify_saved_model(output_path: str) -> None:
        """Guards against the exact bug hit previously: `.fit(...,
        output_path=...)`'s internal `save_best_model` can silently leave
        only an `eval/` folder with NO model weights if training doesn't
        trigger a "best" checkpoint save. Always call this right after an
        explicit `.save()` and raise loudly if it's still missing."""
        model_files = (
            glob.glob(os.path.join(output_path, "*.safetensors"))
            + glob.glob(os.path.join(output_path, "pytorch_model.bin"))
            + glob.glob(os.path.join(output_path, "model.safetensors"))
        )
        config_found = os.path.exists(os.path.join(output_path, "config.json"))
        if not model_files or not config_found:
            existing = os.listdir(output_path) if os.path.isdir(output_path) else []
            raise RuntimeError(
                f"CRITICAL: no model weights or config.json found in {output_path} after save(). "
                f"Files present: {existing}"
            )
        logger.info(f"Verified saved cross-encoder weights ({model_files}) and config.json in {output_path}.")

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
        try:
            self._fit_with_oom_retry(
                model,
                dataloader,
                evaluator=evaluator,
                epochs=cfg.epochs,
                warmup_steps=warmup_steps,
                output_path=output_path,  # save_best_model runs automatically -- may not actually save (see below)
                show_progress_bar=True,
                optimizer_params={"lr": cfg.lr},
                use_amp=(self.device == "cuda"),
            )
        finally:
            # CRITICAL FIX: regardless of whether save_best_model actually
            # saved anything, ALWAYS explicitly save the model's final
            # state to output_path. If training crashed partway through
            # (e.g. OOM), the model object in memory is still valid, so
            # this still succeeds and we don't lose the run.
            os.makedirs(output_path, exist_ok=True)
            model.save(output_path)
            logger.info(f"Explicitly saved cross-encoder to: {output_path}")

        self._verify_saved_model(output_path)

        logger.info("Quick pairwise sanity-check AFTER fine-tuning:")
        self.quick_pairwise_eval(model, eval_examples, tag="AFTER finetune")

        self.model = model
        return model

    @staticmethod
    def quick_pairwise_eval(model, eval_examples: List[TrainingExample], tag: str = "") -> float:
        """Cheap sanity metric: fraction of (positive, negative) pairs
        where the model scores the positive higher. NOT the official
        METEOR metric -- only use this to sanity-check training didn't
        diverge."""
        correct, total = 0, 0
        for ex in eval_examples:
            pos_score = float(model.predict([(ex.question, ex.positive)])[0])
            for neg in ex.negatives:
                neg_score = float(model.predict([(ex.question, neg)])[0])
                total += 1
                if pos_score > neg_score:
                    correct += 1
        acc = correct / max(1, total)
        logger.info(f"[{tag}] pairwise accuracy: {acc:.4f} ({correct}/{total})")
        return acc

    @classmethod
    def train_from_existing_pairs(
        cls,
        model_name: str,
        device: str,
        config: CrossEncoderTrainConfig,
        mined_pairs_path: str,
        output_path: str,
        seed: int = 42,
        eval_holdout_max: int = 200,
        eval_holdout_fraction: float = 0.05,
    ):
        """Re-finetunes the cross-encoder from an existing
        `mined_training_pairs.json` WITHOUT re-running mining -- mirrors
        Part 1 of the `t-ng-2(2).ipynb` "refinetune CE + rebuild dense
        cache" notebook. Uses the SAME shuffle/split (seed=42) as the
        original bi-encoder training run so the holdout set matches.
        """
        import json
        import random

        with open(mined_pairs_path, encoding="utf-8") as f:
            raw = json.load(f)
        logger.info(f"Loaded {len(raw)} existing training examples from {mined_pairs_path}.")

        rng = random.Random(seed)
        shuffled = list(raw)
        rng.shuffle(shuffled)
        n_holdout = min(eval_holdout_max, max(1, int(len(shuffled) * eval_holdout_fraction)))
        eval_raw, train_raw = shuffled[:n_holdout], shuffled[n_holdout:]
        logger.info(f"Split: {len(train_raw)} train / {len(eval_raw)} eval (holdout).")

        train_examples = [TrainingExample(question=r["question"], positive=r["positive"], negatives=r["negatives"]) for r in train_raw]
        eval_examples = [TrainingExample(question=r["question"], positive=r["positive"], negatives=r["negatives"]) for r in eval_raw]

        trainer = cls(model_name, device, config)
        return trainer.train(train_examples, eval_examples, output_path=output_path)
