"""Text-pair reranker compatible with the original CrossEncoder scoring."""


class TransformerReranker:
    """Load a one-logit sequence classifier and return sigmoid scores."""

    def __init__(self, model_name, device, max_length):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        # This repository ships a SentencePiece vocabulary but no tokenizer.json.
        # Force the slow tokenizer so Transformers does not fall back to an
        # incompatible tiktoken conversion path.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True,
        ).to(device)
        self.model.eval()
        if self.model.config.num_labels != 1:
            raise ValueError(
                f"Expected a one-logit reranker, got num_labels={self.model.config.num_labels}"
            )

    def predict(self, pairs, show_progress_bar=False, batch_size=32):
        del show_progress_bar
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = [list(pair) for pair in pairs[start:start + batch_size]]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                    verbose=False,
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1).float()
                scores.extend(self.torch.sigmoid(logits).cpu().numpy().tolist())
        return scores
