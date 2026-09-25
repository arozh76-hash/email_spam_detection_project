from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import settings


class SpamClassifier:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or settings.model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Email text cannot be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        not_spam = float(probabilities[0].item())
        spam = float(probabilities[1].item())
        label = "SPAM" if spam >= settings.threshold else "NOT_SPAM"

        return {
            "label": label,
            "spam_probability": round(spam, 6),
            "not_spam_probability": round(not_spam, 6),
            "threshold": settings.threshold,
        }
