from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import SpamClassifier

app = FastAPI(
    title="BERT Email Spam Detection API",
    version="1.0.0",
    description="Transformer-based email spam classification service.",
)


class EmailRequest(BaseModel):
    text: str | None = Field(default=None, max_length=100000)
    subject: str | None = Field(default=None, max_length=10000)
    body: str | None = Field(default=None, max_length=100000)


class BatchRequest(BaseModel):
    emails: list[EmailRequest] = Field(min_length=1, max_length=100)


def combine_email(request: EmailRequest) -> str:
    if request.text and request.text.strip():
        return request.text.strip()

    subject = (request.subject or "").strip()
    body = (request.body or "").strip()

    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(f"Body: {body}")

    return "\n".join(parts).strip()


@lru_cache(maxsize=1)
def get_classifier():
    model_dir = os.getenv("MODEL_DIR", "models/spam-bert")
    if not os.path.exists(model_dir):
        raise RuntimeError(
            f"Model not found at '{model_dir}'. Train the model first."
        )
    return SpamClassifier(model_dir)


@app.get("/health")
def health():
    model_dir = os.getenv("MODEL_DIR", "models/spam-bert")
    return {
        "status": "ok",
        "model_exists": os.path.exists(model_dir),
        "model_dir": model_dir,
    }


@app.post("/predict")
def predict(request: EmailRequest):
    text = combine_email(request)
    if not text:
        raise HTTPException(status_code=400, detail="Provide text or subject/body.")

    try:
        return get_classifier().predict(text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    classifier = get_classifier()

    for email in request.emails:
        text = combine_email(email)
        if not text:
            results.append({"error": "Empty email"})
            continue
        results.append(classifier.predict(text))

    return {"results": results}
