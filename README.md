# Email Spam Detection with BERT

A complete Python project for training and serving a Transformer-based email spam classifier.

## Architecture

Email text
  -> cleaning
  -> BERT tokenizer
  -> BERT encoder
  -> classification head
  -> spam probability
  -> SPAM / NOT_SPAM

The API also accepts optional subject/body separately and combines them.

## Features

- BERT (`bert-base-uncased`) fine-tuning
- CSV dataset training
- Train/validation/test split
- Accuracy, precision, recall, F1 and confusion matrix
- Class-weighted loss for imbalanced datasets
- FastAPI REST API
- Batch prediction
- Saved Hugging Face model
- Docker support
- Health endpoint
- Unit tests
- Configurable threshold

## Dataset format

Create `data/spam.csv`:

```csv
text,label
"Congratulations, you won a free prize. Click now!",1
"Can we move our meeting to 10 AM?",0
"Your account has been selected for a reward.",1
"Please find the project report attached.",0
```

Labels:
- `1` = spam
- `0` = not spam / ham

You can also use labels `spam` and `ham`.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

## Training

Put your dataset at:

```text
data/spam.csv
```

Then:

```bash
python -m src.train --data data/spam.csv --output models/spam-bert
```

For a GPU:

```bash
python -m src.train --data data/spam.csv --output models/spam-bert --epochs 3 --batch-size 16
```

Training automatically creates:

```text
models/spam-bert/
```

containing the tokenizer and model.

## Start the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Congratulations! You won a free prize. Click now!\"}"
```

Example response:

```json
{
  "label": "SPAM",
  "spam_probability": 0.98,
  "not_spam_probability": 0.02,
  "threshold": 0.5
}
```

## Subject + body

```json
{
  "subject": "You won a prize",
  "body": "Click this link to claim your free reward."
}
```

## Batch prediction

POST to `/predict/batch`:

```json
{
  "emails": [
    {"text": "You won a free prize!"},
    {"text": "The meeting is tomorrow at 10."}
  ]
}
```

## Important production notes

This project classifies email content. It does not automatically open URLs, execute attachments, or contact external domains. Keep those operations isolated if you later add email-security enrichment.

For production, add authentication, rate limiting, logging, monitoring, model versioning, and a human-review path for uncertain predictions.

## Recommended next improvements

1. Add a large, legally usable spam dataset.
2. Compare BERT against TF-IDF + Logistic Regression and Linear SVM.
3. Add character n-grams for obfuscated spam.
4. Add email metadata features such as URL count and HTML ratio.
5. Add calibration and an "UNCERTAIN" class.
6. Add a feedback endpoint so reviewed predictions can become new training data.
