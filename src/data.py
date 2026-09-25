from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_label(value) -> int:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"spam", "1", "true", "yes"}:
            return 1
        if v in {"ham", "not_spam", "not spam", "0", "false", "no"}:
            return 0
    if int(value) in (0, 1):
        return int(value)
    raise ValueError(f"Unsupported label: {value!r}")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].map(normalize_label)
    df = df[df["text"] != ""].drop_duplicates("text")

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both spam and non-spam examples.")
    if len(df) < 20:
        raise ValueError("Use at least 20 examples for a real training run.")

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, seed: int = 42):
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )
    return train_df, val_df, test_df
