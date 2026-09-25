import pandas as pd
from src.data import normalize_label, split_dataset


def test_normalize_label():
    assert normalize_label("spam") == 1
    assert normalize_label("ham") == 0
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0


def test_split_dataset():
    df = pd.DataFrame({
        "text": [f"email {i}" for i in range(40)],
        "label": [0, 1] * 20,
    })
    train, val, test = split_dataset(df)
    assert len(train) + len(val) + len(test) == 40
