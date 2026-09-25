import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    model_dir: str = os.getenv("MODEL_DIR", "models/spam-bert")
    max_length: int = int(os.getenv("MAX_LENGTH", "256"))
    threshold: float = float(os.getenv("SPAM_THRESHOLD", "0.5"))

settings = Settings()
