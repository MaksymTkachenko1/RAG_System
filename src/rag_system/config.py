from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Centralizes frequently used filesystem locations."""

    root: Path = Path(__file__).resolve().parents[2]

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def media(self) -> Path:
        return self.root / "data" / "media"

    @property
    def vector_store(self) -> Path:
        return self.data_processed / "vector_store.pkl"

    @property
    def metadata(self) -> Path:
        return self.data_processed / "articles.json"


@dataclass(frozen=True)
class ModelConfig:
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "sentence-transformers/clip-ViT-B-32"
    summarizer_model: str = "facebook/bart-large-cnn"


paths = Paths()
models = ModelConfig()
