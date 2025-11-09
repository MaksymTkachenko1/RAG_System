from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import models, paths
from .data_ingestion import Article, run_ingestion


def _ensure_articles() -> List[Article]:
    if paths.metadata.exists():
        data = json.loads(paths.metadata.read_text(encoding="utf-8"))
        return [Article(**item) for item in data]
    return run_ingestion()


def _load_image(path: Optional[str]) -> Optional[Image.Image]:
    if not path:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


@dataclass
class VectorStore:
    text_encoder: SentenceTransformer
    clip_encoder: SentenceTransformer

    @classmethod
    def build(cls, limit: int = 15) -> pd.DataFrame:
        articles = _ensure_articles()
        if not articles:
            raise RuntimeError("No articles available for indexing.")

        text_encoder = SentenceTransformer(models.text_model)
        clip_encoder = SentenceTransformer(models.clip_model)
        records = []

        texts = [article.full_text for article in articles]
        text_embeddings = text_encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        image_embeddings = []
        for article in tqdm(articles, desc="Encoding images"):
            image = _load_image(article.image_path)
            if image:
                vec = clip_encoder.encode(
                    [image], convert_to_numpy=True, normalize_embeddings=True
                )[0]
            else:
                vec = np.zeros(
                    clip_encoder.get_sentence_embedding_dimension(), dtype=np.float32
                )
            image_embeddings.append(vec)

        for article, text_vec, image_vec in zip(articles, text_embeddings, image_embeddings):
            records.append(
                {
                    **article.to_dict(),
                    "text_embedding": text_vec,
                    "image_embedding": image_vec,
                }
            )

        df = pd.DataFrame(records)
        paths.vector_store.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(paths.vector_store)
        return df

    @staticmethod
    def load() -> pd.DataFrame:
        if not paths.vector_store.exists():
            return VectorStore.build()
        return pd.read_pickle(paths.vector_store)


if __name__ == "__main__":
    VectorStore.build()
