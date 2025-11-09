from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import models


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class RetrievalResult:
    id: str
    title: str
    url: str
    summary: str
    image_path: Optional[str]
    score: float
    full_text: str


class Retriever:
    def __init__(self, df: pd.DataFrame, text_weight: float = 0.75) -> None:
        self.df = df
        self.text_encoder = SentenceTransformer(models.text_model)
        self.clip_encoder = SentenceTransformer(models.clip_model)
        self.text_weight = text_weight

    def search(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        query_vec = self.text_encoder.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        # Use zeros for image similarity because UI accepts text queries for now.
        image_vec = np.zeros(self.clip_encoder.get_sentence_embedding_dimension(), dtype=np.float32)

        scored: List[RetrievalResult] = []
        for _, row in self.df.iterrows():
            text_score = cosine_similarity(query_vec, row["text_embedding"])
            image_score = cosine_similarity(image_vec, row["image_embedding"])
            score = self.text_weight * text_score + (1 - self.text_weight) * image_score
            scored.append(
                RetrievalResult(
                    id=row["id"],
                    title=row["title"],
                    url=row["url"],
                    summary=row["summary"],
                    image_path=row["image_path"],
                    score=score,
                    full_text=row["full_text"],
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
