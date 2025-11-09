from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .generator import Answer, AnswerGenerator
from .retriever import Retriever, RetrievalResult
from .vector_store import VectorStore


@dataclass
class RagPipeline:
    store: pd.DataFrame
    retriever: Retriever
    generator: AnswerGenerator

    @classmethod
    def initialize(cls) -> "RagPipeline":
        store = VectorStore.load()
        retriever = Retriever(store)
        generator = AnswerGenerator()
        return cls(store=store, retriever=retriever, generator=generator)

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        return self.retriever.search(query=query, top_k=top_k)

    def answer(self, query: str, top_k: int = 4) -> Answer:
        docs = self.retrieve(query, top_k=top_k)
        return self.generator.generate(query=query, docs=docs)


def evaluate_queries(queries: list[str], top_k: int = 3) -> pd.DataFrame:
    pipeline = RagPipeline.initialize()
    rows = []
    for query in queries:
        docs = pipeline.retrieve(query, top_k=top_k)
        best = docs[0] if docs else None
        rows.append(
            {
                "query": query,
                "top_title": best.title if best else None,
                "score": best.score if best else None,
                "url": best.url if best else None,
            }
        )
    return pd.DataFrame(rows)
