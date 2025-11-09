from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from transformers import pipeline

from .config import models
from .retriever import RetrievalResult


@dataclass
class Answer:
    response: str
    citations: List[RetrievalResult]


class AnswerGenerator:
    def __init__(self) -> None:
        try:
            self.summarizer = pipeline(
                "summarization",
                model=models.summarizer_model,
                tokenizer=models.summarizer_model,
                truncation=True,
            )
        except Exception:
            self.summarizer = None

    def _format_context(self, docs: Iterable[RetrievalResult]) -> str:
        blocks = []
        for doc in docs:
            blocks.append(f"Title: {doc.title}\nURL: {doc.url}\nSnippet: {doc.summary}\n")
        return "\n".join(blocks)

    def generate(self, query: str, docs: List[RetrievalResult]) -> Answer:
        if not docs:
            return Answer(
                response="I could not find relevant Batch articles. Try a broader query.",
                citations=[],
            )

        context = " ".join(doc.full_text for doc in docs)
        # Keep prompt small so BART stays within its positional window (~1k tokens).
        if len(context) > 2000:
            context = context[:2000]

        if self.summarizer and context.strip() and len(context) < 2000:
            prompt = f"Question: {query}\nContext: {context}"
            try:
                summary = self.summarizer(
                    prompt, max_length=256, min_length=32, clean_up_tokenization_spaces=True
                )[0]["summary_text"]
            except Exception:
                summary = ""
        else:
            summary_lines = [f"- {doc.title}: {doc.summary}" for doc in docs]
            summary = f"Based on the latest Batch issues, here is what's relevant:\n" + "\n".join(summary_lines)
        if not summary:
            summary_lines = [f"- {doc.title}: {doc.summary}" for doc in docs]
            summary = f"Based on the latest Batch issues, here is what's relevant:\n" + "\n".join(summary_lines)
        return Answer(response=summary, citations=docs)
