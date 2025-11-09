from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

from .config import paths

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
ARCHIVE_URL = "https://www.deeplearning.ai/the-batch/"


@dataclass
class Article:
    id: str
    title: str
    published: str
    summary: str
    full_text: str
    url: str
    image_path: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "published": self.published,
            "summary": self.summary,
            "full_text": self.full_text,
            "url": self.url,
            "image_path": self.image_path,
        }


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _download_image(url: str, article_id: str) -> Optional[str]:
    if not url:
        return None
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        response.raise_for_status()
        ext = ".png" if "image/png" in response.headers.get("Content-Type", "") else ".jpg"
        filename = f"{article_id}{ext}"
        output_path = paths.media / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(response.content)
        # Validate file by loading with PIL and re-saving to avoid corrupt blobs.
        with Image.open(output_path) as img:
            img.convert("RGB").save(output_path, format="JPEG", quality=90)
        return str(output_path)
    except Exception:
        return None


def _extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    content = " ".join(paragraphs)
    return _clean_text(content)


def _first_image(soup: BeautifulSoup) -> Optional[str]:
    candidates = soup.find_all("img")
    for tag in candidates:
        src = tag.get("src") or tag.get("data-src")
        if src and src.startswith("http"):
            return src
    return None


def _discover_articles(session: requests.Session, limit: int) -> list[dict]:
    """Fallback scraper that reads the Next.js data blob when RSS is empty."""
    try:
        html = session.get(ARCHIVE_URL, timeout=20).text
    except Exception:
        return []
    soup = BeautifulSoup(html, "html.parser")
    data_tag = soup.find("script", id="__NEXT_DATA__")
    if not data_tag or not data_tag.string:
        return []
    try:
        payload = json.loads(data_tag.string)
        posts = payload["props"]["pageProps"]["posts"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

    records: list[dict] = []
    for post in posts[:limit]:
        slug = post.get("slug", "")
        records.append(
            {
                "id": post.get("id", str(uuid.uuid4())),
                "title": post.get("title", "Untitled"),
                "summary": post.get("custom_excerpt", ""),
                "link": f"https://www.deeplearning.ai/the-batch/{slug}/",
                "published": post.get("published_at", datetime.utcnow().isoformat()),
                "feature_image": post.get("feature_image"),
            }
        )
    return records


class BatchIngestor:
    def __init__(self, feed_url: str = "https://www.deeplearning.ai/category/the-batch/feed/") -> None:
        self.feed_url = feed_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, limit: int = 20) -> List[Article]:
        parsed = feedparser.parse(self.feed_url)
        entries = parsed.entries[:limit]
        if not entries:
            entries = _discover_articles(self.session, limit)
        articles: List[Article] = []

        for entry in tqdm(entries, desc="Downloading articles"):
            article_id = str(entry.get("id") or uuid.uuid4())
            url = entry.get("link")
            try:
                html = self.session.get(url, timeout=20).text
            except Exception:
                continue
            soup = BeautifulSoup(html, "html.parser")
            text = _extract_article_text(html)
            summary = _clean_text(entry.get("summary", "") or text[:320])
            image_url = _first_image(soup)
            if not image_url:
                image_url = entry.get("feature_image")
            image_path = _download_image(image_url, article_id) if image_url else None
            published = entry.get("published", datetime.utcnow().isoformat())
            articles.append(
                Article(
                    id=article_id,
                    title=_clean_text(entry.get("title", "Untitled")),
                    published=published,
                    summary=summary,
                    full_text=text or summary,
                    url=url,
                    image_path=image_path,
                )
            )
        return articles

    def save(self, articles: Iterable[Article]) -> None:
        paths.data_raw.mkdir(parents=True, exist_ok=True)
        paths.data_processed.mkdir(parents=True, exist_ok=True)
        data = [article.to_dict() for article in articles if article.full_text]
        with open(paths.metadata, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)


def run_ingestion(limit: int = 15) -> List[Article]:
    """Convenience helper used by CLI or pipeline."""
    ingestor = BatchIngestor()
    articles = ingestor.fetch(limit=limit)
    ingestor.save(articles)
    return articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest The Batch articles.")
    parser.add_argument("--limit", type=int, default=15, help="Articles to ingest.")
    args = parser.parse_args()
    run_ingestion(limit=args.limit)
