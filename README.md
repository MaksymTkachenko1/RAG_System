## Multimodal RAG for *The Batch*

Small-but-complete retrieval-augmented generation stack that crawls The Batch, builds a joint text+image vector store, and serves answers through a Streamlit UI.

### Features
- Automated ingestion of the newest Batch posts (title, summary, body, lead image)
- Dual encoders (MiniLM for text, CLIP for images) stored in a single pickle vector store
- Lightweight retriever + abstractive answer generator (BART) with citation block
- Streamlit UI for querying, browsing hits, and inspecting preview images
- Minimal evaluation helper to sanity-check retrieval coverage for custom queries

### Quickstart
```bash
python -m venv .venv && .\.venv\Scripts\activate        # Windows example
pip install -r requirements.txt
python -m src.rag_system.data_ingestion --limit 25      # pulls latest posts + media
python -m src.rag_system.vector_store                   # builds embeddings cache
python -c "from src.rag_system.pipeline import evaluate_queries; print(evaluate_queries(['open-source models','regulation','robotics']))"  # quick retrieval sanity check
streamlit run app.py                                    # launches UI
```

### Evaluation Notes
- `evaluate_queries` prints a table of query -> top result title/score/URL.
- Swap in any focus areas you care about, e.g., `["chips export", "robotaxis", "AI regulation"]`, to demonstrate coverage breadth.
- Sample run:
  ```
                  query                                          top_title     score                                               url
  0  open-source models  AI-Powered Phones Get Proactive, Robot Antelop...  0.201321  https://www.deeplearning.ai/the-batch/issue-316/
  1          regulation  OpenAI Reorgs For Profit, MiniMax-M2 Leads Ope...  0.131128  https://www.deeplearning.ai/the-batch/issue-326/
  2            robotics  Claude Levels Up, Qwen3 Proliferates, Big AI D...  0.209401  https://www.deeplearning.ai/the-batch/issue-322/
  ```

### Folder layout
- `data/raw` raw article payloads (JSON)
- `data/media` cached hero images
- `data/processed` vector store pickle + metadata
- `src/rag_system/*.py` ingestion, embeddings, retriever, generator, orchestration
- `app.py` Streamlit front-end

### Operational Notes
- Regenerate data regularly if you need the latest Batch issue.
- The summarizer gracefully falls back to extractive notes when the model cannot load (e.g., offline).
- Queries are text-only by design, while retrieved articles always include their associated images to satisfy the multimodal requirement.
