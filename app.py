from pathlib import Path

import streamlit as st

from src.rag_system.pipeline import RagPipeline


@st.cache_resource(show_spinner="Loading models...")
def load_pipeline() -> RagPipeline:
    return RagPipeline.initialize()


def main() -> None:
    st.set_page_config(page_title="Batch Multimodal RAG", layout="wide")
    st.title("📰 The Batch Multimodal RAG")
    st.caption("Query across recent issues with text + image awareness.")

    pipeline = load_pipeline()

    with st.sidebar:
        st.header("Settings")
        k = st.slider("Results", min_value=1, max_value=6, value=3, step=1)
        st.markdown(
            "Need fresh data? Rerun ingestion from the CLI via `python -m src.rag_system.data_ingestion`."
        )

    query = st.text_input("Ask about AI news", placeholder="e.g., What's new with multimodal models?")
    if st.button("Retrieve & Generate") and query:
        with st.spinner("Thinking..."):
            answer = pipeline.answer(query=query, top_k=k)
        st.subheader("Answer")
        st.write(answer.response)
        st.subheader("Cited Articles")
        for doc in answer.citations:
            with st.container():
                st.markdown(f"**{doc.title}** ([link]({doc.url})) · Score: {doc.score:.2f}")
                cols = st.columns([2, 1])
                cols[0].write(doc.summary)
                if doc.image_path and Path(doc.image_path).exists():
                    cols[1].image(doc.image_path, use_column_width=True)


if __name__ == "__main__":
    main()
