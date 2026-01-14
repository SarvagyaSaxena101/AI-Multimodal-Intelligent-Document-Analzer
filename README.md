# 📄 Document AI Chatbot

A Streamlit-based application that allows users to upload documents (PDFs or images), extract text using OCR, build a context-aware vector store from the extracted information, and then chat with an AI model to get contextually relevant answers. This project implements a Retrieval-Augmented Generation (RAG) pipeline to enhance the AI's responses.

# AI Multimodal Project

A compact, local-first Retrieval-Augmented Generation (RAG) toolkit that extracts text from documents and images, builds a vector index, and provides a chat interface over the extracted content.

Key components live in the `ai_core` package: `chat.py`, `embeddings.py`, `ocr.py`, and `vector_store.py`. The app entrypoint is `app.py` which launches the Streamlit UI.

## Highlights

- Upload PDFs and images (PNG/JPG/TIFF) and extract text using OCR.
- Chunk extracted text, convert to embeddings, and index with FAISS for fast similarity search.
- Retrieval-augmented chat: the model answers queries using relevant document context.
- Lightweight, modular codebase so you can swap models, embedding backends, or vector stores.

## Quickstart (Local)

Prerequisites:

- Python 3.9+
- `pip`

Install and run:

```bash
git clone <your-repo-url>
cd "AI Multimodal Project"
python -m venv venv
.\\venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` with your OpenRouter (or chosen provider) API key:

```env
OPENROUTER_API_KEY="sk-YOUR_OPENROUTER_API_KEY"
```

Run the app:

```bash
streamlit run app.py
```

This opens the Streamlit UI where you can upload documents and start a chat.

## Example: Use `ai_core` programmatically

Minimal snippet to embed text and query the vector store (adapt to your project):

```python
from ai_core.embeddings import EmbeddingClient
from ai_core.vector_store import VectorStore

text_chunks = ["First chunk text...", "Second chunk..."]
emb = EmbeddingClient()
vectors = emb.embed_texts(text_chunks)
vs = VectorStore()
vs.add_documents(text_chunks, vectors)

# Retrieve
results = vs.similarity_search("Ask a question about the doc", top_k=4)
print(results)
```

See the `ai_core` modules for function names and parameters.

## Project Structure

- `app.py` — Streamlit application (entrypoint)
- `ai_core/` — core modules
  - `chat.py` — chat orchestration and prompt construction
  - `embeddings.py` — embedding client wrapper
  - `ocr.py` — OCR helpers (EasyOCR + PDF helpers)
  - `vector_store.py` — FAISS index and persistence helpers

## Configuration & Notes

- The default embedding model and chat model are configured in the code; swap them in `ai_core/embeddings.py` and `ai_core/chat.py`.
- Large PDFs: the project chunks text; tune chunk size and overlap in the vectorization pipeline for your use-case.

## Development

To run linting or tests (if you add them):

```bash
# example
pytest
flake8
```

If you change dependencies, update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

## Roadmap Ideas

- Structured extraction pipelines (invoices, forms)
- Multi-document cross-search
- Confidence and provenance tracking for answers
- Better PDF layout parsing (layout-aware models)

---

If you'd like, I can also:

- add example screenshots/GIFs for the UI,
- add small unit tests for `ai_core` modules,
- or prepare a `docker-compose` for local reproducibility.

## License

MIT — see the `LICENSE` file.

## Contact

Open an issue or PR in this repository for questions, improvements, or feature requests.
