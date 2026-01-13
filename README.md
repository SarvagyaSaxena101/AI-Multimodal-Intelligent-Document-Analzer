# AI Multimodal Project

Document AI scaffold: FastAPI backend + Streamlit frontend.

Structure:
- backend_ai/: FastAPI app (OCR, embeddings, FAISS, OpenRouter chat)
- streamlit_work/: Streamlit UI for upload and chat

Quick start (development, no Docker):


1. Use the existing `project1` virtual environment for both backend and frontend (Windows):

```powershell
cd project1
.\Scripts\activate
```

2. Install dependencies and run the backend:

```powershell
cd ..\backend_ai
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

3. Run the Streamlit UI (while `project1` venv is active):

```powershell
cd ..\streamlit_work
pip install -r requirements.txt
streamlit run app.py
```

Configuration:
- Put `OPENROUTER_API_KEY` in the project root `.env` (or export it in your shell).
- Use the Streamlit sidebar `API Base URL` to point to your backend (defaults to `http://localhost:8000/api`).

This repository is a scaffold. Next steps: add robust PDF parsing, LayoutLM/Donut integration, field extraction pipelines, confidence scoring and active learning loops.
