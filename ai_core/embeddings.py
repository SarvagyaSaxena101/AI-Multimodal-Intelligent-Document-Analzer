from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    model = _get_model()
    embs = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embs]
