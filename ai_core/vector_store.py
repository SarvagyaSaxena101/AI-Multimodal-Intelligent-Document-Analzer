import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.meta = {}
        self.ids = []

    def add(self, doc_id: str, vector, metadata: dict):
        vec = np.array(vector, dtype="float32").reshape(1, -1)
        if vec.shape[1] != self.dim:
            # lazy resize: recreate index (simple approach for scaffold)
            self.dim = vec.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)
            # re-add existing vectors (not implemented for scaffold)
        self.index.add(vec)
        self.ids.append(doc_id)
        self.meta[doc_id] = metadata

    def query(self, query_vector, top_k=5):
        q = np.array(query_vector, dtype="float32").reshape(1, -1)
        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            doc_id = self.ids[idx]
            # When returning for RAG, typically only need the text content
            results.append({"id": doc_id, "score": float(dist), "text": self.meta.get(doc_id).get("text")})
        return results
