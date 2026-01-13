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

    def search(self, query_text_or_vector, k=5):
        # expect query_text_or_vector to be vector already for simplicity
        q = np.array(query_text_or_vector, dtype="float32").reshape(1, -1)
        D, I = self.index.search(q, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            doc_id = self.ids[idx]
            results.append({"id": doc_id, "score": float(dist), "meta": self.meta.get(doc_id)})
        return results
