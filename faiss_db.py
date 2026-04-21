import faiss
import numpy as np

class FAISSDatabase:

    def build(self, df, vectors):
        self.data = df.to_dict("records")

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)  # type: ignore

    def search(self, query_vector, k=3):
        vec = np.array(query_vector).reshape(1, -1).astype("float32")

        D, I = self.index.search(vec, k)  # type: ignore

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue

            item = self.data[idx]
            results.append({
                "query": item["Query"],
                "category": item["Category"],
                "response": item["Response"],
                "score": float(1 / (1 + score))
            })

        return results