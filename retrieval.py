import faiss
import numpy as np

from config import TOP_K
from embeddings import embed_query


def retrieve_documents(vector_store, question, top_k=None):
    """Return the most relevant document chunks."""

    if not question.strip():
        raise ValueError("Please enter a question.")

    index = vector_store["index"]

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(
        embed_query(question),
        dtype="float32",
    )

    k = min(top_k or TOP_K, index.ntotal)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append(
            {
                "text": vector_store["texts"][int(idx)],
                "source": vector_store["metadatas"][int(idx)]["source"],
                "chunk": vector_store["metadatas"][int(idx)]["chunk"],
                "score": float(score),
            }
        )

    return results