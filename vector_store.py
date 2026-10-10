import faiss
import numpy as np

from embeddings import embed_texts


def create_vector_store(texts, metadatas):
    """Create a FAISS index from document chunks."""

    if not texts:
        raise ValueError("Cannot create a vector store without text.")

    if len(texts) != len(metadatas):
        raise ValueError("Texts and metadata must have the same length.")

    vectors = embed_texts(texts)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return {
        "index": index,
        "texts": texts,
        "metadatas": metadatas,
    }