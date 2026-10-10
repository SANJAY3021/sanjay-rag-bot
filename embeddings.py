from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


_model = None


def get_embedding_model():
    """Load the embedding model once and reuse it."""

    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def embed_texts(texts):
    """Convert document chunks into vectors."""

    model = get_embedding_model()

    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_query(query):
    """Convert a question into a vector."""

    model = get_embedding_model()

    return model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )