from chunking import split_documents
from vector_store import create_vector_store
from retrieval import retrieve_documents
from llm import generate_answer


def build_rag_pipeline(documents):
    """Prepare the searchable index from loaded documents."""

    texts, metadatas = split_documents(documents)

    vector_store = create_vector_store(texts, metadatas)

    return vector_store


def ask_question(vector_store, question):
    """Retrieve relevant content and generate an answer."""

    retrieved_chunks = retrieve_documents(
        vector_store,
        question,
    )

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks,
    }