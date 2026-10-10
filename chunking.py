from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """Split documents into chunks while preserving source information."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts = []
    metadatas = []

    for document in documents:
        chunks = splitter.split_text(document["text"])

        for chunk_number, chunk in enumerate(chunks, start=1):
            if chunk.strip():
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": document["source"],
                        "chunk": chunk_number,
                    }
                )

    if not texts:
        raise ValueError("No text chunks were created.")

    return texts, metadatas