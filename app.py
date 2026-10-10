import streamlit as st

from document_loader import load_uploaded_document
from rag_pipeline import build_rag_pipeline, ask_question


st.set_page_config(
    page_title="My Personal RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 My Personal RAG Chatbot")
st.write(
    "Upload PDF or TXT documents and ask questions "
    "about their contents."
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            try:
                with st.spinner(
                    "Reading documents and building the search index..."
                ):
                    documents = [
                        load_uploaded_document(file)
                        for file in uploaded_files
                    ]

                    new_vector_store = build_rag_pipeline(documents)

                    st.session_state.vector_store = new_vector_store
                    st.session_state.indexed_files = [
                        doc["source"] for doc in documents
                    ]
                    st.session_state.messages = []

                st.success("Documents processed successfully!")

            except Exception as error:
                st.error(f"Could not process documents: {error}")

    if st.session_state.indexed_files:
        st.subheader("Indexed documents")

        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")

        if st.button("Clear documents", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']}** — "
                        f"chunk {source['chunk']}"
                    )
                    st.caption(
                        f"Similarity score: {source['score']:.3f}"
                    )
                    st.write(source["text"])


question = st.chat_input("Ask a question about your documents...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Upload and process documents before asking questions.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    result = ask_question(
                        st.session_state.vector_store,
                        question,
                    )

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("Sources used"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source['source']}** — "
                                f"chunk {source['chunk']}"
                            )
                            st.caption(
                                f"Similarity score: {source['score']:.3f}"
                            )
                            st.write(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )

            except Exception as error:
                st.error(
                    f"Something went wrong: {error}"
                )