import os

import streamlit as st
from groq import APIError, AuthenticationError

from pipeline import (
    MissingAPIKeyError,
    build_vector_index,
    extract_chunks_from_file,
    format_source_label,
    generate_answer,
    load_embedding_model,
    search_similar_chunks,
)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# Page configuration
st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 DocuMind")
st.subheader("Your AI-Powered Document Question Answering Assistant")
st.write("Upload your documents and ask questions about their content.")

st.divider()

# Upload documents
st.header("📂 Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose PDF or text files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if "file_key" not in st.session_state:
    st.session_state.file_key = None
    st.session_state.chunks = []
    st.session_state.index = None

all_chunks = []

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) uploaded successfully!")

    for file in uploaded_files:
        st.write(f"📄 **{file.name}**")

        try:
            file_chunks = extract_chunks_from_file(file.name, file.getvalue())

            if file_chunks:
                all_chunks.extend(file_chunks)
                document_text = "\n\n".join(chunk["text"] for chunk in file_chunks)

                st.caption(
                    f"Extracted {len(document_text)} characters "
                    f"and created {len(file_chunks)} chunk(s)."
                )

                with st.expander(f"Preview text: {file.name}"):
                    st.text_area(
                        "Extracted content",
                        document_text,
                        height=250,
                        key=f"preview_{file.name}"
                    )

                with st.expander(f"View chunks: {file.name}"):
                    for index, chunk in enumerate(file_chunks, start=1):
                        page_label = (
                            f" (page {chunk['page']})"
                            if chunk["page"] is not None
                            else ""
                        )
                        st.markdown(f"**Chunk {index}{page_label}**")
                        st.write(chunk["text"])
                        st.divider()

            else:
                st.warning(
                    f"No readable text found in {file.name}. "
                    "Scanned PDFs may require OCR."
                )

        except Exception as error:
            st.error(f"Could not read {file.name}: {error}")

    file_key = tuple((file.name, file.size) for file in uploaded_files)

    if all_chunks and st.session_state.file_key != file_key:
        with st.spinner("Creating document embeddings..."):
            model = get_embedding_model()
            st.session_state.index = build_vector_index(all_chunks, model)
            st.session_state.chunks = all_chunks
            st.session_state.file_key = file_key
    elif not all_chunks:
        st.session_state.file_key = None
        st.session_state.chunks = []
        st.session_state.index = None
else:
    st.session_state.file_key = None
    st.session_state.chunks = []
    st.session_state.index = None

st.divider()

# Question section
st.header("💬 Ask Questions")

if not os.getenv("GROQ_API_KEY", "").strip():
    st.warning(
        "No GROQ_API_KEY found. Add it to a `.env` file before asking a question."
    )

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the main topic of my document?"
)

if st.button("Ask DocuMind"):
    if not st.session_state.chunks or st.session_state.index is None:
        st.warning("Please upload a readable PDF or TXT document first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching similar document sections..."):
            model = get_embedding_model()
            results = search_similar_chunks(
                question,
                model,
                st.session_state.index,
                st.session_state.chunks,
                top_k=3,
            )

        if not results:
            st.info("No document chunks are available to search.")
        else:
            try:
                with st.spinner("Generating an answer from the retrieved passages..."):
                    answer = generate_answer(question, results)

                st.subheader("🧠 Answer")
                st.write(answer)

                st.subheader("📌 Sources")
                seen_sources = []
                for result in results:
                    label = format_source_label(result)
                    if label not in seen_sources:
                        seen_sources.append(label)
                        st.markdown(f"- {label}")

                with st.expander("Retrieved passages"):
                    for result in results:
                        st.markdown(f"**{format_source_label(result)}**")
                        st.caption(f"Similarity: {result['score']:.3f}")
                        st.write(result["text"])
                        st.divider()

                if results[0]["score"] < 0.3:
                    st.info(
                        "The best match is weak. Treat this answer carefully; "
                        "the document may not discuss this topic."
                    )

            except MissingAPIKeyError as error:
                st.error(str(error))
                st.info(
                    "Create a free Groq key at https://console.groq.com/keys "
                    "and put it in a `.env` file as GROQ_API_KEY=your_key"
                )
            except AuthenticationError:
                st.error(
                    "Groq rejected the API key. Check GROQ_API_KEY in your `.env` file."
                )
            except APIError as error:
                st.error(f"The language model request failed: {error}")
            except Exception as error:
                st.error(f"Could not generate an answer: {error}")
