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

st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


def inject_custom_css():
    """Warm cream and terracotta styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        /* Base typography */
        html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, label, input, textarea {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Fraunces', serif !important;
            color: #2B2118;
            letter-spacing: -0.01em;
        }

        /* Hide Streamlit chrome (keep header so the sidebar toggle works) */
        #MainMenu, footer {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stToolbar"] {display: none;}
        .stDeployButton {display: none;}
        header[data-testid="stHeader"] {background: transparent;}

        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #F1E8DB;
            border-right: 1px solid #E6DAC8;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #E6DAC8;
            background: #FFFDF9;
            color: #2B2118;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            border-color: #D9774B;
            color: #D9774B;
            transform: translateY(-1px);
        }
        .stButton > button[kind="primary"] {
            background: #D9774B;
            color: #FFFFFF;
            border: none;
            box-shadow: 0 2px 6px rgba(217, 119, 75, 0.3);
        }
        .stButton > button[kind="primary"]:hover {
            background: #C4663B;
            color: #FFFFFF;
        }

        /* File uploader */
        [data-testid="stFileUploader"] section {
            border: 2px dashed #D9774B;
            border-radius: 14px;
            background: #FFFDF9;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            background: #FFFFFF;
            border: 1px solid #EADFCF;
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 1px 4px rgba(43, 33, 24, 0.06);
        }

        /* Chat input */
        [data-testid="stChatInput"] {
            border-radius: 999px;
            border: 1px solid #E6DAC8;
            background: #FFFDF9;
        }

        /* Expanders and metrics as soft cards */
        [data-testid="stExpander"] {
            background: #FFFDF9;
            border: 1px solid #EADFCF;
            border-radius: 12px;
        }
        [data-testid="stMetric"] {
            background: #FFFDF9;
            border: 1px solid #EADFCF;
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
        }

        /* Hero and step cards */
        .dm-hero {
        .dm-hero {
    background: #F7EFE3;
    border: 1px solid #E6DAC8;
    border-radius: 18px;
    padding: 2rem 2rem;
    margin-bottom: 1.5rem;
}
            
        }
        .dm-hero h1 {
            margin: 0;
            font-size: 3rem;
            color: #2B2118;
        }
        .dm-hero p {
            margin: 0.6rem 0 0 0;
            font-size: 1.1rem;
            color: #7A6A5C;
        }
        .dm-step {
            background: #FFFDF9;
            border: 1px solid #EADFCF;
            border-radius: 16px;
            padding: 1.2rem 1.2rem;
            height: 100%;
            box-shadow: 0 1px 4px rgba(43, 33, 24, 0.05);
        }
        .dm-step .num {
            color: #D9774B;
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.1rem;
        }
        .dm-step h4 {
            margin: 0.3rem 0 0.3rem 0;
            font-family: 'Fraunces', serif;
            color: #2B2118;
        }
        .dm-step p {
            margin: 0;
            color: #7A6A5C;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    defaults = {
        "messages": [],
        "chunks": [],
        "index": None,
        "file_key": None,
        "doc_meta": None,
        "processed": False,
        "top_k": 3,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_document_state():
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.file_key = None
    st.session_state.doc_meta = None
    st.session_state.processed = False


def compute_doc_meta(uploaded_files, chunks):
    """Build compact sidebar stats from uploaded files and chunks."""
    names = [file.name for file in uploaded_files]
    total_chars = sum(len(chunk["text"]) for chunk in chunks)
    pages = {chunk["page"] for chunk in chunks if chunk.get("page") is not None}
    return {
        "names": names,
        "char_count": total_chars,
        "page_count": len(pages) if pages else None,
        "chunk_count": len(chunks),
    }


def process_uploaded_files(uploaded_files):
    """Extract chunks and build the vector index for the current upload."""
    all_chunks = []
    had_unreadable = False

    with st.status("Processing document…", expanded=True) as status:
        status.write("Extracting text…")
        for file in uploaded_files:
            try:
                file_chunks = extract_chunks_from_file(file.name, file.getvalue())
                if file_chunks:
                    all_chunks.extend(file_chunks)
                else:
                    had_unreadable = True
                    st.warning(
                        f"No readable text found in {file.name}. "
                        "Scanned PDFs may require OCR."
                    )
            except Exception:
                had_unreadable = True
                st.error(
                    f"Could not read {file.name}. "
                    "The file may be empty, corrupted, or unreadable."
                )

        if not all_chunks:
            status.update(label="Processing failed", state="error")
            clear_document_state()
            if not had_unreadable:
                st.error("No readable content found in the uploaded file(s).")
            return False

        status.write("Chunking and building search index…")
        try:
            model = get_embedding_model()
            index = build_vector_index(all_chunks, model)
        except Exception:
            status.update(label="Indexing failed", state="error")
            clear_document_state()
            st.error(
                "Could not build the document index. "
                "Please try again with a different file."
            )
            return False

        file_key = tuple((file.name, file.size) for file in uploaded_files)
        st.session_state.chunks = all_chunks
        st.session_state.index = index
        st.session_state.file_key = file_key
        st.session_state.doc_meta = compute_doc_meta(uploaded_files, all_chunks)
        st.session_state.processed = True
        st.session_state.messages = []
        status.update(label="Document ready", state="complete")

    return True


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📄 DocMind")
        st.caption("Ask clear questions about your documents.")

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Select one or more PDF or TXT files.",
        )

        with st.expander("Settings"):
            st.session_state.top_k = st.slider(
                "Chunks to retrieve",
                min_value=1,
                max_value=8,
                value=int(st.session_state.top_k),
                help="How many document passages to use when answering.",
            )

        process_clicked = st.button(
            "Process document",
            type="primary",
            use_container_width=True,
        )

        if process_clicked:
            if not uploaded_files:
                st.warning("Upload a PDF or TXT file before processing.")
            else:
                process_uploaded_files(uploaded_files)

        if st.session_state.processed and st.session_state.doc_meta:
            meta = st.session_state.doc_meta
            st.markdown("---")
            st.caption(", ".join(meta["names"]))
            cols = st.columns(2)
            if meta["page_count"] is not None:
                cols[0].metric("Pages", meta["page_count"])
            else:
                cols[0].metric("Characters", f"{meta['char_count']:,}")
            cols[1].metric("Chunks", meta["chunk_count"])
            if meta["page_count"] is not None:
                st.caption(f"{meta['char_count']:,} characters extracted")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return uploaded_files


def render_welcome():
    st.markdown(
        """
        <div class="dm-hero">
            <h1>DocMind</h1>
            <p>Upload a document and ask it anything. Answers come with the exact passages they were drawn from.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    steps = [
        (col1, "1", "📄 Upload", "Add a PDF or TXT file from the sidebar."),
        (col2, "2", "⚙️ Process", "Click Process document to read and index the text."),
        (col3, "3", "💬 Ask", "Chat about the document once it is ready."),
    ]
    for col, num, title, text in steps:
        with col:
            st.markdown(
                f"""
                <div class="dm-step">
                    <div class="num">Step {num}</div>
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{format_source_label(source)}**")
                        st.caption(f"Similarity: {source['score']:.3f}")
                        st.write(source["text"])


def answer_question(question):
    """Retrieve chunks and generate an answer; returns (answer, sources) or None."""
    with st.spinner("Searching similar document sections…"):
        model = get_embedding_model()
        results = search_similar_chunks(
            question,
            model,
            st.session_state.index,
            st.session_state.chunks,
            top_k=int(st.session_state.top_k),
        )

    if not results:
        st.info("No document chunks are available to search.")
        return None

    try:
        with st.spinner("Thinking…"):
            # generate_answer returns full text (no streaming API).
            answer = generate_answer(question, results)
    except MissingAPIKeyError as error:
        st.error(str(error))
        st.info(
            "Create a free Groq key at https://console.groq.com/keys "
            "and put it in a `.env` file as GROQ_API_KEY=your_key"
        )
        return None
    except AuthenticationError:
        st.error(
            "Groq rejected the API key. Check GROQ_API_KEY in your `.env` file."
        )
        return None
    except APIError:
        st.error(
            "The language model request failed. "
            "Please try again in a moment."
        )
        return None
    except Exception:
        st.error("Could not generate an answer. Please try again.")
        return None

    if results[0]["score"] < 0.3:
        answer = (
            f"{answer}\n\n"
            "_The best match is weak. Treat this answer carefully; "
            "the document may not discuss this topic._"
        )

    return answer, results


def render_chat():
    st.markdown("### Chat")
    if not os.getenv("GROQ_API_KEY", "").strip():
        st.warning(
            "No GROQ_API_KEY found. Add it to a `.env` file before asking a question."
        )

    render_chat_history()

    ready = bool(st.session_state.processed and st.session_state.index is not None)
    prompt = st.chat_input(
        "Ask anything about your document…",
        disabled=not ready,
    )

    if not prompt:
        return

    if not ready:
        st.warning("Process a document in the sidebar before asking questions.")
        return

    question = prompt.strip()
    if not question:
        st.warning("Please enter a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result = answer_question(question)
        if result is None:
            return

        answer, sources = result
        st.markdown(answer)
        with st.expander("Sources"):
            for source in sources:
                st.markdown(f"**{format_source_label(source)}**")
                st.caption(f"Similarity: {source['score']:.3f}")
                st.write(source["text"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


def main():
    inject_custom_css()
    init_session_state()

    render_sidebar()

    if not st.session_state.processed:
        render_welcome()
    else:
        st.markdown("# DocMind")
        st.caption("Your document question-answering assistant")
        render_chat()


main()