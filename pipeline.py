import os
from io import BytesIO

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are DocuMind, a document question-answering assistant. "
    "Answer using ONLY the document passages provided by the user. "
    "If those passages do not contain enough information, say that "
    "the document does not contain enough information to answer. "
    "Do not use outside knowledge. Do not invent facts, filenames, "
    "or page numbers."
)


class MissingAPIKeyError(Exception):
    """Raised when GROQ_API_KEY is not set."""


def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def load_embedding_model():
    """Load the local sentence embedding model (downloaded once)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_chunks_from_file(filename, file_bytes):
    """Extract text from a PDF or TXT file and return chunk records."""
    records = []
    name = filename.lower()

    if name.endswith(".txt"):
        document_text = file_bytes.decode("utf-8", errors="replace")
        for chunk in split_text_into_chunks(document_text):
            records.append(
                {
                    "filename": filename,
                    "page": None,
                    "text": chunk,
                }
            )

    elif name.endswith(".pdf"):
        pdf_reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""

            for chunk in split_text_into_chunks(page_text):
                records.append(
                    {
                        "filename": filename,
                        "page": page_number,
                        "text": chunk,
                    }
                )

    return records


def build_vector_index(chunks, model):
    """Turn chunk texts into embeddings and store them in a FAISS index."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_similar_chunks(question, model, index, chunks, top_k=3):
    """Embed the question and return the most similar document chunks."""
    if not chunks or index.ntotal == 0:
        return []

    k = min(top_k, len(chunks))
    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_vector = np.asarray(question_vector, dtype=np.float32)

    scores, indices = index.search(question_vector, k)

    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "score": float(score),
                "filename": chunk["filename"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    return results


def format_source_label(chunk):
    """Build a short source label with filename and page when available."""
    filename = chunk["filename"]
    page = chunk.get("page")
    if page is None:
        return filename
    return f"{filename}, page {page}"


def build_answer_prompt(question, retrieved_chunks):
    """Combine instructions, retrieved passages, and the user question."""
    passage_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        passage_blocks.append(
            f"[{index}] Source: {format_source_label(chunk)}\n{chunk['text']}"
        )

    passages = "\n\n".join(passage_blocks)
    return (
        f"Document passages:\n{passages}\n\n"
        f"Question: {question}\n\n"
        "Write a clear answer using only these passages. "
        "If they are not relevant enough, say so."
    )


def generate_answer(question, retrieved_chunks):
    """Call the Groq LLM with retrieved chunks and return the answer text."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "Missing GROQ_API_KEY. Add it to a .env file in the project folder."
        )

    if not retrieved_chunks:
        raise ValueError("No retrieved passages were provided to the language model.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        temperature=0.2,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_answer_prompt(question, retrieved_chunks),
            },
        ],
    )

    message = completion.choices[0].message
    answer = (message.content or "").strip()
    if not answer:
        answer = (getattr(message, "reasoning", None) or "").strip()
    if not answer:
        raise RuntimeError("The language model returned an empty answer.")
    return answer
