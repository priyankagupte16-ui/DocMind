# DocuMind Learning Log

## Stage: Document embeddings and semantic retrieval

- **What we completed:** PDF/TXT extraction still works. Chunks now keep filename and PDF page number. Chunks are converted to embeddings with `sentence-transformers` and stored in FAISS. Questions are searched by meaning, not shared keywords. No LLM yet.
- **Important files:** `pipeline.py` (chunking, extraction, embeddings, FAISS search), `app.py` (Streamlit UI), `requirements.txt`
- **Key code:** `load_embedding_model()`, `model.encode()`, `build_vector_index()`, `search_similar_chunks()`
- **Concepts learned:** embeddings, cosine similarity via normalized inner product, FAISS index, semantic retrieval
- **Commands used:** `.\.venv\Scripts\python.exe -m pip list`, `streamlit run app.py`
- **Current status:** Semantic retrieval of top 3 chunks works in code. Answer generation is not implemented.
- **Next step:** Connect a language model so retrieved chunks are turned into a grounded answer.

## Stage: LLM answer generation

- **What we completed:** After FAISS retrieval, DocuMind sends the top 3 chunks plus the question to Groq (`openai/gpt-oss-20b`). The model is instructed to answer only from those passages. The UI shows the answer and sources. The API key is read from `.env`.
- **Important files:** `pipeline.py` (`build_answer_prompt`, `generate_answer`), `app.py` (answer + source display, API errors), `.env.example`, `.gitignore`
- **Key code:** system prompt, `client.chat.completions.create()`, `GROQ_API_KEY` via `python-dotenv`
- **Concepts learned:** RAG generation, grounding, hallucinations, environment variables, API error handling
- **Commands used:** `pip install groq`, `streamlit run app.py`
- **Current status:** Retrieval verified. LLM path is implemented; a live Groq answer requires your own `GROQ_API_KEY`. Follow-up chat history is not implemented.
- **Next step:** Conversation history for follow-up questions, then UI polish and tests.
