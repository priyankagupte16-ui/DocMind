# DocMind

Ask questions about your own documents and get answers with the exact passages they came from.

![DocMind screenshot](screenshot.png)

**Live demo:** ADD LINK AFTER DEPLOYING

## How it works

1. **Upload** a PDF or TXT file.
2. **Extract and chunk** the text into small overlapping passages.
3. **Embed** each chunk with a sentence-embedding model and store the vectors in a search index.
4. **Retrieve** the chunks most similar to your question.
5. **Generate** an answer with a Groq-hosted LLM, using only the retrieved chunks, and show them as sources.

## Tech stack

Python, Streamlit, MODEL NAME (embeddings), INDEX LIBRARY (vector search), Groq API (LLM)

## Run it locally

    git clone https://github.com/YOUR-USERNAME/DocMind.git
    cd DocMind
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Create a `.env` file (see `.env.example`) and add your key:

    GROQ_API_KEY=your_key_here

Then run:

    streamlit run app.py

## Features

- Multiple PDF/TXT upload
- Chat-style Q&A with source passages and similarity scores
- Adjustable number of retrieved chunks
- Clear error messages for unreadable files and API problems

## What I would improve next

- Streaming answers
- Conversation memory for follow-up questions
- Retrieval evaluation on a set of test questions
