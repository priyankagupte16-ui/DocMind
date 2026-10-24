import os

from pipeline import (
    MissingAPIKeyError,
    build_answer_prompt,
    generate_answer,
    search_similar_chunks,
    build_vector_index,
    load_embedding_model,
)


CHUNKS = [
    {
        "filename": "ml_notes.txt",
        "page": 1,
        "text": (
            "A model memorizes the training set and then fails on new examples. "
            "This can be reduced with more data, simpler models, dropout, "
            "and regularization."
        ),
    },
    {
        "filename": "biology.txt",
        "page": 2,
        "text": (
            "Photosynthesis lets plants turn sunlight, water, and carbon dioxide "
            "into glucose and oxygen."
        ),
    },
]


def test_prompt_includes_sources():
    prompt = build_answer_prompt(
        "How can overfitting be prevented?",
        CHUNKS[:1],
    )
    assert "ml_notes.txt, page 1" in prompt
    assert "regularization" in prompt
    assert "How can overfitting be prevented?" in prompt
    print("PASS: prompt includes question, passage, and source.")


def test_missing_api_key():
    previous = os.environ.pop("GROQ_API_KEY", None)
    try:
        generate_answer("test", CHUNKS[:1])
        raise AssertionError("Expected MissingAPIKeyError")
    except MissingAPIKeyError:
        print("PASS: missing API key is handled.")
    finally:
        if previous is not None:
            os.environ["GROQ_API_KEY"] = previous


def test_live_answer_if_key_present():
    if not os.getenv("GROQ_API_KEY", "").strip():
        print("SKIP: no GROQ_API_KEY, live LLM call was not tested.")
        return

    model = load_embedding_model()
    index = build_vector_index(CHUNKS, model)
    question = "How can we stop a model from fitting noise in the training data?"
    results = search_similar_chunks(question, model, index, CHUNKS, top_k=3)
    answer = generate_answer(question, results)
    print("LIVE ANSWER:", answer)
    assert answer
    print("PASS: live Groq answer was generated.")


if __name__ == "__main__":
    test_prompt_includes_sources()
    test_missing_api_key()
    test_live_answer_if_key_present()
