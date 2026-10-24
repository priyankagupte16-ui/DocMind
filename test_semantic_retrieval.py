from pipeline import build_vector_index, load_embedding_model, search_similar_chunks


def main():
    chunks = [
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

    model = load_embedding_model()
    index = build_vector_index(chunks, model)
    question = "How can we stop a model from fitting noise in the training data?"
    results = search_similar_chunks(question, model, index, chunks, top_k=2)

    print("Question:", question)
    for result in results:
        print(
            f"{result['score']:.3f} | {result['filename']} | page {result['page']}"
        )
        print(result["text"][:80], "...", sep="")

    top = results[0]
    assert top["filename"] == "ml_notes.txt", top
    assert top["score"] > results[1]["score"]
    print("PASS: semantic search ranked the ML passage above photosynthesis.")


if __name__ == "__main__":
    main()
