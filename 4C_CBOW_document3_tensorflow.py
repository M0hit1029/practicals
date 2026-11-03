from cbow_tensorflow_utils import (
    build_cbow_corpus,
    most_similar,
    tokenize,
)


with open("../datasets/CBOW/document3.txt", encoding="utf-8") as file:
    text = file.read()

sentences = [tokenize(raw) for raw in text.split(".") if raw.strip()]
print("sentences:", len(sentences))
print(sentences[0])

_, embeddings, word_to_id, vocabulary = build_cbow_corpus(
    sentences, embedding_dim=64, window=2, epochs=50, max_pairs=200000
)
print("vocab:", len(vocabulary) - 2)

for word in ["covid", "virus", "influenza", "transmission"]:
    print(f"\nsimilar to {word!r}:")
    for similar_word, score in most_similar(
        word, embeddings, word_to_id, vocabulary
    ):
        print(f"  {similar_word}: {score:.3f}")
