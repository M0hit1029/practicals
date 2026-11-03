import pandas as pd

from cbow_tensorflow_utils import (
    build_cbow_corpus,
    most_similar,
    tokenize,
)


df = pd.read_csv("../datasets/CBOW/twitter_dataset.csv")
sentences = [tokenize(text) for text in df["Text"].dropna()]

_, embeddings, word_to_id, vocabulary = build_cbow_corpus(
    sentences, embedding_dim=64, window=2, epochs=5
)
print("tweets:", len(sentences))
print("vocab:", len(vocabulary) - 2)

for word in ["love", "happy", "day"]:
    print(f"\nsimilar to {word!r}:")
    for similar_word, score in most_similar(
        word, embeddings, word_to_id, vocabulary
    ):
        print(f"  {similar_word}: {score:.3f}")
