import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from cbow_tensorflow_utils import (
    build_cbow_corpus,
    sentence_vector,
    tokenize,
)


df = pd.read_csv("../datasets/CBOW/twitter_dataset.csv")
texts = df["Text"].dropna().tolist()
sentences = [tokenize(text) for text in texts]

_, embeddings, word_to_id, _ = build_cbow_corpus(
    sentences, embedding_dim=64, window=2, epochs=5
)
doc_vecs = np.asarray([
    sentence_vector(sentence, embeddings, word_to_id)
    for sentence in sentences
])
print("doc matrix:", doc_vecs.shape)

query = "happy day great"
query_vector = sentence_vector(
    tokenize(query), embeddings, word_to_id
).reshape(1, -1)
sims = cosine_similarity(query_vector, doc_vecs).flatten()

for index in sims.argsort()[::-1][:5]:
    print(f"  [{sims[index]:.3f}] {texts[index][:120]}")
