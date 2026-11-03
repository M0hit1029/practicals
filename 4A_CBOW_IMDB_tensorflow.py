import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from cbow_tensorflow_utils import (
    build_cbow_corpus,
    most_similar,
    sentence_vector,
    tokenize,
)


df = pd.read_csv("../datasets/CBOW/IMDB Dataset.csv")
df = df.sample(10000, random_state=42).reset_index(drop=True)
y = (df["sentiment"] == "positive").astype(int).to_numpy()
sentences = [tokenize(review) for review in df["review"]]

_, embeddings, word_to_id, vocabulary = build_cbow_corpus(
    sentences, embedding_dim=64, window=2, epochs=5
)
X = np.asarray([
    sentence_vector(sentence, embeddings, word_to_id)
    for sentence in sentences
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=10)

loss, accuracy = model.evaluate(X_te, y_te)
print(f"test acc: {accuracy:.4f}")
print("similar to 'great':", most_similar(
    "great", embeddings, word_to_id, vocabulary
))

for name, values in history.history.items():
    plt.plot(values, label=name)
plt.legend()
plt.grid()
plt.show()
