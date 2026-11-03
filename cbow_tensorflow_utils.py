import re
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def tokenize(text):
    return re.findall(r"[a-z]+", str(text).lower())


def build_cbow_corpus(sentences, embedding_dim=64, window=2, epochs=5,
                      max_pairs=200000, batch_size=512):
    """Train a CBOW embedding model using TensorFlow/Keras."""
    counts = Counter(token for sentence in sentences for token in sentence)
    vocabulary = ["<PAD>", "<UNK>"] + sorted(counts)
    word_to_id = {word: index for index, word in enumerate(vocabulary)}

    pairs = []
    for sentence in sentences:
        ids = [word_to_id.get(word, 1) for word in sentence]
        for center_index, target in enumerate(ids):
            context = [
                ids[index]
                for index in range(max(0, center_index - window),
                                   min(len(ids), center_index + window + 1))
                if index != center_index
            ]
            if len(context) == 2 * window:
                pairs.append((context, target))
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        raise ValueError("The corpus does not contain enough tokens for CBOW.")

    context_ids = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
    targets = np.asarray([pair[1] for pair in pairs], dtype=np.int32)

    inputs = keras.Input(shape=(2 * window,), dtype="int32")
    embeddings = layers.Embedding(len(vocabulary), embedding_dim, name="word_embeddings")
    averaged = layers.Lambda(lambda values: tf.reduce_mean(values, axis=1))(embeddings(inputs))
    outputs = layers.Dense(len(vocabulary), activation="softmax")(averaged)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        context_ids,
        targets,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1 if len(targets) > 10 else 0,
    )

    return model, embeddings, word_to_id, vocabulary


def sentence_vector(sentence, embeddings, word_to_id):
    ids = [word_to_id[word] for word in sentence if word in word_to_id]
    if not ids:
        return np.zeros(embeddings.output_dim, dtype=np.float32)
    return embeddings(np.asarray(ids, dtype=np.int32)).numpy().mean(axis=0)


def most_similar(word, embeddings, word_to_id, vocabulary, topn=5):
    if word not in word_to_id:
        return []
    matrix = embeddings.weights[0].numpy()
    query = matrix[word_to_id[word]]
    scores = matrix @ query / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-8
    )
    scores[word_to_id[word]] = -1
    indices = np.argsort(scores)[::-1]
    return [(vocabulary[index], float(scores[index])) for index in indices[:topn]]
