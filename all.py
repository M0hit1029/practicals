# ============================================================================
# Source: 1B_FNN_FashionMNIST.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv("../datasets/FNN and CNN/Fashion MNIST/fashion-mnist_train.csv")
test = pd.read_csv("../datasets/FNN and CNN/Fashion MNIST/fashion-mnist_test.csv")

y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
x_train = train.drop(columns="label").to_numpy()
x_test = test.drop(columns="label").to_numpy()

print(x_train.shape, x_test.shape)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Rescaling(1./255),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.4f}  test accuracy: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 1C_FNN_CIFAR10.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

df = pd.read_csv("../datasets/FNN and CNN/CIFAR10/cifar-10.csv")
y = df["label"].to_numpy()
x = df.drop(columns="label").to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x_train.shape, x_test.shape)

model = keras.Sequential([
    layers.Input(shape=(3072,)),
    layers.Rescaling(1./255),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.4f}  test accuracy: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 1E_FNN_SignMNIST.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv("../datasets/FNN and CNN/Sign Language MNIST/sign_mnist_train.csv")
test = pd.read_csv("../datasets/FNN and CNN/Sign Language MNIST/sign_mnist_test.csv")

y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
x_train = train.drop(columns="label").to_numpy()
x_test = test.drop(columns="label").to_numpy()

print(x_train.shape, x_test.shape)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Rescaling(1./255),
    layers.Dense(128, activation="relu"),
    layers.Dense(25, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.4f}  test accuracy: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 2A_CNN_CIFAR10.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

df = pd.read_csv("../datasets/FNN and CNN/CIFAR10/cifar-10.csv")
y = df["label"].to_numpy()
x = df.drop(columns="label").to_numpy().reshape(-1, 32, 32, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x_train.shape, x_test.shape)

model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.4f}  test accuracy: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 2B_CNN_HandMNIST.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv("../datasets/FNN and CNN/HandWritten MNIST/mnist_train.csv")
test = pd.read_csv("../datasets/FNN and CNN/HandWritten MNIST/mnist_test.csv")

y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
x_train = train.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)
x_test = test.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.4f}  test accuracy: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 2C_CNN_SignMNIST.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv("../datasets/FNN and CNN/Sign Language MNIST/sign_mnist_train.csv")
test = pd.read_csv("../datasets/FNN and CNN/Sign Language MNIST/sign_mnist_test.csv")

y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
x_train = train.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)
x_test = test.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(25, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

loss, acc = model.evaluate(x_test, y_test)
print(f"test loss: {loss:.4f}  test accuracy: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 3A_AE_CreditCard.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("../datasets/AutoEncoder/CreditCard/creditcard.csv")
y = df["Class"].to_numpy()
x = df.drop(columns=["Class", "Time"]).to_numpy()

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train_norm = x[y == 0]
print("normal train:", x_train_norm.shape, "frauds:", y.sum())

model = keras.Sequential([
    layers.Input(shape=(29,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(29),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(x_train_norm, x_train_norm, epochs=5, validation_split=0.1)

recon = model.predict(x)
err = np.mean((x - recon) ** 2, axis=1)

auc = roc_auc_score(y, err)
threshold = np.quantile(err[y == 0], 0.99)
y_pred = (err > threshold).astype(int)

print(f"AUC: {auc:.4f}")
print(classification_report(y, y_pred, target_names=["normal", "fraud"]))

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 3B_AE_ECG_sub_MVTec.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("../datasets/AutoEncoder/ECG/ecg.csv", header=None)
data = df.to_numpy()

x = data[:, :-1].astype("float32")
y = (data[:, -1] != 1).astype(int)  # label 1 = normal, else anomaly

x_min = x.min()
x_max = x.max()
x = (x - x_min) / (x_max - x_min)

x_train_norm = x[y == 0]
print("normal:", x_train_norm.shape, "anomalies:", y.sum())

model = keras.Sequential([
    layers.Input(shape=(140,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(140),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(x_train_norm, x_train_norm, epochs=20, validation_split=0.1)

recon = model.predict(x)
err = np.mean((x - recon) ** 2, axis=1)

auc = roc_auc_score(y, err)
threshold = np.quantile(err[y == 0], 0.95)
y_pred = (err > threshold).astype(int)

print(f"AUC: {auc:.4f}")
print(classification_report(y, y_pred, target_names=["normal", "anomaly"]))

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 3C_AE_Energy_LSTM.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../datasets/AutoEncoder/Energy Consumption/energy consumption.csv")
df = df.dropna()

y = df["Anomaly_Label"].to_numpy().astype(int)
x = df.drop(columns=["Timestamp", "Anomaly_Label"]).to_numpy().astype("float32")

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# sliding windows (W=10)
W = 10
Xw = []
yw = []
for i in range(len(x) - W + 1):
    Xw.append(x[i:i+W])
    yw.append(y[i:i+W].max())
Xw = np.array(Xw)
yw = np.array(yw)

Xw_norm = Xw[yw == 0]
print("windows:", Xw.shape, "normal:", Xw_norm.shape)

model = keras.Sequential([
    layers.Input(shape=(10, Xw.shape[2])),
    layers.LSTM(16, activation="relu"),
    layers.RepeatVector(10),
    layers.LSTM(16, activation="relu", return_sequences=True),
    layers.TimeDistributed(layers.Dense(Xw.shape[2])),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(Xw_norm, Xw_norm, epochs=10, validation_split=0.1)

recon = model.predict(Xw)
err = np.mean((Xw - recon) ** 2, axis=(1, 2))

threshold = np.quantile(err[yw == 0], 0.95)
y_pred = (err > threshold).astype(int)

print(f"anomalies predicted: {y_pred.sum()}  actual: {yw.sum()}")
print(f"threshold: {threshold:.6f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 3D_AE_NASA_LSTM.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../datasets/AutoEncoder/NASA Turbofan ENgine/NASA TurboFan Engine.csv")

df["max_cycle"] = df.groupby("unit")["cycle"].transform("max")
df["healthy"] = df["cycle"] <= df["max_cycle"] * 0.2

feat_cols = [c for c in df.columns if c.startswith("sensor") or c.startswith("setting")]
x = df[feat_cols].to_numpy().astype("float32")

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = (~df["healthy"]).astype(int).to_numpy()

# sliding windows (W=15)
W = 15
Xw = []
yw = []
for i in range(len(x) - W + 1):
    Xw.append(x[i:i+W])
    yw.append(y[i:i+W].max())
Xw = np.array(Xw)
yw = np.array(yw)

Xw_norm = Xw[yw == 0]
print("windows:", Xw.shape, "healthy:", Xw_norm.shape)

model = keras.Sequential([
    layers.Input(shape=(15, Xw.shape[2])),
    layers.LSTM(32, activation="relu"),
    layers.RepeatVector(15),
    layers.LSTM(32, activation="relu", return_sequences=True),
    layers.TimeDistributed(layers.Dense(Xw.shape[2])),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(Xw_norm, Xw_norm, epochs=5, validation_split=0.1)

recon = model.predict(Xw)
err = np.mean((Xw - recon) ** 2, axis=(1, 2))

threshold = np.quantile(err[yw == 0], 0.95)
y_pred = (err > threshold).astype(int)

print(f"flagged faulty: {y_pred.sum()}  actual non-healthy: {yw.sum()}")
print(f"threshold: {threshold:.6f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 3E_AE_ECG_sub_NSLKDD.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("../datasets/AutoEncoder/ECG/ecg.csv", header=None)
data = df.to_numpy()

x = data[:, :-1].astype("float32")
y = (data[:, -1] != 1).astype(int)  # label 1 = normal, else anomaly

x_min = x.min()
x_max = x.max()
x = (x - x_min) / (x_max - x_min)

x_train_norm = x[y == 0]
print("normal:", x_train_norm.shape, "anomalies:", y.sum())

model = keras.Sequential([
    layers.Input(shape=(140,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(140),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(x_train_norm, x_train_norm, epochs=20, validation_split=0.1)

recon = model.predict(x)
err = np.mean((x - recon) ** 2, axis=1)

auc = roc_auc_score(y, err)
threshold = np.quantile(err[y == 0], 0.95)
y_pred = (err > threshold).astype(int)

print(f"AUC: {auc:.4f}")
print(classification_report(y, y_pred, target_names=["normal", "anomaly"]))

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 3F_AE_SignMNIST_anomaly.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, classification_report

train = pd.read_csv("../datasets/FNN and CNN/Sign Language MNIST/sign_mnist_train.csv")
test = pd.read_csv("../datasets/FNN and CNN/Sign Language MNIST/sign_mnist_test.csv")

df = pd.concat([train, test], ignore_index=True)
y_anom = (df["label"] == 0).to_numpy().astype(int)
x = df.drop(columns="label").to_numpy().astype("float32") / 255.0

x_train_norm = x[y_anom == 0]
print("normal (23 classes):", x_train_norm.shape, "anomaly (class 0):", y_anom.sum())

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(x_train_norm, x_train_norm, epochs=5, validation_split=0.1)

recon = model.predict(x)
err = np.mean((x - recon) ** 2, axis=1)

auc = roc_auc_score(y_anom, err)
threshold = np.quantile(err[y_anom == 0], 0.95)
y_pred = (err > threshold).astype(int)

print(f"AUC: {auc:.4f}")
print(classification_report(y_anom, y_pred, target_names=["normal", "anomaly"]))

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 4A_CBOW_IMDB.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("../datasets/CBOW/IMDB Dataset.csv")
df = df.sample(10000, random_state=42).reset_index(drop=True)
y = (df["sentiment"] == "positive").astype(int).to_numpy()

sentences = []
for review in df["review"]:
    tokens = re.findall(r"[a-z]+", review.lower())
    sentences.append(tokens)

print("docs:", len(sentences), "example:", sentences[0][:10])

w2v = Word2Vec(sentences, sg=0)
print("vocab:", len(w2v.wv))

X = []
for sent in sentences:
    vec = w2v.wv.get_mean_vector(sent, ignore_missing=True)
    X.append(vec)
X = np.array(X)
print(X.shape)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=10)

loss, acc = model.evaluate(X_te, y_te)
print(f"test acc: {acc:.4f}")
print("similar to 'great':", w2v.wv.most_similar("great", topn=5))

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 4B_CBOW_Twitter.ipynb
# ============================================================================

import pandas as pd
import re
from gensim.models import Word2Vec

df = pd.read_csv("../datasets/CBOW/twitter_dataset.csv")
texts = df["Text"].dropna().tolist()

sentences = []
for tweet in texts:
    tokens = re.findall(r"[a-z]+", str(tweet).lower())
    sentences.append(tokens)

print("tweets:", len(sentences))

w2v = Word2Vec(sentences, sg=0)
print("vocab:", len(w2v.wv))

for word in ["love", "happy", "day"]:
    if word in w2v.wv:
        print(f"\nsimilar to {word!r}:")
        for w, s in w2v.wv.most_similar(word, topn=5):
            print(f"  {w}: {s:.3f}")

# ============================================================================
# Source: 4C_CBOW_document3.ipynb
# ============================================================================

import re
from gensim.models import Word2Vec

with open("../datasets/CBOW/document3.txt") as f:
    text = f.read()

sentences = []
for raw in text.split("."):
    if raw.strip():
        tokens = re.findall(r"[a-z]+", raw.lower())
        sentences.append(tokens)

print("sentences:", len(sentences))
print(sentences[0])

w2v = Word2Vec(sentences, sg=0, min_count=1, epochs=50)
print("vocab:", len(w2v.wv))

for word in ["covid", "virus", "influenza", "transmission"]:
    if word in w2v.wv:
        print(f"\nsimilar to {word!r}:")
        for w, s in w2v.wv.most_similar(word, topn=5):
            print(f"  {w}: {s:.3f}")

# ============================================================================
# Source: 4D_CBOW_Recommender.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("../datasets/CBOW/twitter_dataset.csv")
texts = df["Text"].dropna().tolist()

sentences = []
for tweet in texts:
    tokens = re.findall(r"[a-z]+", str(tweet).lower())
    sentences.append(tokens)

w2v = Word2Vec(sentences, sg=0)

doc_vecs = []
for sent in sentences:
    vec = w2v.wv.get_mean_vector(sent, ignore_missing=True)
    doc_vecs.append(vec)
doc_vecs = np.array(doc_vecs)
print("doc matrix:", doc_vecs.shape)

query = "happy day great"
q_tokens = re.findall(r"[a-z]+", query.lower())
q_vec = w2v.wv.get_mean_vector(q_tokens, ignore_missing=True).reshape(1, -1)

sims = cosine_similarity(q_vec, doc_vecs).flatten()
top = sims.argsort()[::-1][:5]

print(f"query: {query!r}\n")
for i in top:
    print(f"  [{sims[i]:.3f}] {texts[i][:120]}")

# ============================================================================
# Source: 5A_Transfer_Caltech101.ipynb
# ============================================================================

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train = keras.utils.image_dataset_from_directory(
    "../datasets/Object Detection/caltech-101",
    image_size=(128, 128), validation_split=0.2, subset="training", seed=42)
val = keras.utils.image_dataset_from_directory(
    "../datasets/Object Detection/caltech-101",
    image_size=(128, 128), validation_split=0.2, subset="validation", seed=42)

n_classes = len(train.class_names)
print("classes:", n_classes)

train = train.prefetch(2)
val = val.prefetch(2)

base = keras.applications.VGG16(include_top=False, input_shape=(128, 128, 3))
base.trainable = False

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(n_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train, validation_data=val, epochs=3)

loss, acc = model.evaluate(val)
print(f"val acc (frozen base): {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 5B_Transfer_IntelImage.ipynb
# ============================================================================

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

BASE = "../datasets/Object Detection/Intel Image Classification"
train = keras.utils.image_dataset_from_directory(
    BASE, image_size=(128, 128), validation_split=0.2, subset="training", seed=42)
val = keras.utils.image_dataset_from_directory(
    BASE, image_size=(128, 128), validation_split=0.2, subset="validation", seed=42)

n_classes = len(train.class_names)
print("classes:", train.class_names)

train = train.prefetch(2)
val = val.prefetch(2)

base = keras.applications.MobileNetV2(include_top=False, input_shape=(128, 128, 3))
base.trainable = False

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(n_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train, validation_data=val, epochs=3)

loss, acc = model.evaluate(val)
print(f"val acc: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()

# ============================================================================
# Source: 5C_Transfer_FashionMNIST.ipynb
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv("../datasets/FNN and CNN/Fashion MNIST/fashion-mnist_train.csv")
test = pd.read_csv("../datasets/FNN and CNN/Fashion MNIST/fashion-mnist_test.csv")

y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
x_train = train.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)
x_test = test.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)

# subsample to keep it fast
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]

print(x_train.shape)

base = keras.applications.MobileNetV2(include_top=False, input_shape=(32, 32, 3), alpha=0.35)
base.trainable = False

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Resizing(32, 32)(inputs)
x = layers.Concatenate()([x, x, x])
x = layers.Rescaling(1./127.5, offset=-1)(x)
x = base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

loss, acc = model.evaluate(x_test, y_test)
print(f"test acc: {acc:.4f}")

for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.show()
