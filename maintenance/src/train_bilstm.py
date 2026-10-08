"""v5 BiLSTM: 하나의 모델이 담당 분야와 조치 시점을 동시에 예측 (출력층 2개)

빈칸(패딩) 처리 방식: Keras의 마스크 기능에 의존하지 않는 '양방향 앞 패딩' 방식
  - 정방향 입력: [빈칸 ... 빈칸, w1, w2, ..., wn]
  - 역방향 입력: [빈칸 ... 빈칸, wn, ..., w2, w1]  (단어 순서만 뒤집음)
  - 두 LSTM 모두 빈칸을 먼저 읽고 실제 단어를 마지막에 읽으므로,
    마지막 상태가 실제 단어로 결정된다. (Keras 버전과 무관하게 동작)
"""
import os, json
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from common import tokenize, load, report, to_labels, TYPES, URGS, MODELS

SEED = int(os.environ.get("SEED", 42))
OUT = f"{MODELS}/v5_bilstm"
EMBED_DIM, LSTM_UNITS, DROPOUT, EPOCHS, BATCH = 128, 64, 0.3, 30, 32
tf.random.set_seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

train, val, dev = load("train"), load("val"), load("dev")
tok_train = [tokenize(t) for t in train["text"]]
vocab = {"<PAD>": 0, "<OOV>": 1}
for w, _ in Counter(w for ts in tok_train for w in ts).most_common():
    vocab[w] = len(vocab)
MAX_LEN = int(np.clip(np.percentile([len(t) for t in tok_train], 95), 10, 60))
print(f"단어장 {len(vocab)} / 최대 길이 {MAX_LEN} / 시드 {SEED}")


def encode(texts):
    """정방향·역방향 입력을 모두 앞쪽 빈칸으로 맞춤"""
    fwd = np.zeros((len(texts), MAX_LEN), dtype="int32")
    bwd = np.zeros((len(texts), MAX_LEN), dtype="int32")
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN]
        if ids:
            fwd[i, -len(ids):] = ids
            bwd[i, -len(ids):] = ids[::-1]
    return [fwd, bwd]


def ys(df):
    return {"type": np.array([TYPES.index(v) for v in df["type"]]),
            "urgency": np.array([URGS.index(v) for v in df["urgency"]])}


in_fwd = keras.Input(shape=(MAX_LEN,), dtype="int32", name="forward")
in_bwd = keras.Input(shape=(MAX_LEN,), dtype="int32", name="backward")
emb = layers.Embedding(len(vocab), EMBED_DIM)            # 두 방향이 같은 단어 표현을 공유
h_fwd = layers.LSTM(LSTM_UNITS, name="lstm_forward")(emb(in_fwd))
h_bwd = layers.LSTM(LSTM_UNITS, name="lstm_backward")(emb(in_bwd))
x = layers.Concatenate()([h_fwd, h_bwd])                 # = BiLSTM의 양방향 결합
x = layers.Dropout(DROPOUT)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(DROPOUT)(x)
model = keras.Model([in_fwd, in_bwd], [layers.Dense(len(TYPES), activation="softmax", name="type")(x),
                                       layers.Dense(len(URGS), activation="softmax", name="urgency")(x)])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss={"type": "sparse_categorical_crossentropy", "urgency": "sparse_categorical_crossentropy"},
              metrics={"type": ["accuracy"], "urgency": ["accuracy"]})
model.fit(encode(train["text"]), ys(train), validation_data=(encode(val["text"]), ys(val)),
          epochs=EPOCHS, batch_size=BATCH, verbose=2,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])

model.save(f"{OUT}/bilstm.keras")
with open(f"{OUT}/meta.json", "w", encoding="utf-8") as f:
    json.dump({"vocab": vocab, "max_len": MAX_LEN, "seed": SEED, "input": "two_way_prepad"}, f, ensure_ascii=False)

for name, df in [("val (Claude)", val), ("dev (GPT+Gemini)", dev)]:
    report(f"v5 BiLSTM — {name}", df, *to_labels(*model.predict(encode(df["text"]), verbose=0)))
print(f"\n저장: {OUT}/")