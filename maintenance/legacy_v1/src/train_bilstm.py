"""메인 모델: BiLSTM 하나로 고장 유형과 긴급도를 동시에 예측 (출력층 2개)"""
import os, json
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from text_utils import tokenize, load, report, TYPES, URGS

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(42)
np.random.seed(42)

# ---------- 하이퍼파라미터 ----------
EMBED_DIM = 128
LSTM_UNITS = 64
DROPOUT = 0.3
EPOCHS = 30
BATCH_SIZE = 32
MIN_COUNT = 1       # 학습 데이터에서 이 횟수 이상 나온 형태소만 단어장에 포함

# ---------- 데이터 ----------
train = load(f"{DATA_DIR}/train.csv")
val = load(f"{DATA_DIR}/val.csv")
test_llm = load(f"{DATA_DIR}/test_llm.csv")

train_tokens = [tokenize(t) for t in train["text"]]

# 단어장: 0 = 패딩, 1 = 모르는 단어(OOV)
counter = Counter(tok for toks in train_tokens for tok in toks)
vocab = {"<PAD>": 0, "<OOV>": 1}
for tok, cnt in counter.most_common():
    if cnt >= MIN_COUNT:
        vocab[tok] = len(vocab)

# 문장 길이: 학습 문장의 95%가 잘리지 않는 길이 (최소 10, 최대 60)
MAX_LEN = int(np.clip(np.percentile([len(t) for t in train_tokens], 95), 10, 60))
print(f"단어장 크기: {len(vocab)} / 최대 길이: {MAX_LEN}")


def encode(texts):
    seqs = np.zeros((len(texts), MAX_LEN), dtype="int32")
    for i, text in enumerate(texts):
        ids = [vocab.get(tok, 1) for tok in tokenize(text)][:MAX_LEN]
        seqs[i, :len(ids)] = ids
    return seqs


def labels(df):
    return {"type": np.array([TYPES.index(v) for v in df["type"]]),
            "urgency": np.array([URGS.index(v) for v in df["urgency"]])}


X_train, y_train = encode(train["text"]), labels(train)
X_val, y_val = encode(val["text"]), labels(val)

# ---------- 모델 ----------
inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
x = layers.Embedding(len(vocab), EMBED_DIM, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(LSTM_UNITS))(x)
x = layers.Dropout(DROPOUT)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(DROPOUT)(x)
out_type = layers.Dense(len(TYPES), activation="softmax", name="type")(x)
out_urg = layers.Dense(len(URGS), activation="softmax", name="urgency")(x)
model = keras.Model(inputs, [out_type, out_urg])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={"type": "sparse_categorical_crossentropy", "urgency": "sparse_categorical_crossentropy"},
    metrics={"type": ["accuracy"], "urgency": ["accuracy"]},
)
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=2)

# ---------- 평가 ----------
def predict_labels(df):
    p_type, p_urg = model.predict(encode(df["text"]), verbose=0)
    return (np.array([TYPES[i] for i in p_type.argmax(1)]),
            np.array([URGS[i] for i in p_urg.argmax(1)]))

for name, df in [("검증 데이터 (val)", val), ("다른 LLM 테스트 (test_llm)", test_llm)]:
    if df is not None:
        p_t, p_u = predict_labels(df)
        report(name, df, p_t, p_u)
        mask = df["urgency"].values != p_u
        print(f"\n[긴급도 틀린 문장 {mask.sum()}개]")
        print(df[mask].assign(pred=p_u[mask])[["text", "urgency", "pred"]].to_string())

# ---------- 저장 ----------
model.save(f"{MODEL_DIR}/bilstm.keras")
with open(f"{MODEL_DIR}/bilstm_meta.json", "w", encoding="utf-8") as f:
    json.dump({"vocab": vocab, "max_len": MAX_LEN, "types": TYPES, "urgs": URGS}, f, ensure_ascii=False)
with open(f"{MODEL_DIR}/bilstm_history.json", "w", encoding="utf-8") as f:
    json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)
print(f"\n모델 저장 완료: {MODEL_DIR}/bilstm.keras, bilstm_meta.json")

p_t, p_u = predict_labels(df)
wrong = df[(df["urgency"] != p_u)].assign(pred=p_u[df["urgency"] != p_u])
print(wrong[["text", "urgency", "pred"]].to_string())