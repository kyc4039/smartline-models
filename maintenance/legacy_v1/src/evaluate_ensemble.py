"""기준 모델 + BiLSTM 앙상블 평가
두 모델의 확률을 가중 평균해서, 어떤 비율이 가장 좋은지 확인한다.
가중치는 반드시 검증 데이터(val)로 고르고, test_llm은 확인용으로만 본다.
"""
import json
import joblib
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score
from text_utils import tokenize, load, report, TYPES, URGS

DATA_DIR = "data"
MODEL_DIR = "models"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # BiLSTM 비중 (0 = 기준 모델만, 1 = BiLSTM만)

# ---------- 모델 불러오기 ----------
base = joblib.load(f"{MODEL_DIR}/baseline.joblib")
lstm = keras.models.load_model(f"{MODEL_DIR}/bilstm.keras")
with open(f"{MODEL_DIR}/bilstm_meta.json", encoding="utf-8") as f:
    meta = json.load(f)
vocab, max_len = meta["vocab"], meta["max_len"]


def encode(texts):
    seqs = np.zeros((len(texts), max_len), dtype="int32")
    for i, text in enumerate(texts):
        ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
        seqs[i, :len(ids)] = ids
    return seqs


def base_proba(model, texts, labels):
    """기준 모델 확률을 TYPES/URGS 순서에 맞춰 재정렬"""
    p = model.predict_proba(texts)
    order = [list(model.classes_).index(l) for l in labels]
    return p[:, order]


def all_probs(df):
    bt = base_proba(base["type"], df["text"], TYPES)
    bu = base_proba(base["urgency"], df["text"], URGS)
    lt, lu = lstm.predict(encode(df["text"]), verbose=0)
    return bt, bu, lt, lu


def combine(probs, w):
    bt, bu, lt, lu = probs
    pt = (1 - w) * bt + w * lt
    pu = (1 - w) * bu + w * lu
    return (np.array([TYPES[i] for i in pt.argmax(1)]),
            np.array([URGS[i] for i in pu.argmax(1)]))


val = load(f"{DATA_DIR}/val.csv")
test = load(f"{DATA_DIR}/test_llm.csv")
val_probs, test_probs = all_probs(val), all_probs(test)

print(f"{'BiLSTM 비중':>10} | {'val 유형':>8} {'val 긴급도':>9} {'val 상재현':>9} | "
      f"{'test 유형':>9} {'test 긴급도':>10} {'test 상재현':>10}")
best_w, best_score = None, -1
for w in WEIGHTS:
    row = []
    for df, probs in [(val, val_probs), (test, test_probs)]:
        pt, pu = combine(probs, w)
        row += [accuracy_score(df["type"], pt), accuracy_score(df["urgency"], pu),
                recall_score(df["urgency"], pu, labels=["상"], average="macro", zero_division=0)]
    print(f"{w:>10.2f} | {row[0]:>8.3f} {row[1]:>9.3f} {row[2]:>9.3f} | "
          f"{row[3]:>9.3f} {row[4]:>10.3f} {row[5]:>10.3f}")
    val_score = row[1] + row[2]          # 검증 긴급도 정확도 + '상' 재현율
    if val_score > best_score:
        best_w, best_score = w, val_score

print(f"\n검증 데이터 기준으로 고른 BiLSTM 비중: {best_w}")
report(f"앙상블 (BiLSTM 비중 {best_w}) — test_llm", test, *combine(test_probs, best_w))
