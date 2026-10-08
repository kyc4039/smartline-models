"""실험: 사전학습 한국어 문장 임베딩 + 로지스틱 회귀
1) 문장을 ko-sroberta 임베딩(768차원 벡터)으로 변환
2) 유형·긴급도 로지스틱 회귀 학습 (C는 test_llm으로 선택)
3) 단독 성능 + 기존 최종 앙상블과 섞었을 때 성능 비교

데이터 역할 (최종 평가 이후)
  val        : 참고용 (학습 데이터와 같은 Claude 문장)
  test_llm   : 모델·설정 선택용
  test_human : 개선 확인용 (이미 한 번 본 데이터라 '최종 점수'로 쓰지 않음)
"""
import json
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from text_utils import tokenize, load, report, TYPES, URGS

EMB_MODEL = "jhgan/ko-sroberta-multitask"
DATA_DIR = "data"
MODEL_DIR = "models_v4"          # 기존 기준 모델·BiLSTM (최종 앙상블 재료)
OUT_PATH = "models_emb/embedding_lr.joblib"
C_GRID = [0.5, 1, 2, 5, 10]
LSTM_WEIGHT = 0.25               # 기존 최종 앙상블 안에서 BiLSTM 비중
EMB_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # (기존 앙상블 : 임베딩 모델) 중 임베딩 비중


# ---------- 1. 임베딩 ----------
def get_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    return lambda texts: model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                      show_progress_bar=False)


embed = get_embedder()
train, val = load(f"{DATA_DIR}/train.csv"), load(f"{DATA_DIR}/val.csv")
test_llm, test_human = load(f"{DATA_DIR}/test_llm.csv"), load(f"{DATA_DIR}/test_human.csv")
human_user = test_human[test_human["source"] == "user"].reset_index(drop=True)
sets = {"val": val, "test_llm": test_llm, "human(전체)": test_human, "human(사용자)": human_user}

t0 = time.time()
X_train = embed(train["text"])
X = {k: embed(d["text"]) for k, d in sets.items()}
print(f"임베딩 완료: {X_train.shape[1]}차원, {time.time() - t0:.1f}초")


# ---------- 2. 로지스틱 회귀 (C 선택: test_llm 긴급도 정확도) ----------
def fit(C):
    mk = lambda: LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    return {"type": mk().fit(X_train, train["type"]), "urgency": mk().fit(X_train, train["urgency"])}


def proba(m, Xs, labels):
    p = m.predict_proba(Xs)
    return p[:, [list(m.classes_).index(l) for l in labels]]


print("\n[C 선택] test_llm 기준")
best_C, best_acc = None, -1
for C in C_GRID:
    m = fit(C)
    acc = accuracy_score(test_llm["urgency"], m["urgency"].predict(X["test_llm"]))
    print(f"  C={C:<4} 긴급도 {acc:.3f}")
    if acc > best_acc:
        best_C, best_acc = C, acc
emb = fit(best_C)
print(f"→ 선택된 C = {best_C}")

import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
joblib.dump({"model_name": EMB_MODEL, "C": best_C, **emb}, OUT_PATH)
print(f"저장: {OUT_PATH}")


# ---------- 3. 기존 최종 앙상블 확률 ----------
from tensorflow import keras
base = joblib.load(f"{MODEL_DIR}/baseline.joblib")
lstm = keras.models.load_model(f"{MODEL_DIR}/bilstm.keras")
with open(f"{MODEL_DIR}/bilstm_meta.json", encoding="utf-8") as f:
    meta = json.load(f)


def encode(texts):
    seqs = np.zeros((len(texts), meta["max_len"]), dtype="int32")
    for i, t in enumerate(texts):
        ids = [meta["vocab"].get(tok, 1) for tok in tokenize(t)][:meta["max_len"]]
        seqs[i, :len(ids)] = ids
    return seqs


def old_ensemble(df):
    bt, bu = proba(base["type"], df["text"], TYPES), proba(base["urgency"], df["text"], URGS)
    lt, lu = lstm.predict(encode(df["text"]), verbose=0)
    w = LSTM_WEIGHT
    return (1 - w) * bt + w * lt, (1 - w) * bu + w * lu


probs = {}
for k, d in sets.items():
    ot, ou = old_ensemble(d)
    et, eu = proba(emb["type"], X[k], TYPES), proba(emb["urgency"], X[k], URGS)
    probs[k] = (ot, ou, et, eu)


def combine(k, e):
    ot, ou, et, eu = probs[k]
    pt, pu = (1 - e) * ot + e * et, (1 - e) * ou + e * eu
    return (np.array([TYPES[i] for i in pt.argmax(1)]), np.array([URGS[i] for i in pu.argmax(1)]))


def row(df, pt, pu):
    cm = confusion_matrix(df["urgency"], pu, labels=URGS)
    return (accuracy_score(df["type"], pt), accuracy_score(df["urgency"], pu),
            recall_score(df["urgency"], pu, labels=["상"], average="macro", zero_division=0),
            int(cm[1:, 0].sum()))


# ---------- 4. 결과표 ----------
print("\n임베딩 비중 0 = 기존 최종 앙상블 / 1 = 임베딩 모델 단독")
print("각 칸: 유형 / 긴급도 / 상 탐지 / 오경보")
header = f"{'임베딩 비중':>10} | " + " | ".join(f"{k:^27}" for k in sets)
print(header)
for e in EMB_WEIGHTS:
    cells = []
    for k, d in sets.items():
        a, b, c, n = row(d, *combine(k, e))
        cells.append(f"{a:.3f} {b:.3f} {c:.3f} {n:>3}")
    print(f"{e:>10.2f} | " + " | ".join(f"{c:^27}" for c in cells))

# test_llm 긴급도 기준으로 비중 선택 후 상세 보고
best_e = max(EMB_WEIGHTS, key=lambda e: (row(test_llm, *combine("test_llm", e))[1],
                                         row(test_llm, *combine("test_llm", e))[2]))
print(f"\ntest_llm 기준으로 고른 임베딩 비중: {best_e}")
pt, pu = combine("human(사용자)", best_e)
report(f"임베딩 비중 {best_e} — human(사용자)", human_user, pt, pu)
wrong = (human_user["type"].values != pt) | (human_user["urgency"].values != pu)
print(f"\n[틀린 문장 {wrong.sum()}개]")
print(human_user[wrong].assign(pred_type=pt[wrong], pred_urg=pu[wrong])
      [["text", "type", "pred_type", "urgency", "pred_urg"]].to_string())