"""최종 평가: 사람 테스트 데이터(test_human.csv)로 단 한 번 평가한다.
- 최종 모델: 앙상블 (기준 모델 0.75 + BiLSTM 0.25)
- 참고: 기준 모델 단독, BiLSTM 단독
- 전체 / 사용자 작성 문장(source=user)만 두 가지로 보고
- 최종 시스템 점수에는 위험 단어 규칙까지 포함
"""
import json
import joblib
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from text_utils import tokenize, load, report, TYPES, URGS
from predict import DANGER_WORDS

DATA_PATH = "data/test_human.csv"
MODEL_DIR = "models"
LSTM_WEIGHT = 0.25

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


def base_proba(m, texts, labels):
    p = m.predict_proba(texts)
    return p[:, [list(m.classes_).index(l) for l in labels]]


def predict(df, w, rule=False):
    bt = base_proba(base["type"], df["text"], TYPES)
    bu = base_proba(base["urgency"], df["text"], URGS)
    lt, lu = lstm.predict(encode(df["text"]), verbose=0)
    pt, pu = (1 - w) * bt + w * lt, (1 - w) * bu + w * lu
    p_type = np.array([TYPES[i] for i in pt.argmax(1)])
    p_urg = np.array([URGS[i] for i in pu.argmax(1)])
    if rule:
        for i, t in enumerate(df["text"]):
            if p_urg[i] != "상" and any(d in t for d in DANGER_WORDS):
                p_urg[i] = "상"
    return p_type, p_urg


def scores(df, p_type, p_urg):
    cm = confusion_matrix(df["urgency"], p_urg, labels=URGS)
    return (accuracy_score(df["type"], p_type), accuracy_score(df["urgency"], p_urg),
            recall_score(df["urgency"], p_urg, labels=["상"], average="macro", zero_division=0),
            int(cm[1:, 0].sum()))   # 오경보: 상이 아닌데 상으로 판단


test = load(DATA_PATH)
subsets = {"전체": test}
if "source" in test.columns:
    subsets["사용자 작성만"] = test[test["source"] == "user"].reset_index(drop=True)

configs = [("최종: 앙상블 + 위험단어 규칙", LSTM_WEIGHT, True),
           ("앙상블 (모델만)", LSTM_WEIGHT, False),
           ("기준 모델 단독", 0.0, False),
           ("BiLSTM 단독", 1.0, False)]

print(f"\n{'모델':<24}{'데이터':<12}{'문장':>5}{'유형':>8}{'긴급도':>8}{'상재현':>8}{'오경보':>7}")
for name, w, rule in configs:
    for sname, sdf in subsets.items():
        a, b, c, d = scores(sdf, *predict(sdf, w, rule))
        print(f"{name:<24}{sname:<12}{len(sdf):>5}{a:>8.3f}{b:>8.3f}{c:>8.3f}{d:>7}")

for sname, sdf in subsets.items():
    p_type, p_urg = predict(sdf, LSTM_WEIGHT, rule=True)
    report(f"최종 시스템 상세 — {sname}", sdf, p_type, p_urg)
    wrong = (sdf["type"].values != p_type) | (sdf["urgency"].values != p_urg)
    print(f"\n[틀린 문장 {wrong.sum()}개]")
    print(sdf[wrong].assign(pred_type=p_type[wrong], pred_urg=p_urg[wrong])
          [["text", "type", "pred_type", "urgency", "pred_urg"]].to_string())