"""고도화 실험 3: '상' 판정 기준값 조정 (재학습 없음)
판정 방식: '상' 확률이 기준값 이상이면 '상', 아니면 '중'과 '하' 중 큰 쪽
  기준값 None = 기존 방식(세 확률 중 가장 큰 것)

선택 규칙 (실행 전에 고정): test_llm에서
  1) 긴급도 정확도가 가장 높고  2) 같으면 오경보가 적고  3) 그래도 같으면 상 탐지가 높은 조합
"""
import json
import joblib
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from text_utils import tokenize, load, report, TYPES, URGS

OLD_DIR, KLUE_DIR = "models_v4", "models_klue"
LSTM_WEIGHT = 0.25
KLUE_WEIGHTS = [0.0, 0.25, 0.5, 1.0]          # 0 = 현재 최종 앙상블
THRESHOLDS = [None, 0.4, 0.5, 0.6, 0.7, 0.8]

# ---------- 데이터 ----------
test_llm, test_human = load("data/test_llm.csv"), load("data/test_human.csv")
human_user = test_human[test_human["source"] == "user"].reset_index(drop=True)
sets = {"test_llm": test_llm, "human(사용자)": human_user}

# ---------- 기존 앙상블 ----------
base = joblib.load(f"{OLD_DIR}/baseline.joblib")
lstm = keras.models.load_model(f"{OLD_DIR}/bilstm.keras")
with open(f"{OLD_DIR}/bilstm_meta.json", encoding="utf-8") as f:
    bmeta = json.load(f)


def encode(texts):
    seqs = np.zeros((len(texts), bmeta["max_len"]), dtype="int32")
    for i, t in enumerate(texts):
        ids = [bmeta["vocab"].get(tk, 1) for tk in tokenize(t)][:bmeta["max_len"]]
        seqs[i, :len(ids)] = ids
    return seqs


def base_proba(m, texts, labels):
    p = m.predict_proba(texts)
    return p[:, [list(m.classes_).index(l) for l in labels]]


def old_ensemble(df):
    bt, bu = base_proba(base["type"], df["text"], TYPES), base_proba(base["urgency"], df["text"], URGS)
    lt, lu = lstm.predict(encode(df["text"]), verbose=0)
    w = LSTM_WEIGHT
    return (1 - w) * bt + w * lt, (1 - w) * bu + w * lu


# ---------- KLUE (train_klue.py와 같은 구조) ----------
with open(f"{KLUE_DIR}/meta.json", encoding="utf-8") as f:
    kmeta = json.load(f)


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(kmeta["model_name"])
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.type_head = nn.Linear(hidden, len(TYPES))
        self.urg_head = nn.Linear(hidden, len(URGS))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0])
        return self.type_head(cls), self.urg_head(cls)


tok = AutoTokenizer.from_pretrained(KLUE_DIR)
klue = MultiHead()
klue.load_state_dict(torch.load(f"{KLUE_DIR}/model.pt", map_location="cpu"))
klue.eval()


@torch.no_grad()
def klue_proba(df):
    pt, pu = [], []
    texts = df["text"].tolist()
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=kmeta["max_len"],
                  return_tensors="pt")
        lt, lu = klue(enc["input_ids"], enc["attention_mask"])
        pt.append(lt.softmax(-1).numpy())
        pu.append(lu.softmax(-1).numpy())
    return np.concatenate(pt), np.concatenate(pu)


probs = {k: (*old_ensemble(d), *klue_proba(d)) for k, d in sets.items()}


# ---------- 판정 ----------
def decide(k, w, th):
    ot, ou, kt, ku = probs[k]
    pt, pu = (1 - w) * ot + w * kt, (1 - w) * ou + w * ku
    p_type = np.array([TYPES[i] for i in pt.argmax(1)])
    if th is None:
        p_urg = np.array([URGS[i] for i in pu.argmax(1)])
    else:   # URGS = [상, 중, 하]
        p_urg = np.where(pu[:, 0] >= th, "상", np.where(pu[:, 1] >= pu[:, 2], "중", "하"))
    return p_type, p_urg


def row(df, p_type, p_urg):
    cm = confusion_matrix(df["urgency"], p_urg, labels=URGS)
    return (accuracy_score(df["type"], p_type), accuracy_score(df["urgency"], p_urg),
            recall_score(df["urgency"], p_urg, labels=["상"], average="macro", zero_division=0),
            int(cm[1:, 0].sum()))


print("\n각 칸: 유형 / 긴급도 / 상 탐지 / 오경보   (KLUE 비중 0 = 현재 최종 앙상블)")
print(f"{'KLUE':>5} {'기준값':>6} | {'test_llm':^27} | {'human(사용자)':^27}")
results = []
for w in KLUE_WEIGHTS:
    for th in THRESHOLDS:
        r = {k: row(d, *decide(k, w, th)) for k, d in sets.items()}
        results.append((w, th, r))
        cells = [f"{a:.3f} {b:.3f} {c:.3f} {n:>3}" for a, b, c, n in r.values()]
        print(f"{w:>5.2f} {str(th):>6} | {cells[0]:^27} | {cells[1]:^27}")
    print("-" * 70)

# test_llm 기준 선택: 긴급도 ↑, 오경보 ↓, 상 탐지 ↑
best_w, best_th, _ = max(results, key=lambda x: (x[2]["test_llm"][1], -x[2]["test_llm"][3], x[2]["test_llm"][2]))
print(f"\ntest_llm 기준 선택: KLUE 비중 {best_w}, 기준값 {best_th}")

p_type, p_urg = decide("human(사용자)", best_w, best_th)
report(f"KLUE {best_w} / 기준값 {best_th} — human(사용자)", human_user, p_type, p_urg)
wrong = (human_user["type"].values != p_type) | (human_user["urgency"].values != p_urg)
print(f"\n[틀린 문장 {wrong.sum()}개]")
print(human_user[wrong].assign(pred_type=p_type[wrong], pred_urg=p_urg[wrong])
      [["text", "type", "pred_type", "urgency", "pred_urg"]].to_string())