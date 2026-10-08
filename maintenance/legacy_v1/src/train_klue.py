"""고도화 실험 2: KLUE-RoBERTa 미세조정
하나의 RoBERTa가 문장을 읽고, 출력층 두 개로 고장 유형과 긴급도를 동시에 예측한다.
(BiLSTM 구조와 같지만, 대량의 한국어로 미리 학습된 모델에서 출발한다)

데이터 역할
  train      : 학습
  test_llm   : 에포크·설정 선택
  val        : 참고용 (학습 데이터와 같은 Claude 문장)
  test_human : 개선 확인용 (이미 본 데이터, 최종 점수로 쓰지 않음)
"""
import os, json, time, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from text_utils import tokenize, load, report, TYPES, URGS

MODEL_NAME = "klue/roberta-base"   # 너무 느리면 "klue/roberta-small"
OUT_DIR = "models_klue"
OLD_DIR = "models_v4"              # 기존 최종 앙상블 (비교·혼합용)
MAX_LEN = 64
BATCH = 16
EPOCHS = 6
LR = 2e-5
SEED = int(os.environ.get("SEED", 42))
LSTM_WEIGHT = 0.25
MIX_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # (기존 앙상블 : KLUE) 중 KLUE 비중

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"장치: {device} / 모델: {MODEL_NAME} / 시드: {SEED}")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 데이터 ----------
train, val = load("data/train.csv"), load("data/val.csv")
test_llm, test_human = load("data/test_llm.csv"), load("data/test_human.csv")
human_user = test_human[test_human["source"] == "user"].reset_index(drop=True)
sets = {"val": val, "test_llm": test_llm, "human(전체)": test_human, "human(사용자)": human_user}

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def collate(rows):
    texts, y_type, y_urg = zip(*rows)
    enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(y_type), torch.tensor(y_urg)


train_rows = list(zip(train["text"], [TYPES.index(v) for v in train["type"]],
                      [URGS.index(v) for v in train["urgency"]]))
loader = DataLoader(train_rows, batch_size=BATCH, shuffle=True, collate_fn=collate)


# ---------- 모델 ----------
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.type_head = nn.Linear(hidden, len(TYPES))
        self.urg_head = nn.Linear(hidden, len(URGS))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0])      # 문장 대표 벡터([CLS])
        return self.type_head(cls), self.urg_head(cls)


model = MultiHead().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def klue_proba(df):
    model.eval()
    pt, pu = [], []
    texts = df["text"].tolist()
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        lt, lu = model(enc["input_ids"], enc["attention_mask"])
        pt.append(lt.softmax(-1).cpu().numpy())
        pu.append(lu.softmax(-1).cpu().numpy())
    return np.concatenate(pt), np.concatenate(pu)


def to_labels(pt, pu):
    return np.array([TYPES[i] for i in pt.argmax(1)]), np.array([URGS[i] for i in pu.argmax(1)])


def row(df, p_type, p_urg):
    cm = confusion_matrix(df["urgency"], p_urg, labels=URGS)
    return (accuracy_score(df["type"], p_type), accuracy_score(df["urgency"], p_urg),
            recall_score(df["urgency"], p_urg, labels=["상"], average="macro", zero_division=0),
            int(cm[1:, 0].sum()))


# ---------- 학습 (에포크 선택: test_llm 긴급도 → 상 탐지) ----------
best_key, best_ep = (-1, -1), 0
for ep in range(1, EPOCHS + 1):
    model.train()
    t0, total = time.time(), 0.0
    for ids, mask, y_type, y_urg in loader:
        ids, mask, y_type, y_urg = ids.to(device), mask.to(device), y_type.to(device), y_urg.to(device)
        lt, lu = model(ids, mask)
        loss = loss_fn(lt, y_type) + loss_fn(lu, y_urg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()

    v = row(val, *to_labels(*klue_proba(val)))
    t = row(test_llm, *to_labels(*klue_proba(test_llm)))
    print(f"에포크 {ep}  손실 {total / len(loader):.3f}  ({time.time() - t0:.0f}초) | "
          f"val 유형 {v[0]:.3f} 긴급도 {v[1]:.3f} | test_llm 유형 {t[0]:.3f} 긴급도 {t[1]:.3f} 상탐지 {t[2]:.3f}")
    if (t[1], t[2]) > best_key:
        best_key, best_ep = (t[1], t[2]), ep
        torch.save(model.state_dict(), f"{OUT_DIR}/model.pt")

print(f"\n선택된 에포크: {best_ep} (test_llm 긴급도 {best_key[0]:.3f})")
model.load_state_dict(torch.load(f"{OUT_DIR}/model.pt", map_location=device))
tok.save_pretrained(OUT_DIR)
with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
    json.dump({"model_name": MODEL_NAME, "max_len": MAX_LEN, "best_epoch": best_ep, "seed": SEED,
               "types": TYPES, "urgs": URGS}, f, ensure_ascii=False)

# ---------- 기존 최종 앙상블과 혼합 ----------
import joblib
from tensorflow import keras
base = joblib.load(f"{OLD_DIR}/baseline.joblib")
lstm = keras.models.load_model(f"{OLD_DIR}/bilstm.keras")
with open(f"{OLD_DIR}/bilstm_meta.json", encoding="utf-8") as f:
    meta = json.load(f)


def encode(texts):
    seqs = np.zeros((len(texts), meta["max_len"]), dtype="int32")
    for i, t in enumerate(texts):
        ids = [meta["vocab"].get(tk, 1) for tk in tokenize(t)][:meta["max_len"]]
        seqs[i, :len(ids)] = ids
    return seqs


def base_proba(m, texts, labels):
    p = m.predict_proba(texts)
    return p[:, [list(m.classes_).index(l) for l in labels]]


probs = {}
for k, d in sets.items():
    bt, bu = base_proba(base["type"], d["text"], TYPES), base_proba(base["urgency"], d["text"], URGS)
    lt, lu = lstm.predict(encode(d["text"]), verbose=0)
    ot, ou = (1 - LSTM_WEIGHT) * bt + LSTM_WEIGHT * lt, (1 - LSTM_WEIGHT) * bu + LSTM_WEIGHT * lu
    probs[k] = (ot, ou, *klue_proba(d))


def combine(k, w):
    ot, ou, kt, ku = probs[k]
    return to_labels((1 - w) * ot + w * kt, (1 - w) * ou + w * ku)


print("\nKLUE 비중 0 = 기존 최종 앙상블 / 1 = KLUE 단독")
print("각 칸: 유형 / 긴급도 / 상 탐지 / 오경보")
print(f"{'KLUE 비중':>9} | " + " | ".join(f"{k:^27}" for k in sets))
for w in MIX_WEIGHTS:
    cells = [f"{a:.3f} {b:.3f} {c:.3f} {n:>3}" for a, b, c, n in (row(d, *combine(k, w)) for k, d in sets.items())]
    print(f"{w:>9.2f} | " + " | ".join(f"{c:^27}" for c in cells))

best_w = max(MIX_WEIGHTS, key=lambda w: row(test_llm, *combine("test_llm", w))[1:3])
print(f"\ntest_llm 기준으로 고른 KLUE 비중: {best_w}")
for w in sorted({1.0, best_w}):
    p_type, p_urg = combine("human(사용자)", w)
    report(f"KLUE 비중 {w} — human(사용자)", human_user, p_type, p_urg)
    wrong = (human_user["type"].values != p_type) | (human_user["urgency"].values != p_urg)
    print(f"\n[틀린 문장 {wrong.sum()}개]")
    print(human_user[wrong].assign(pred_type=p_type[wrong], pred_urg=p_urg[wrong])
          [["text", "type", "pred_type", "urgency", "pred_urg"]].to_string())