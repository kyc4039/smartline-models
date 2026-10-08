"""v5 KLUE-RoBERTa 미세조정: 담당 분야와 조치 시점 동시 예측
에포크 선택: dev에서 선택 규칙(common.select_key)으로 가장 좋은 에포크
"""
import os, json, time, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from common import load, report, metrics, select_key, fmt, to_labels, TYPES, URGS, MODELS

MODEL_NAME = "klue/roberta-base"     # 너무 느리면 "klue/roberta-small"
OUT = f"{MODELS}/v5_klue"
MAX_LEN, BATCH, EPOCHS, LR = 64, 16, 6, 2e-5
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT, exist_ok=True)
print(f"장치 {device} / 모델 {MODEL_NAME} / 시드 {SEED}")

train, val, dev = load("train"), load("val"), load("dev")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


class MultiHead(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.type_head, self.urg_head = nn.Linear(h, len(TYPES)), nn.Linear(h, len(URGS))

    def forward(self, ids, mask):
        cls = self.drop(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return self.type_head(cls), self.urg_head(cls)


def collate(rows):
    texts, yt, yu = zip(*rows)
    enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(yt), torch.tensor(yu)


rows = list(zip(train["text"], [TYPES.index(v) for v in train["type"]], [URGS.index(v) for v in train["urgency"]]))
loader = DataLoader(rows, batch_size=BATCH, shuffle=True, collate_fn=collate)
model = MultiHead(MODEL_NAME).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = len(loader) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def proba(df):
    model.eval()
    pt, pu, texts = [], [], df["text"].tolist()
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        a, b = model(enc["input_ids"], enc["attention_mask"])
        pt.append(a.softmax(-1).cpu().numpy()); pu.append(b.softmax(-1).cpu().numpy())
    return np.concatenate(pt), np.concatenate(pu)


best_key, best_ep = None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); t0, total = time.time(), 0.0
    for ids, mask, yt, yu in loader:
        ids, mask, yt, yu = ids.to(device), mask.to(device), yt.to(device), yu.to(device)
        a, b = model(ids, mask)
        loss = loss_fn(a, yt) + loss_fn(b, yu)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); total += loss.item()
    mv = metrics(val, *to_labels(*proba(val)))
    md = metrics(dev, *to_labels(*proba(dev)))
    print(f"에포크 {ep} 손실 {total / len(loader):.3f} ({time.time() - t0:.0f}초)")
    print(f"  val {fmt(mv)}\n  dev {fmt(md)}")
    if best_key is None or select_key(md) > best_key:
        best_key, best_ep = select_key(md), ep
        torch.save(model.state_dict(), f"{OUT}/model.pt")

print(f"\n선택된 에포크: {best_ep}")
model.load_state_dict(torch.load(f"{OUT}/model.pt", map_location=device))
tok.save_pretrained(OUT)
with open(f"{OUT}/meta.json", "w", encoding="utf-8") as f:
    json.dump({"model_name": MODEL_NAME, "max_len": MAX_LEN, "best_epoch": best_ep, "seed": SEED}, f, ensure_ascii=False)
for name, df in [("val (Claude)", val), ("dev (GPT+Gemini)", dev)]:
    report(f"v5 KLUE — {name}", df, *to_labels(*proba(df)))
print(f"\n저장: {OUT}/")