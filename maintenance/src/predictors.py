"""저장된 모델을 불러와 확률을 내는 함수 모음 (dev 선택·최종 평가에서 공통 사용)
각 로더는 모델이 없으면 None을 돌려준다.
반환 함수: f(df) → (담당 확률 [n,4], 조치 확률 [n,3])  순서는 common.TYPES / URGS
"""
import os, sys, json
import numpy as np
from common import tokenize, TYPES, URGS, MODELS


def _sklearn_proba(m, texts, labels):
    p = m.predict_proba(texts)
    return p[:, [list(m.classes_).index(l) for l in labels]]


def _baseline(path):
    if not os.path.exists(path):
        return None
    import joblib
    m = joblib.load(path)
    return lambda df: (_sklearn_proba(m["type"], df["text"], TYPES), _sklearn_proba(m["urgency"], df["text"], URGS))


def _bilstm(model_path, meta_path):
    if not os.path.exists(model_path):
        return None
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    two_way = meta.get("input") == "two_way_prepad"     # v5 수정판 / 없으면 v1 방식

    def f(df):
        n, L = len(df), meta["max_len"]
        fwd, bwd = np.zeros((n, L), dtype="int32"), np.zeros((n, L), dtype="int32")
        for i, t in enumerate(df["text"]):
            ids = [meta["vocab"].get(w, 1) for w in tokenize(t)][:L]
            if not ids:
                continue
            if two_way:
                fwd[i, -len(ids):] = ids
                bwd[i, -len(ids):] = ids[::-1]
            else:
                fwd[i, :len(ids)] = ids
        pt, pu = model.predict([fwd, bwd] if two_way else fwd, verbose=0)
        return pt, pu
    return f


def load_v5():
    """v5 모델들 {이름: 함수}"""
    out = {}
    b = _baseline(f"{MODELS}/v5_baseline/baseline.joblib")
    if b: out["baseline"] = b
    l = _bilstm(f"{MODELS}/v5_bilstm/bilstm.keras", f"{MODELS}/v5_bilstm/meta.json")
    if l: out["bilstm"] = l
    k = load_v5_klue()
    if k: out["klue"] = k
    return out


def load_v5_klue():
    d = f"{MODELS}/v5_klue"
    if not os.path.exists(f"{d}/model.pt"):
        return None
    import torch
    from torch import nn
    from transformers import AutoTokenizer, AutoModel
    with open(f"{d}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(meta["model_name"])
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.type_head, self.urg_head = nn.Linear(h, len(TYPES)), nn.Linear(h, len(URGS))

        def forward(self, ids, mask):
            cls = self.drop(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
            return self.type_head(cls), self.urg_head(cls)

    tok = AutoTokenizer.from_pretrained(d)
    model = MultiHead()
    model.load_state_dict(torch.load(f"{d}/model.pt", map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def f(df):
        pt, pu, texts = [], [], df["text"].tolist()
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=meta["max_len"], return_tensors="pt")
            a, b = model(enc["input_ids"], enc["attention_mask"])
            pt.append(a.softmax(-1).numpy()); pu.append(b.softmax(-1).numpy())
        return np.concatenate(pt), np.concatenate(pu)
    return f


def load_v1(baseline_only=False):
    """이전 작업(em_model_v1) 최종 모델: 기준 모델 0.75 + BiLSTM 0.25
    baseline_only=True: 기준 모델 단독 (v1 BiLSTM은 마스크 기능이 꺼진 환경에서 불리할 수 있어 공정 비교용)
    """
    d = "legacy_v1/models_v4"
    if not os.path.exists(f"{d}/baseline.joblib"):
        return None
    if "legacy_v1/src" not in sys.path:
        sys.path.insert(0, "legacy_v1/src")  # v1 기준 모델이 text_utils.tokenize를 참조함
    b = _baseline(f"{d}/baseline.joblib")
    if baseline_only:
        return b
    l = _bilstm(f"{d}/bilstm.keras", f"{d}/bilstm_meta.json")

    def f(df):
        bt, bu = b(df)
        lt, lu = l(df)
        return 0.75 * bt + 0.25 * lt, 0.75 * bu + 0.25 * lu
    return f


def combine(probs, weights):
    """probs: {이름: (담당 확률, 조치 확률)}, weights: {이름: 비중}"""
    pt = sum(w * probs[k][0] for k, w in weights.items() if w > 0)
    pu = sum(w * probs[k][1] for k, w in weights.items() if w > 0)
    return pt, pu