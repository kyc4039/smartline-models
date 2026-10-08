"""v5 공통 도구: 라벨, 형태소 분석, 데이터 로드, 평가 지표, 선택 규칙"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

TYPES = ["기계", "전기", "유압공압", "센서"]
URGS = ["상", "중", "하"]
URG_NAME = {"상": "즉시", "중": "당일", "하": "정기"}
TYPE_NAME = {"기계": "기계 정비", "전기": "전기·제어", "유압공압": "유공압", "센서": "센서·계측"}
DATA, MODELS = "data", "models"

# 선택 규칙 (결과를 보기 전에 확정, 2026-10-01)
MIN_URGENT_RECALL = 0.9      # 즉시 조치 탐지율 0.9 이상만 후보

_kiwi = None


def tokenize(text):
    """형태소 단위로 자르기 (문장부호만 제거)"""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return [t.form for t in _kiwi.tokenize(str(text))
            if not (t.tag.startswith("S") and t.tag not in ("SL", "SN", "SH"))]


def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv", encoding="utf-8-sig")
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["type"].isin(TYPES) & df["urgency"].isin(URGS)].reset_index(drop=True)


def to_labels(p_type, p_urg):
    return (np.array([TYPES[i] for i in p_type.argmax(1)]),
            np.array([URGS[i] for i in p_urg.argmax(1)]))


def metrics(df, p_type, p_urg):
    y = df["urgency"].values
    cm = confusion_matrix(y, p_urg, labels=URGS)
    return {
        "n": len(df),
        "type_acc": accuracy_score(df["type"], p_type),
        "urg_acc": accuracy_score(y, p_urg),
        "both_acc": float(np.mean((df["type"].values == p_type) & (y == p_urg))),
        "urgent_recall": recall_score(y, p_urg, labels=["상"], average="macro", zero_division=0),
        "false_alarm": int(cm[1:, 0].sum()),   # 즉시가 아닌데 즉시로 판단
    }


def select_key(m):
    """선택 규칙: 1) 즉시 탐지율 기준 충족 2) 조치 시점 정확도 3) 오경보 적음 4) 담당 정확도"""
    return (m["urgent_recall"] >= MIN_URGENT_RECALL, round(m["urg_acc"], 6), -m["false_alarm"],
            round(m["type_acc"], 6))


def tone_subsets(df):
    """말투와 심각도가 엇갈린 문장 (지름길에 속는지 확인)"""
    if "style" not in df.columns:
        return {}
    return {"과장된 정기 점검": (df["urgency"] == "하") & (df["style"] == "과장"),
            "담담한 즉시 조치": (df["urgency"] == "상") & (df["style"] == "담담")}


def fmt(m):
    return (f"담당 {m['type_acc']:.3f} | 조치 {m['urg_acc']:.3f} | 둘 다 {m['both_acc']:.3f} | "
            f"즉시 탐지 {m['urgent_recall']:.3f} | 오경보 {m['false_alarm']}")


def report(title, df, p_type, p_urg):
    m = metrics(df, p_type, p_urg)
    print(f"\n===== {title} ({m['n']}문장) =====")
    print(fmt(m))
    print("조치 시점 혼동행렬 (행=정답, 열=예측, 즉시/당일/정기)")
    print(confusion_matrix(df["urgency"], p_urg, labels=URGS))
    for name, mask in tone_subsets(df).items():
        if mask.sum():
            acc = float(np.mean(df["urgency"].values[mask] == p_urg[mask]))
            print(f"  {name}: {acc:.3f} ({int(mask.sum())}문장)")
    return m