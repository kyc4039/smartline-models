"""공통 유틸: 한국어 형태소 토큰화, 데이터 로드, 평가 리포트"""
import os
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

TYPES = ["기계", "전기", "유압공압", "센서"]
URGS = ["상", "중", "하"]

_kiwi = Kiwi()
_KEEP_SYMBOLS = ("SL", "SN", "SH")  # 영어, 숫자, 한자는 유지


def tokenize(text):
    """문장을 형태소 단위로 자른다. 문장부호만 제거하고 조사·어미는 유지한다."""
    tokens = []
    for t in _kiwi.tokenize(str(text)):
        if t.tag.startswith("S") and t.tag not in _KEEP_SYMBOLS:
            continue
        tokens.append(t.form)
    return tokens


def load(path):
    """CSV를 읽고 라벨이 올바른 행만 남긴다. 파일이 없으면 None."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["type"].isin(TYPES) & df["urgency"].isin(URGS)].reset_index(drop=True)
    return df


def report(title, df, pred_type, pred_urg):
    """유형·긴급도 정확도와 '상' 재현율, 혼동행렬을 출력한다."""
    print(f"\n==================== {title} ({len(df)}문장) ====================")
    print(f"유형 정확도   : {accuracy_score(df['type'], pred_type):.3f}")
    print(f"긴급도 정확도 : {accuracy_score(df['urgency'], pred_urg):.3f}")
    both = ((df["type"].values == pred_type) & (df["urgency"].values == pred_urg)).mean()
    print(f"둘 다 정답    : {both:.3f}")
    print(f"긴급도 '상' 재현율 (긴급 고장을 놓치지 않은 비율): "
          f"{recall_score(df['urgency'], pred_urg, labels=['상'], average='macro', zero_division=0):.3f}")

    print("\n[유형 상세]")
    print(classification_report(df["type"], pred_type, labels=TYPES, zero_division=0))
    print("[긴급도 상세]")
    print(classification_report(df["urgency"], pred_urg, labels=URGS, zero_division=0))
    print("[긴급도 혼동행렬] 행=정답, 열=예측 (상/중/하)")
    print(confusion_matrix(df["urgency"], pred_urg, labels=URGS))
