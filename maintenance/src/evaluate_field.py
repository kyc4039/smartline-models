"""현장 피드백 데이터로 서비스 분류기 평가 (피드백 학습 루프의 '평가' 단계)

1) 웹의 AI 학습 페이지에서 '학습 데이터 내보내기'로 받은 CSV를 data/field/ 에 저장
2) EM_model 폴더에서:  uv run python src/evaluate_field.py
정답은 정비원이 확인한 값(type, urgency). 서비스와 똑같이 predict.py(위험 단어 규칙 포함)로 판정한다.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
from common import report, TYPES, URGS

files = sorted(glob.glob("data/field/*.csv"))
if not files:
    sys.exit("data/field/ 에 현장 피드백 CSV가 없습니다. 웹의 AI 학습 페이지에서 내보내 저장하세요.")
df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
df = df.drop_duplicates("ticket_id", keep="last")
df = df[df["type"].isin(TYPES) & df["urgency"].isin(URGS)].reset_index(drop=True)
print(f"현장 피드백 {len(df)}건 ({', '.join(os.path.basename(f) for f in files)})")
if df.empty:
    sys.exit("평가할 데이터가 없습니다.")

from predict import classify_batch          # 모델 불러오기 (몇 초 걸림)
res = classify_batch(df["text"])
p_type = np.array([r["type"] for r in res])
p_urg = np.array([r["urgency"] for r in res])

report("현재 서비스 모델 — 현장 피드백 전체", df, p_type, p_urg)
if "corrected" in df and df["corrected"].any():
    m = df["corrected"].astype(bool).values
    report("그중 정비원이 고친 문장 (당시 AI가 틀렸던 것)", df[m].reset_index(drop=True), p_type[m], p_urg[m])

review = np.array([r["need_review"] for r in res])
wrong = (df["type"].values != p_type) | (df["urgency"].values != p_urg)
print(f"\n확인 필요로 넘어간 비율: {review.mean():.3f} · 오답 중 확인 필요로 걸러진 비율: "
      f"{review[wrong].mean() if wrong.any() else float('nan'):.3f}")
print("\n※ 고친 문장의 점수가 낮을수록, 이 데이터를 학습에 더했을 때 얻을 것이 많다는 뜻입니다.")
