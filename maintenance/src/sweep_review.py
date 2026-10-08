"""확인 필요 기준값(LOW_CONF) 고르기 — dev로만 결정

선택 규칙 (실행 전에 정함):
  확인 필요 비율이 MAX_REVIEW 이하인 기준값 중에서
  1) 자동 처리 정확도가 가장 높고  2) 같으면 확인 필요 비율이 낮은 값
결정한 값을 predict.py의 LOW_CONF에 반영한다. test_final로는 고르지 않는다.
"""
import numpy as np
import pandas as pd
import predict
from common import load, URG_NAME

MAX_REVIEW = 0.20                    # 관리자에게 넘기는 문장은 최대 20%
CANDIDATES = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

df = load("dev")
res = predict.classify_batch(df["text"])
conf = np.array([min(max(r["type_probs"].values()), max(r["urgency_probs"].values())) for r in res])
other = np.array([any(x != "확신도 낮음" for x in r["review_reasons"]) for r in res])   # 짧음·추측 표현
ok = (df["type"].values == np.array([r["type"] for r in res])) & \
     (df["urgency"].values == np.array([r["urgency"] for r in res]))
urgent_miss = (df["urgency"].values == "상") & (np.array([r["urgency"] for r in res]) != "상")

rows = []
for th in CANDIDATES:
    review = (conf < th) | other
    auto = ~review
    rows.append({"LOW_CONF": th, "확인필요비율": review.mean(), "자동처리정확도": ok[auto].mean() if auto.any() else np.nan,
                 "오답포착률": review[~ok].mean() if (~ok).any() else np.nan,
                 "즉시미탐_포착": f"{int(review[urgent_miss].sum())}/{int(urgent_miss.sum())}"})
r = pd.DataFrame(rows)
print(f"[dev {len(df)}문장] 현재 LOW_CONF = {predict.LOW_CONF}")
print(r.round(3).to_string(index=False))

ok_rows = r[r["확인필요비율"] <= MAX_REVIEW]
best = ok_rows.sort_values(["자동처리정확도", "확인필요비율"], ascending=[False, True]).iloc[0]
print(f"\n선택 규칙(확인 필요 {MAX_REVIEW:.0%} 이하 → 자동 처리 정확도 최고 → 확인 필요 비율 낮음)")
print(f"→ 추천 LOW_CONF = {best['LOW_CONF']}  (predict.py에서 직접 바꿔 주세요)")