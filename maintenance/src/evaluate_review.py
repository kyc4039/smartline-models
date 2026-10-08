"""확인 필요 장치 평가: 사람에게 넘긴 문장과 자동 처리한 문장을 나눠서 본다

- 확인 필요 비율: 관리자에게 넘어가는 문장의 비율 (너무 높으면 자동화 의미가 줄어듦)
- 자동 처리 정확도: 확인 필요가 아닌 문장만의 정확도 (실제로 자동 배정되는 문장의 품질)
- 오답 포착률: 틀린 문장 중 확인 필요로 걸러진 비율

주의: '추측·질문·요청' 규칙은 최종 시험 오답 분석을 보고 만든 것이라,
     test_final 결과는 낙관적인 참고값이다. 공정한 평가는 dev 결과로 본다.
"""
import numpy as np
import pandas as pd
from common import load, URG_NAME
from predict import classify_batch

pd.set_option("display.max_colwidth", 70, "display.width", 220)

for name in ["dev", "test_final"]:
    df = load(name)
    res = classify_batch(df["text"])
    df["pred_type"] = [r["type"] for r in res]
    df["pred_urg"] = [r["urgency"] for r in res]
    df["review"] = [r["need_review"] for r in res]
    df["reasons"] = [", ".join(r["review_reasons"]) for r in res]
    df["danger"] = [r["danger_rule_applied"] for r in res]
    df["ok"] = (df["type"] == df["pred_type"]) & (df["urgency"] == df["pred_urg"])
    auto = df[~df["review"]]
    wrong = df[~df["ok"]]
    urgent = df["urgency"] == "상"

    note = "  ※ 규칙 설계에 쓴 데이터라 낙관적인 참고값" if name == "test_final" else "  (공정한 평가)"
    print(f"\n==================== {name} ({len(df)}문장){note} ====================")
    print(f"위험 단어 규칙 발동: {int(df['danger'].sum())}건")
    print(f"담당·조치 둘 다 정답 (전체): {df['ok'].mean():.3f}")
    print(f"확인 필요 비율: {df['review'].mean():.3f} ({int(df['review'].sum())}문장)")
    print(f"자동 처리 정확도 (확인 필요 제외 {len(auto)}문장): {auto['ok'].mean():.3f}")
    print(f"오답 포착률: {wrong['review'].mean() if len(wrong) else float('nan'):.3f} "
          f"(오답 {len(wrong)}개 중 {int(wrong['review'].sum())}개)")
    missed_urgent = df[urgent & (df["pred_urg"] != "상")]
    print(f"즉시 조치 미탐 {len(missed_urgent)}건 중 확인 필요로 넘어간 것: {int(missed_urgent['review'].sum())}건")
    if "generator" in df.columns:
        g = df.groupby("generator").agg(문장=("ok", "size"), 정답률=("ok", "mean"), 확인필요=("review", "mean"))
        print(g.round(3).to_string())
    print("\n[확인 필요 이유별]")
    print(df[df["review"]]["reasons"].value_counts().to_string())
    print("\n[자동 처리됐지만 틀린 문장] (사람이 보지 못하고 지나가는 실수)")
    cols = ["text", "type", "pred_type", "urgency", "pred_urg"] + (["generator"] if "generator" in df else [])
    print(auto[~auto["ok"]][cols].to_string(index=False) if (~auto["ok"]).any() else "없음")