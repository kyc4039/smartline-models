"""6단계 준비: 상황 300개를 문장화 요청용 묶음으로 나누기

- 한 묶음 25개 → 12묶음
- 묶음마다 담당 분야와 조치 시점이 고르게 섞이도록 나눈다
  (한 묶음이 즉시 조치만 있으면 문장 말투가 심각도에 끌려가기 쉬움)
- 문장을 쓰는 LLM에게는 정답(조치 시점·속성)을 보여주지 않는다. 상황 설명만 준다.
출력: data/batches/batch_01.csv ~ batch_12.csv
"""
import os
import pandas as pd

BATCH = 25
SEED = 42
s = pd.read_csv("data/scenarios.csv", encoding="utf-8-sig")
os.makedirs("data/batches", exist_ok=True)

# 담당·조치 시점별로 섞은 뒤 번갈아 뽑아 묶음마다 고르게 분배
s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
n = (len(s) + BATCH - 1) // BATCH
g = s.groupby(["type", "urgency"]).ngroup()        # 담당×조치 조합 번호
k = s.groupby(["type", "urgency"]).cumcount()      # 조합 안에서의 순번
s["batch"] = (k + g) % n + 1                        # 조합마다 시작 묶음을 어긋나게 → 묶음마다 고르게

for b, g in s.groupby("batch"):
    g[["scenario_id", "equipment", "part", "situation"]].to_csv(
        f"data/batches/batch_{b:02d}.csv", index=False, encoding="utf-8-sig")
print(f"{n}개 묶음 저장: data/batches/")
print(pd.crosstab(s["urgency"].map({"상": "즉시", "중": "당일", "하": "정기"}), s["batch"]).to_string())
print("묶음 크기:", s["batch"].value_counts().sort_index().tolist())