"""8단계-②: dev로 최종 조합 선택
v5 모델들의 확률을 0.25 간격 비중으로 섞은 모든 조합을 dev에서 평가하고,
선택 규칙(common.select_key)으로 하나를 고른다.
규칙: 1) 즉시 조치 탐지율 ≥ 0.9  2) 조치 시점 정확도 최고  3) 오경보 최소  4) 담당 정확도 최고
      (그래도 같으면 쓰는 모델 수가 적은 조합, 그다음 기준 모델 비중이 큰 조합)
출력: models/v5_final.json
"""
import itertools, json
import pandas as pd
from common import load, metrics, select_key, fmt, report, to_labels, MIN_URGENT_RECALL, MODELS
from predictors import load_v5, load_v1, combine

dev = load("dev")
models = load_v5()
if not models:
    raise SystemExit("v5 모델이 없습니다. 먼저 train_*.py를 실행하세요.")
print("불러온 v5 모델:", ", ".join(models))
probs = {k: f(dev) for k, f in models.items()}

names = list(models)
steps = [0, 0.25, 0.5, 0.75, 1.0]
rows = []
for ws in itertools.product(steps, repeat=len(names)):
    if abs(sum(ws) - 1) > 1e-9:
        continue
    w = dict(zip(names, ws))
    m = metrics(dev, *to_labels(*combine(probs, w)))
    # 동점이면 더 단순한 조합(쓰는 모델 수가 적은 것), 그다음 가벼운 기준 모델 비중이 큰 것
    simple = (-sum(v > 0 for v in ws), w.get("baseline", 0))
    rows.append({**{f"w_{k}": v for k, v in w.items()}, **m, "key": select_key(m) + simple})

res = pd.DataFrame(rows).sort_values("key", ascending=False).reset_index(drop=True)
show = res.drop(columns="key").copy()
show["후보"] = res["key"].apply(lambda k: "O" if k[0] else "")
print(f"\n[dev 결과] 즉시 탐지율 {MIN_URGENT_RECALL} 이상만 후보(O), 위에서부터 선택 순서")
print(show.round(3).to_string(index=False))

best = res.iloc[0]
if not best["key"][0]:
    print(f"\n[주의] 즉시 탐지율 {MIN_URGENT_RECALL} 이상인 조합이 없어, 규칙의 2~4순위로 선택했습니다.")
weights = {k: float(best[f"w_{k}"]) for k in names}
print(f"\n선택된 조합: {weights}")
pt, pu = combine(probs, weights)
m_best = report("v5 선택 조합 — dev", dev, *to_labels(pt, pu))

v1 = load_v1()
if v1:
    report("v1 최종 모델(비교) — dev", dev, *to_labels(*v1(dev)))

with open(f"{MODELS}/v5_final.json", "w", encoding="utf-8") as f:
    json.dump({"weights": weights, "rule": f"즉시 탐지율 >= {MIN_URGENT_RECALL} → 조치 정확도 → 오경보 → 담당 정확도",
               "dev": {k: (float(v) if not isinstance(v, int) else v) for k, v in m_best.items()}}, f,
              ensure_ascii=False, indent=2)
print(f"\n저장: {MODELS}/v5_final.json")