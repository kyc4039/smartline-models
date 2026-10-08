"""8단계-③: 최종 시험 (test_final) — 단 한 번
- models/v5_final.json의 조합으로 평가, v1 최종 모델과 같은 시험으로 비교
- 전체 / 생성기별 / 말투 엇갈림 문장으로 나눠 보고
- 한 번 실행하면 잠금 파일이 생겨, 다시 실행하려면 --force 필요
"""
import os, sys, json
import numpy as np
import pandas as pd
from common import load, metrics, fmt, report, to_labels, tone_subsets, URG_NAME, MODELS
from predictors import load_v5, load_v1, combine

LOCK = f"{MODELS}/v5_final_test.done"
if os.path.exists(LOCK) and "--force" not in sys.argv:
    raise SystemExit("최종 시험은 이미 한 번 실행했습니다. 결과는 models/v5_final_test_result.json을 보세요.\n"
                     "(정말 다시 실행해야 하면 --force, 단 그 결과는 '최종 시험'으로 보고하지 않습니다)")

with open(f"{MODELS}/v5_final.json", encoding="utf-8") as f:
    cfg = json.load(f)
test = load("test_final")
models = load_v5()
probs = {k: f(test) for k, f in models.items() if cfg["weights"].get(k, 0) > 0}
p_type, p_urg = to_labels(*combine(probs, cfg["weights"]))
print(f"최종 조합: {cfg['weights']}")

result = {"weights": cfg["weights"]}
result["v5_전체"] = report("v5 — 최종 시험 전체", test, p_type, p_urg)
for gen in ["gpt", "gemini", "human"]:
    mask = (test["generator"] == gen).values
    if mask.any():
        result[f"v5_{gen}"] = report(f"v5 — {gen}", test[mask].reset_index(drop=True), p_type[mask], p_urg[mask])

# ---------- v1과 비교 ----------
# v1 최종 모델(기준 0.75 + BiLSTM 0.25)과, 마스크 환경 차이의 영향을 받지 않는 v1 기준 모델 단독을 모두 비교
compare = {"v5": (p_type, p_urg)}
for name, loader in [("v1", lambda: load_v1()), ("v1_기준단독", lambda: load_v1(baseline_only=True))]:
    f = loader()
    if f:
        compare[name] = to_labels(*f(test))
        result[f"{name}_전체"] = report(f"{name} — 최종 시험 전체", test, *compare[name])
        for gen in ["gpt", "gemini", "human"]:
            mask = (test["generator"] == gen).values
            if mask.any():
                result[f"{name}_{gen}"] = metrics(test[mask].reset_index(drop=True),
                                                  compare[name][0][mask], compare[name][1][mask])

if len(compare) > 1:
    print("\n========== v1 vs v5 요약 ==========")
    rows = []
    for part in ["전체", "gpt", "gemini", "human"]:
        for ver in compare:
            m = result.get(f"{ver}_{part}")
            if m:
                rows.append({"데이터": part, "모델": ver, "담당": m["type_acc"], "조치": m["urg_acc"],
                             "즉시 탐지": m["urgent_recall"], "오경보": m["false_alarm"], "문장": m["n"]})
    for name, mask in tone_subsets(test).items():
        for ver, (_, pu) in compare.items():
            rows.append({"데이터": name, "모델": ver, "조치": float(np.mean(test["urgency"].values[mask] == pu[mask])),
                         "문장": int(mask.sum())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

out = test.assign(pred_type=p_type, pred_urgency=p_urg)
out.to_csv(f"{MODELS}/v5_final_test_pred.csv", index=False, encoding="utf-8-sig")
with open(f"{MODELS}/v5_final_test_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=float)
open(LOCK, "w").write("최종 시험 실행 완료\n")
print(f"\n저장: {MODELS}/v5_final_test_result.json, v5_final_test_pred.csv")