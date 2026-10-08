"""설비 보전 AI 서비스용 분류기 (v5)
Flask 등에서:  from predict import classify
              result = classify("컨베이어 롤러에서 소리 남")

동작
  1) 최종 조합(models/v5_final.json)으로 담당 분야·조치 시점 확률 계산
  2) 위험 단어 규칙: 모델이 즉시가 아니라고 해도 위험 단어가 있으면 즉시 조치로 올림
  3) 확인 필요 판단: 아래 중 하나라도 해당하면 관리자 확인 필요
       - 확신도 낮음 (담당·조치 중 가장 높은 확률이 LOW_CONF 미만)
       - 문장이 너무 짧음 (MIN_LEN 미만)
       - 추측·질문·요청 표현 ("~것 같은데", "점검해야 하나?", "봐 주세요" 등)
  4) 협업 필요: 두 번째 담당 확률이 첫 번째와 가까우면 함께 표시
"""
import json
import re
import numpy as np
from common import TYPES, URGS, TYPE_NAME, URG_NAME, MODELS
from predictors import load_v5, combine

DANGER_WORDS = ["연기", "불꽃", "불똥", "타는 냄새", "탄내", "감전", "스파크", "누전", "화재",
                "폭발", "침수", "젖었", "물이 떨어", "물 떨어", "터졌", "터진", "협착", "끼임", "끼였"]
# "불이"는 제외: "센서 불이 깜빡"처럼 표시등을 뜻하는 경우가 많아 오경보를 만듦 (dev에서 확인)
LOW_CONF = 0.5          # 가장 높은 확률이 이보다 낮으면 확인 필요
MIN_LEN = 5             # 이보다 짧은 문장은 확인 필요
COLLAB_GAP = 0.15       # 1·2위 담당 확률 차이가 이보다 작으면 협업 필요
HEDGE = re.compile(r"\?|것 ?같[은아다]|인지 ?모르|모르겠|봐 ?주세요|봐줘|해야 ?하나|불러 ?줘|없나|어떡|어떻게 ?해")

with open(f"{MODELS}/v5_final.json", encoding="utf-8") as f:
    WEIGHTS = {k: v for k, v in json.load(f)["weights"].items() if v > 0}
_models = load_v5()
_missing = [k for k in WEIGHTS if k not in _models]
if _missing:
    raise SystemExit(f"최종 조합에 필요한 모델이 없습니다: {_missing}")


def _probs(texts):
    import pandas as pd
    df = pd.DataFrame({"text": [str(t).strip() for t in texts]})
    return combine({k: _models[k](df) for k in WEIGHTS}, WEIGHTS)


def _decide(text, pt, pu):
    text = str(text).strip()
    t_order = np.argsort(pt)[::-1]
    urgency = URGS[int(pu.argmax())]

    danger = urgency != "상" and any(w in text for w in DANGER_WORDS)
    if danger:
        urgency = "상"

    reasons = []
    if min(pt.max(), pu.max()) < LOW_CONF:     # 담당·조치 중 하나라도 확신이 낮으면
        reasons.append("확신도 낮음")
    if len(text) < MIN_LEN:
        reasons.append("문장이 짧음")
    if HEDGE.search(text):
        reasons.append("추측·질문·요청 표현")

    second = TYPES[int(t_order[1])]
    collab = second if pt[t_order[0]] - pt[t_order[1]] < COLLAB_GAP else None

    return {
        "type": TYPES[int(t_order[0])],
        "type_name": TYPE_NAME[TYPES[int(t_order[0])]],
        "type_probs": {t: round(float(p), 3) for t, p in zip(TYPES, pt)},
        "urgency": urgency,
        "urgency_name": URG_NAME[urgency] + " 조치" if urgency != "하" else "정기 점검",
        "urgency_probs": {URG_NAME[u]: round(float(p), 3) for u, p in zip(URGS, pu)},
        "danger_rule_applied": danger,
        "need_review": bool(reasons),
        "review_reasons": reasons,
        "collab": TYPE_NAME[collab] if collab else None,
    }


def classify(text):
    """문장 하나 분류"""
    pt, pu = _probs([text])
    return _decide(text, pt[0], pu[0])


def classify_batch(texts):
    """여러 문장 한 번에 분류 (평가용, 훨씬 빠름)"""
    texts = list(texts)
    pt, pu = _probs(texts)
    return [_decide(t, pt[i], pu[i]) for i, t in enumerate(texts)]


if __name__ == "__main__":
    for t in ["프레스 내려올 때 끼익 소리 계속 남",
              "제어반에서 타는 냄새 나요 빨리 와주세요",
              "로봇 적재기에서 이상한 소리 들리는데, 이거 점검해야하나?",
              "소음"]:
        r = classify(t)
        flag = f" [확인 필요: {', '.join(r['review_reasons'])}]" if r["need_review"] else ""
        collab = f" (협업: {r['collab']})" if r["collab"] else ""
        print(f"{t}\n  → {r['type_name']}{collab} / {r['urgency_name']}{flag}  {r['urgency_probs']}")