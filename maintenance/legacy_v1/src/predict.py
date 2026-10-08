"""고장 코멘트 분류기 (Flask에서 `from predict import classify` 로 사용)

MODEL_TYPE 으로 사용할 모델을 고른다.
  "ensemble" : 기준 모델 + BiLSTM 확률 가중 평균 (현재 최종 모델)
  "baseline" : TF-IDF + 로지스틱 회귀 단독 (TensorFlow 불필요 → 가볍고 빠름)
  "bilstm"   : BiLSTM 단독 (비교 실험용)
"""
import json
import numpy as np

MODEL_TYPE = "ensemble"
MODEL_DIR = "D:\\EM_model\legacy_v1\models_v4"
LSTM_WEIGHT = 0.25     # 앙상블에서 BiLSTM 비중 (val로 선택한 값)

# 모델이 '상'이 아니라고 해도, 이 단어가 있으면 긴급도를 '상'으로 올린다
DANGER_WORDS = ["연기", "불꽃", "불똥", "불이", "타는 냄새", "탄내", "감전", "스파크",
                "누전", "화재", "폭발", "침수", "젖었", "물이 떨어", "물 떨어", "터졌", "터진"]
LOW_CONFIDENCE = 0.5   # 이 확률보다 낮으면 '관리자 확인 필요'

TYPES = ["기계", "전기", "유압공압", "센서"]
URGS = ["상", "중", "하"]

# ---------- 모델 불러오기 ----------
def _load_baseline():
    import joblib
    base = joblib.load(f"{MODEL_DIR}/baseline.joblib")

    def probs(text):
        out = []
        for key, labels in (("type", TYPES), ("urgency", URGS)):
            m = base[key]
            p = m.predict_proba([text])[0]
            out.append(np.array([p[list(m.classes_).index(l)] for l in labels]))
        return out
    return probs


def _load_bilstm():
    from tensorflow import keras
    from text_utils import tokenize
    model = keras.models.load_model(f"{MODEL_DIR}/bilstm.keras")
    with open(f"{MODEL_DIR}/bilstm_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    def probs(text):
        ids = [meta["vocab"].get(tok, 1) for tok in tokenize(text)][:meta["max_len"]]
        seq = np.zeros((1, meta["max_len"]), dtype="int32")
        seq[0, :len(ids)] = ids
        p_type, p_urg = model.predict(seq, verbose=0)
        return p_type[0], p_urg[0]
    return probs


if MODEL_TYPE == "baseline":
    _probs = _load_baseline()
elif MODEL_TYPE == "bilstm":
    _probs = _load_bilstm()
elif MODEL_TYPE == "ensemble":
    _base_probs, _lstm_probs = _load_baseline(), _load_bilstm()

    def _probs(text):
        bt, bu = _base_probs(text)
        lt, lu = _lstm_probs(text)
        w = LSTM_WEIGHT
        return (1 - w) * bt + w * lt, (1 - w) * bu + w * lu
else:
    raise ValueError(f"MODEL_TYPE을 확인하세요: {MODEL_TYPE}")


def classify(text):
    p_type, p_urg = _probs(text)

    urgency = URGS[int(p_urg.argmax())]
    rule_applied = False
    if urgency != "상" and any(w in text for w in DANGER_WORDS):
        urgency, rule_applied = "상", True

    need_review = bool(p_type.max() < LOW_CONFIDENCE or p_urg.max() < LOW_CONFIDENCE)
    if len(text.strip()) < 5:          # 너무 짧아 판단 근거가 부족한 문장
        need_review = True

    return {
        "type": TYPES[int(p_type.argmax())],
        "type_probs": {t: round(float(p), 3) for t, p in zip(TYPES, p_type)},
        "urgency": urgency,
        "urgency_probs": {u: round(float(p), 3) for u, p in zip(URGS, p_urg)},
        "danger_rule_applied": rule_applied,
        "need_review": need_review,
        "model": MODEL_TYPE,
    }


if __name__ == "__main__":
    examples = [
        "프레스 내려올 때 끼익 소리 계속 남",
        "제어반에서 타는 냄새 나요 빨리 와주세요",
        "천장 케이블 트레이에서 불똥떨어짐",
        "에어호스 연결부에서 쉭쉭 소리",
        "로봇이 제품 위치 못잡고 헛손질함 ㅠ",
        "위치센서 값 흔들림 치수는 정상",
        "소음",
    ]
    for t in examples:
        r = classify(t)
        print(f"{t}\n  → {r['type']} / {r['urgency']}"
              f"{' (위험단어 규칙)' if r['danger_rule_applied'] else ''}"
              f"{' [확인 필요]' if r['need_review'] else ''}  {r['urgency_probs']}")