"""기준 모델: TF-IDF + 로지스틱 회귀 (유형, 긴급도 각각 학습)"""
import os
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from text_utils import tokenize, load, report

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

train = load(f"{DATA_DIR}/train.csv")
val = load(f"{DATA_DIR}/val.csv")
test_llm = load(f"{DATA_DIR}/test_llm.csv")  # 없으면 건너뜀


def make_pipeline():
    features = FeatureUnion([
        # 형태소 단위 특징 (단어 1~2개 묶음)
        ("word", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                 lowercase=False, ngram_range=(1, 2))),
        # 글자 단위 특징 (오타·줄임말에 강함)
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)
    return Pipeline([("features", features), ("clf", clf)])


type_model = make_pipeline().fit(train["text"], train["type"])
urg_model = make_pipeline().fit(train["text"], train["urgency"])

for name, df in [("검증 데이터 (val)", val), ("다른 LLM 테스트 (test_llm)", test_llm)]:
    if df is not None:
        report(name, df, type_model.predict(df["text"]), urg_model.predict(df["text"]))

joblib.dump({"type": type_model, "urgency": urg_model}, f"{MODEL_DIR}/baseline.joblib")
print(f"\n모델 저장 완료: {MODEL_DIR}/baseline.joblib")