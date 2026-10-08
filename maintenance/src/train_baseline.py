"""v5 기준 모델: TF-IDF(형태소 1~2gram + 글자 2~4gram) + 로지스틱 회귀"""
import os
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import tokenize, load, report, MODELS

OUT = f"{MODELS}/v5_baseline"
os.makedirs(OUT, exist_ok=True)
train, val, dev = load("train"), load("val"), load("dev")


def make():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
    ])
    return Pipeline([("features", feats),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=5.0))])


model = {"type": make().fit(train["text"], train["type"]),
         "urgency": make().fit(train["text"], train["urgency"])}
joblib.dump(model, f"{OUT}/baseline.joblib")

for name, df in [("val (Claude)", val), ("dev (GPT+Gemini)", dev)]:
    report(f"v5 기준 모델 — {name}", df, model["type"].predict(df["text"]), model["urgency"].predict(df["text"]))
print(f"\n저장: {OUT}/baseline.joblib")