import pandas as pd
df = pd.read_csv("data/test_llm_review2.csv")       # 검수한 파일
df["urgency_orig"] = df["urgency"]
df["urgency"] = df["new_urgency"]
df.to_csv("data/test_llm.csv", index=False, encoding="utf-8-sig")
print((df["urgency"] != df["urgency_orig"]).sum(), "문장 변경")
print(df["urgency"].value_counts())