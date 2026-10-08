import pandas as pd, glob, os
from sklearn.model_selection import train_test_split

GEN_DIR = "data/gen"
OUT_DIR = "data"
TYPES = ["기계", "전기", "유압공압", "센서"]
URGS = ["상", "중", "하"]

# 1) 파일별로 읽고 개수 확인
frames = []
for f in sorted(glob.glob(f"{GEN_DIR}/gen_*.csv")):
    name = os.path.basename(f)
    try:
        d = pd.read_csv(f, encoding="utf-8")
    except UnicodeDecodeError:
        d = pd.read_csv(f, encoding="cp949")
    except Exception as e:
        print(f"[읽기 오류] {name}: {e}")
        continue
    d["source"] = name.split("_")[1]
    d["file"] = name
    print(f"{name}: {len(d)}개")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
before = len(df)

# 2) 정제
df["text"] = df["text"].astype(str).str.strip().str.strip('"')
df["type"] = df["type"].astype(str).str.strip()
df["urgency"] = df["urgency"].astype(str).str.strip()

bad = df[~df["type"].isin(TYPES) | ~df["urgency"].isin(URGS)]
if len(bad):
    print(f"\n[라벨 오류 {len(bad)}개 제거]")
    print(bad[["file", "text", "type", "urgency"]].head(10))

df = df[df["type"].isin(TYPES) & df["urgency"].isin(URGS)]
df = df[df["text"].str.len() >= 2]
df = df.drop_duplicates(subset="text")
print(f"\n정제 전 {before}개 → 정제 후 {len(df)}개")

# 3) 정답 노출 단어 점검
leak = ["긴급", "위험도", "기계 고장", "전기 문제", "센서 고장", "유압 문제"]
leaked = df[df["text"].str.contains("|".join(leak))]
print(f"\n[정답 노출 의심 문장 {len(leaked)}개]")
print(leaked[["text", "type", "urgency"]].head(10))

# 4) 조합별 개수
print("\n[유형 × 긴급도 개수]")
print(pd.crosstab(df["type"], df["urgency"]))

# 5) 품질 점검용 샘플
sample = df.sample(30, random_state=0)[["text", "type", "urgency"]]
sample.to_csv(f"{OUT_DIR}/check_sample.csv", index=False, encoding="utf-8-sig")

# 6) 저장 및 분할
df = df.drop(columns=["file"])
df.to_csv(f"{OUT_DIR}/train_all.csv", index=False, encoding="utf-8-sig")

df["label"] = df["type"] + "_" + df["urgency"]
train, val = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train.to_csv(f"{OUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
val.to_csv(f"{OUT_DIR}/val.csv", index=False, encoding="utf-8-sig")
print(f"\n학습 {len(train)}개 / 검증 {len(val)}개 저장 완료")