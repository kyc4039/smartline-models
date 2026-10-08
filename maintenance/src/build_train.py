"""6~7단계: 학습 문장 합치기, 품질 검사, 상황 단위 분할

입력: data/raw/train_*.csv (scenario_id, style, text) + data/scenarios.csv (정답)
처리
  1) 값 검사: 모르는 scenario_id, 잘못된 style, 빈 문장, 중복
  2) 정답 노출 단어 검사
  3) 상황마다 4개 말투가 다 있는지
  4) 지름길 점검: 조치 시점별로 문장 길이·감정 표현이 쏠리지 않았는지
  5) 상황 단위로 train / val 분할 (같은 상황의 문장은 한쪽에만)
출력: data/train.csv, data/val.csv
"""
import glob
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

VAL_SIZE = 0.15
SEED = 42
STYLES = ["담담", "과장", "보고", "구어"]
LEAK = ["긴급", "즉시", "당일", "정기 점검", "정기점검", "담당", "안전 위험", "품질 영향"]
URG_NAME = {"상": "즉시", "중": "당일", "하": "정기"}

sc = pd.read_csv("D:/EM_model/data/scenarios.csv", encoding="utf-8-sig")
frames = []
for path in sorted(glob.glob("data/raw/train_*.csv")):
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    d.columns = [c.strip() for c in d.columns]
    for c in ("scenario_id", "style", "text"):
        d[c] = d[c].astype(str).str.strip()
    d["source_file"] = os.path.basename(path)
    frames.append(d)
    print(f"{os.path.basename(path)}: {len(d)}행")
if not frames:
    sys.exit("data/raw/train_*.csv 파일이 없습니다.")
df = pd.concat(frames, ignore_index=True)

# 1) 값 검사
problems = []
for _, r in df[~df["scenario_id"].isin(sc["scenario_id"])].iterrows():
    problems.append(f"{r['source_file']}: 모르는 scenario_id '{r['scenario_id']}'")
for _, r in df[~df["style"].isin(STYLES)].iterrows():
    problems.append(f"{r['source_file']} {r['scenario_id']}: style = '{r['style']}'")
for _, r in df[df["text"].isin(["", "nan"])].iterrows():
    problems.append(f"{r['source_file']} {r['scenario_id']} {r['style']}: 빈 문장")
if problems:
    print("\n[수정 필요]")
    print("\n".join(problems[:50]))
    sys.exit("위 항목을 고친 뒤 다시 실행하세요.")

before = len(df)
df = df.drop_duplicates(["scenario_id", "style"], keep="last")
df = df.drop_duplicates("text", keep="first")
if len(df) < before:
    print(f"\n[제거] 중복 {before - len(df)}행")

# 2) 정답 노출 단어 → 제거
leak = df["text"].str.contains("|".join(LEAK))
if leak.any():
    print(f"\n[제거] 정답 노출 단어 포함 {leak.sum()}행")
    print(df[leak][["scenario_id", "style", "text"]].to_string(index=False))
    df = df[~leak]

# 3) 상황별 말투 개수
cnt = df.groupby("scenario_id")["style"].nunique().reindex(sc["scenario_id"], fill_value=0)
lack = cnt[cnt < 4]
print(f"\n상황 {len(sc)}개 중 4개 말투 완비 {int((cnt == 4).sum())}개, 부족 {len(lack)}개")
if len(lack):
    print("부족한 상황:", ", ".join(f"{i}({n}개)" for i, n in lack.items()))

# 정답 붙이기
df = df.merge(sc[["scenario_id", "type", "urgency", "reason", "equipment"]], on="scenario_id")
df["generator"] = "claude"

# 4) 지름길 점검: 조치 시점에 따라 길이나 감정 표현이 쏠리면 모델이 말투로 정답을 외울 수 있음
df["길이"] = df["text"].str.len()
df["감정표현"] = df["text"].str.contains("!|ㅠ|ㅜ|엄청|완전|진짜|대박|큰일").astype(int)
print("\n[지름길 점검] 조치 시점별 평균 길이와 감정 표현 비율 (조치 시점끼리 비슷해야 좋음)")
chk = df.groupby(df["urgency"].map(URG_NAME))[["길이", "감정표현"]].mean().reindex(["즉시", "당일", "정기"])
chk["감정표현"] = (chk["감정표현"] * 100).round(1).astype(str) + "%"
chk["길이"] = chk["길이"].round(1)
print(chk.to_string())
print("\n[말투 × 조치 시점]")
print(pd.crosstab(df["style"], df["urgency"].map(URG_NAME)).reindex(columns=["즉시", "당일", "정기"]).to_string())

# 5) 상황 단위 분할 (담당 × 조치 시점 비율 유지)
ids = sc[sc["scenario_id"].isin(df["scenario_id"])].copy()
ids["strat"] = ids["type"] + "_" + ids["urgency"]
tr_ids, va_ids = train_test_split(ids["scenario_id"], test_size=VAL_SIZE, stratify=ids["strat"], random_state=SEED)
cols = ["text", "type", "urgency", "scenario_id", "style", "reason", "equipment", "generator"]
train, val = df[df["scenario_id"].isin(tr_ids)][cols], df[df["scenario_id"].isin(va_ids)][cols]
assert not set(train["scenario_id"]) & set(val["scenario_id"])
train.to_csv("data/train.csv", index=False, encoding="utf-8-sig")
val.to_csv("data/val.csv", index=False, encoding="utf-8-sig")
print(f"\n저장: data/train.csv {len(train)}문장 ({len(tr_ids)}개 상황) / data/val.csv {len(val)}문장 ({len(va_ids)}개 상황)")
print("\n[train 담당 × 조치 시점]")
print(pd.crosstab(train["type"], train["urgency"].map(URG_NAME)).reindex(columns=["즉시", "당일", "정기"]).to_string())