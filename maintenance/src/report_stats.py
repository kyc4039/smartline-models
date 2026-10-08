r"""보고서용 통계 출력 (읽기 전용) — build_train.py와 같은 검사를 하지만 파일을 저장하지 않는다.
실행: maintenance 폴더에서  uv run python src/report_stats.py
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

sc = pd.read_csv("data/scenarios.csv", encoding="utf-8-sig")
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
df = df.drop_duplicates(["scenario_id", "style"], keep="last")   # train_fix.csv(나중 파일)의 고친 문장을 남김
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

# 5) 이미 저장된 분할 결과 읽기 (다시 나누지 않음)
train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
val = pd.read_csv("data/val.csv", encoding="utf-8-sig")
print(f"\n[저장된 분할] train {len(train)}문장 ({train['scenario_id'].nunique()}개 상황) / val {len(val)}문장 ({val['scenario_id'].nunique()}개 상황)")
for name, d in (("train", train), ("val", val)):
    print(f"\n[{name} 담당 × 조치 시점]")
    print(pd.crosstab(d["type"], d["urgency"].map(URG_NAME)).reindex(columns=["즉시", "당일", "정기"]).to_string())
print("\n[상황 300개 담당 × 조치 시점]")
print(pd.crosstab(sc["type"], sc["urgency"].map(URG_NAME)).reindex(columns=["즉시", "당일", "정기"]).to_string())
print("\n[상황 300개 설비별 개수]")
print(sc["equipment"].value_counts().to_string())
print("\n[train 말투별 평균 길이]")
print(train.assign(길이=train["text"].str.len()).groupby("style")["길이"].mean().round(1).to_string())
