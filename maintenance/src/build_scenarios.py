"""학습용 상황 정리 (5단계)

입력: data/raw/scenarios_*.csv  (Claude가 설계한 상황)
처리: 1) 값 검사  2) 기준표 규칙으로 조치 시점 계산  3) 분포 점검  4) 상황 ID 부여
출력: data/scenarios.csv
"""
import glob
import os
import sys
import pandas as pd

RAW, OUT = "data/raw", "data/scenarios.csv"
COLS = ["no", "equipment", "part", "type", "safety", "operation", "quality", "frequency", "trend", "situation"]
VALID = {
    "equipment": ["프레스 #1", "컨베이어", "로봇 적재기", "비전 검사기", "자재 투입기"],
    "type": ["기계", "전기", "유압공압", "센서"],
    "safety": ["있음", "없음"], "operation": ["정상", "저하", "정지"], "quality": ["있음", "없음"],
    "frequency": ["1회성", "간헐", "지속"], "trend": ["변화없음", "악화"],
}
TYPE_ALIAS = {"기계 정비": "기계", "전기·제어": "전기", "유공압": "유압공압", "센서·계측": "센서"}
URG_NAME = {"상": "즉시", "중": "당일", "하": "정기"}


def urgency(r):
    """기준표 조치 시점 판정 규칙"""
    if r["safety"] == "있음" or r["operation"] == "정지":
        return "상"
    if r["frequency"] != "1회성" and (r["quality"] == "있음" or r["operation"] == "저하" or r["trend"] == "악화"):
        return "중"
    return "하"


def reason(r):
    """조치 시점이 정해진 근거 (분포 점검용)"""
    if r["urgency"] == "상":
        return "+".join(k for k, ok in [("안전", r["safety"] == "있음"), ("정지", r["operation"] == "정지")] if ok)
    if r["urgency"] == "중":
        return "+".join(k for k, ok in [("품질", r["quality"] == "있음"), ("저하", r["operation"] == "저하"),
                                        ("악화", r["trend"] == "악화")] if ok)
    return "1회성해소" if r["frequency"] == "1회성" else "영향없음"


frames = []
for path in sorted(glob.glob(f"{RAW}/scenarios_*.csv")):
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    d.columns = [c.strip() for c in d.columns]
    missing = [c for c in COLS if c not in d.columns]
    if missing:
        sys.exit(f"{os.path.basename(path)}: 빠진 열 {missing}")
    for c in COLS:
        if d[c].dtype == object:
            d[c] = d[c].str.strip()
    d["type"] = d["type"].replace(TYPE_ALIAS)
    d["source_file"] = os.path.basename(path)
    frames.append(d)
    print(f"{os.path.basename(path)}: {len(d)}개")
if not frames:
    sys.exit(f"{RAW}/scenarios_*.csv 파일이 없습니다.")
df = pd.concat(frames, ignore_index=True)

# 1) 값 검사
problems = []
for c, ok in VALID.items():
    for _, r in df[~df[c].isin(ok)].iterrows():
        problems.append(f"{r['source_file']} no {r['no']}: {c} = '{r[c]}'")
for _, r in df[df["situation"].isna() | (df["situation"].astype(str).str.len() < 5)].iterrows():
    problems.append(f"{r['source_file']} no {r['no']}: situation 비어 있음")
if problems:
    print("\n[수정 필요]")
    print("\n".join(problems))
    sys.exit("위 항목을 고친 뒤 다시 실행하세요.")

dup = df["situation"].duplicated()
if dup.any():
    print(f"\n[제거] 같은 상황 설명 {dup.sum()}개")
    df = df[~dup]

# 2) 조치 시점 계산
df["urgency"] = df.apply(urgency, axis=1)
df["reason"] = df.apply(reason, axis=1)

# 3) 분포 점검 (열 순서: 즉시 → 당일 → 정기)
ORDER = ["즉시", "당일", "정기"]
df["조치"] = pd.Categorical(df["urgency"].map(URG_NAME), categories=ORDER)
print(f"\n총 {len(df)}개 상황")
print("\n[담당 × 조치 시점]  (목표: 칸마다 25)")
print(pd.crosstab(df["type"], df["조치"]).to_string())
print("\n[조치 시점별 근거]  (한 근거에 쏠리지 않았는지)")
print(pd.crosstab(df["reason"], df["조치"]).to_string())
print("\n[설비 × 조치 시점]")
print(pd.crosstab(df["equipment"], df["조치"]).to_string())
part_share = df.groupby("type")["part"].value_counts(normalize=True)
over = part_share[part_share > 0.12]
if len(over):
    print("\n[주의] 한 담당 안에서 12%를 넘는 부품")
    print((over * 100).round(1).astype(str).add("%").to_string())

# 4) 상황 ID 부여 후 저장
df = df.sort_values(["type", "urgency", "equipment"]).reset_index(drop=True)
df.insert(0, "scenario_id", [f"S{i + 1:04d}" for i in range(len(df))])
df[["scenario_id", "equipment", "part", "type", "urgency", "reason", "safety", "operation", "quality",
    "frequency", "trend", "situation"]].to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n저장: {OUT}")