"""평가 데이터 합치기 (문장 내용은 출력하지 않음)

사용법
  uv run python src/build_test_final.py            → 최종 시험 (test_final)
  uv run python src/build_test_final.py dev        → 선택용 데이터 (dev)

입력 (data/raw/, NAME = test_final 또는 dev)
  NAME_gpt.csv / NAME_gpt_add.csv (선택)
  NAME_gemini.csv / NAME_gemini_fix.csv (선택, 같은 id는 교체)
  NAME_human.csv (선택, 직접 작성: target 빈칸 허용, 표시 이름(유공압 등) 자동 변환, 균형 조정 없이 전부 사용)
처리
  1) 값 검사  2) 기준표 v2 규칙으로 라벨 재계산  3) 조합당 3개로 균형 맞추기
출력
  data/NAME.csv  (test_final은 봉인: 열어 보지 않는다)
"""
import os
import sys
import pandas as pd

NAME = sys.argv[1] if len(sys.argv) > 1 else "test_final"
RAW, OUT = "data/raw", f"data/{NAME}.csv"
PER_COMBO = 3
SEED = 42
COLS = ["id", "equipment", "type", "safety", "operation", "quality", "frequency", "trend", "target", "style", "text"]
VALID = {
    "equipment": ["프레스 #1", "컨베이어", "로봇 적재기", "비전 검사기", "자재 투입기"],
    "type": ["기계", "전기", "유압공압", "센서"],
    "safety": ["있음", "없음"], "operation": ["정상", "저하", "정지"], "quality": ["있음", "없음"],
    "frequency": ["1회성", "간헐", "지속"], "trend": ["변화없음", "악화"],
    "target": ["상", "중", "하"], "style": ["담담", "과장", "보고"],
}
LEAK = ["안전 위험", "품질 영향", "긴급", "즉시 조치", "정기 점검", "정기점검", "담당"]


def urgency(r):
    """기준표 v2 조치 시점 판정 규칙"""
    if r["safety"] == "있음" or r["operation"] == "정지":
        return "상"
    if r["frequency"] != "1회성" and (r["quality"] == "있음" or r["operation"] == "저하" or r["trend"] == "악화"):
        return "중"
    return "하"


def read(name):
    path = f"{RAW}/{name}"
    if not os.path.exists(path):
        return None
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].str.strip()
    if "type" in d:   # 화면 표시 이름으로 적어도 코드 값으로 변환
        d["type"] = d["type"].replace({"기계 정비": "기계", "전기·제어": "전기", "유공압": "유압공압", "센서·계측": "센서"})
    for c in ("target", "style"):
        if c not in d:
            d[c] = ""
    d["target"] = d["target"].fillna("")
    d["style"] = d["style"].fillna("")
    return d


def load_generator(gen, base, extra):
    d = read(base)
    if d is None:
        sys.exit(f"{RAW}/{base} 파일이 없습니다.")
    e = read(extra)
    if e is not None:   # 같은 id는 교체, 새 id는 추가
        d = pd.concat([d[~d["id"].isin(e["id"])], e], ignore_index=True)
        print(f"[{gen}] {extra} 반영: {len(e)}행")
    d["generator"] = gen
    return d


frames = [load_generator("gpt", f"{NAME}_gpt.csv", f"{NAME}_gpt_add.csv"),
          load_generator("gemini", f"{NAME}_gemini.csv", f"{NAME}_gemini_fix.csv")]
if os.path.exists(f"{RAW}/{NAME}_human.csv"):     # 직접 작성 문장 (있을 때만)
    frames.append(load_generator("human", f"{NAME}_human.csv", f"{NAME}_human_fix.csv"))
df = pd.concat(frames, ignore_index=True)

# dev는 최종 시험과 같은 문장이 있으면 제거 (문장은 출력하지 않음)
if NAME != "test_final" and os.path.exists("data/test_final.csv"):
    sealed = set(pd.read_csv("data/test_final.csv", encoding="utf-8-sig")["text"].str.strip())
    overlap = df["text"].str.strip().isin(sealed)
    if overlap.any():
        print(f"[제거] 최종 시험과 같은 문장 {overlap.sum()}개")
        df = df[~overlap]

# 1) 값 검사
problems = []
for c, ok in VALID.items():
    allowed = ok + ([""] if c in ("target", "style") else [])
    for _, r in df[~df[c].isin(allowed)].iterrows():
        problems.append(f"{r['generator']} id {r['id']}: {c} = '{r[c]}'")
for _, r in df[df["text"].astype(str).str.contains("|".join(LEAK))].iterrows():
    problems.append(f"{r['generator']} id {r['id']}: 정답 노출 단어 포함")
dup = df[df.duplicated(["generator", "id"], keep=False)]
for _, r in dup.iterrows():
    problems.append(f"{r['generator']} id {r['id']}: id 중복")
if problems:
    print("\n[수정 필요]")
    print("\n".join(sorted(set(problems))))
    sys.exit("위 항목을 고친 뒤 다시 실행하세요.")

# 2) 규칙으로 라벨 재계산
df["urgency"] = df.apply(urgency, axis=1)
mism = df[(df["target"] != "") & (df["urgency"] != df["target"])]
if len(mism):
    print(f"\n[참고] 생성기 target과 규칙 계산이 다른 행 {len(mism)}개 → 규칙 계산 값을 정답으로 사용")
    print(mism[["generator", "id", "target", "urgency"]].to_string(index=False))

# 3) 생성기별로 조합당 PER_COMBO개 (넘치면 무작위 선택, 모자라면 경고)
parts = []
human = df[df["generator"] == "human"]
if len(human):
    parts.append(human)
    print(f"[사람 작성] {len(human)}문장은 균형 조정 없이 모두 사용")
for gen in [g for g in df["generator"].unique() if g != "human"]:
    for t in VALID["type"]:
        for u in VALID["target"]:
            g = df[(df["generator"] == gen) & (df["type"] == t) & (df["urgency"] == u)]
            if len(g) < PER_COMBO:     # 0개인 조합도 빠짐없이 알려줌
                print(f"[부족] {gen} {t}·{u}: {len(g)}개 (목표 {PER_COMBO})")
            if len(g):
                parts.append(g.sample(min(len(g), PER_COMBO), random_state=SEED))
final = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

final = final[["generator", "id", "equipment", "type", "urgency", "safety", "operation", "quality",
               "frequency", "trend", "style", "text"]]
final.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"\n저장: {OUT}  ({len(final)}문장)")
print(pd.crosstab([final["generator"], final["type"]], final["urgency"]).to_string())
print("\n말투 × 조치 시점")
print(pd.crosstab(final["urgency"], final["style"]).to_string())