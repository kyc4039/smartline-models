"""학습용 상황의 속성 오류 의심 행 찾기 (5단계 검수용)

상황 설명(situation)에 나온 표현과 속성 값이 어긋나 보이는 행을 규칙으로 찾아 보여준다.
찾은 행은 사람이 읽고 판단해서 data/raw/scenarios_*.csv를 고친 뒤 build_scenarios.py를 다시 실행한다.
"""
import warnings
import pandas as pd

warnings.filterwarnings("ignore", message="This pattern")

s = pd.read_csv("data/scenarios.csv", encoding="utf-8-sig")
has = lambda pat: s["situation"].str.contains(pat, regex=True)

checks = [
    ("타는 냄새·연기·불꽃인데 안전 위험 없음 → safety=있음?",
     has("타는 냄새|탄내|연기|불꽃|스파크") & (s["safety"] == "없음")),
    ("바닥 기름 고임·미끄럼인데 안전 위험 없음 → safety=있음?",
     has("바닥.*(고임|고여|흥건)|미끄") & (s["safety"] == "없음")),
    ("원인 불명·재발 의심인데 1회성이거나 가동 정상 → frequency=간헐, operation=저하?",
     has("원인 불명|재발 (가능|의심|우려)") & ((s["frequency"] == "1회성") | (s["operation"] == "정상"))),
    ("불량 유출·불량 발생·오판정인데 품질 영향 없음 → quality=있음?",
     has("불량 유출|불량 (발생|증가)|오판정|검사 없이 통과") & ~has("없었음|없음|아님") & (s["quality"] == "없음")),
    ("검사 불가·재기동 불가인데 정지가 아님 → operation=정지?",
     has("검사 불가|재기동 불가|가동 불가|전원 투입 불가|작동 불가") & (s["operation"] != "정지")),
    ("점점·늘어남·지난주보다인데 악화 아님 → trend=악화?",
     has("점점|늘어|증가|서서히|보다 (많|잦|심)|어제는") & ~has("변화 없") & (s["trend"] == "변화없음")),
    ("모터·팬의 소리·진동·커버 문제인데 전기 담당 → type=기계?",
     (s["type"] == "전기") & has("모터") & has("소음|진동|베어링|커버|흔들") & ~has("트립|기동|과열 알람|타는|전류 이상|안 돌")),
    ("모터가 안 켜짐·트립·타는 냄새인데 기계 담당 → type=전기?",
     (s["type"] == "기계") & has("모터") & has("트립|기동 (불가|안)|타는|과전류")),
    ("인터록·광커튼 감지기 문제인데 전기 담당, 또는 회로·배선 문제인데 센서 담당?",
     ((s["type"] == "전기") & has("감지기|센서 (오염|정렬|감도)")) | ((s["type"] == "센서") & has("배선|회로|단선|릴레이"))),
]

pd.set_option("display.max_colwidth", 90, "display.width", 220)
total = set()
for title, mask in checks:
    rows = s[mask]
    if len(rows):
        total |= set(rows["scenario_id"])
        print(f"\n■ {title}  ({len(rows)}개)")
        print(rows[["scenario_id", "type", "urgency", "safety", "operation", "quality", "frequency", "trend",
                    "situation"]].to_string(index=False))
print(f"\n의심 행 합계: {len(total)}개 / 전체 {len(s)}개")
print("※ 규칙에 걸렸다고 모두 오류는 아님. 상황 설명을 읽고 기준표로 판단해서 고칠 것만 고친다.")