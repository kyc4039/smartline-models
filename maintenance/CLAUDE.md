# SMART LINE MES — 프로젝트 안내 (Claude Code용)

이 파일은 이 프로젝트를 처음 보는 Claude가 맥락을 이해하도록 정리한 문서다.
사용자는 AI 스마트제조 융합인재 과정의 4인 팀 팀장이고, **발표일은 2026-10-08**이다.
설명은 한국어로, 전문 용어는 풀어서 한다. 사용자는 글보다 그림·표·목업을 더 잘 이해한다.

---

## 1. 무엇을 만드는 프로젝트인가

공장 작업 라인을 관리하는 MES 웹 대시보드 (Flask). 팀원 4명의 AI 모델을 한 화면에 묶는다.

| 기능 | 담당 | 모델 | 웹 모듈 |
|---|---|---|---|
| 안전 게이트 (안전모·조끼·장갑 착용 확인 = 로그인) | 팀원1 | YOLO | `web/modules/gate.py` |
| 품질 검사 (컵 불량: scratch, wrinkled) | 팀원2 | YOLO | `web/modules/quality.py` |
| 프레스 안전 인터록 (위험구역 5초 비면 자동 재가동, 그동안 전 화면 경고) | 팀원3 | YOLO | `web/modules/safety.py` |
| 설비 보전 (고장 보고 문장 → 담당·조치 시점 분류) | 팀장(사용자) | 앙상블 NLP | `web/modules/maintenance.py` + `src/` |

카메라는 3대 (게이트·품질·프레스). 재가동 수신호(gesture) 기능은 제거했다 — 되살리지 말 것. 카메라나 모델 파일이 없으면 각 워커가 '더미 모드'로 돌아서 화면은 뜬다.

---

## 2. 폴더 구조 (D:\EM_model, 집에서는 Mac의 .../AI_Class_Prj/EM_model)

```
EM_model/
├── CLAUDE.md          ← 이 파일
├── pyproject.toml, uv.lock   (uv 가상환경: em-model)
├── src/               설비 보전 NLP 파이프라인 (데이터 생성 → 학습 → 선택 → 최종 평가 → 서비스)
├── data/              학습·평가 데이터 (아래 '절대 삭제 금지' 참고)
├── models/            학습된 모델 (v5 최종: models/v5_final.json이 가리키는 파일들)
├── docs/              기준표_v2.md, 현장용어.md, 팀원_모델_정리_가이드.md
├── prompts/           데이터 생성에 쓴 프롬프트 (실험 기록)
└── web/               Flask MES (아래 3장)
```

### src/ 파일 역할 (모두 실험 재현에 필요한 기록이다 — 지우지 말 것)
- 데이터: `build_test_final.py`(최종 시험·dev 합치기), `build_scenarios.py`, `check_scenarios.py`, `make_batches.py`, `build_train.py`
- 공통: `common.py`(라벨 목록, 토큰화 kiwipiepy, 평가 지표, 선택 규칙)
- 학습: `train_baseline.py`, `train_bilstm.py`(양방향 pre-pad 입력), `train_klue.py`(KLUE-RoBERTa)
- 선택·평가: `predictors.py`, `select_dev.py`, `final_test.py`(봉인된 최종 시험, lock 파일 사용)
- 서비스: `predict.py`(웹이 불러 쓰는 분류기: classify / classify_batch, 위험 단어·재확인 필요 판단)
- 분석: `evaluate_review.py`, `sweep_review.py`, `evaluate_field.py`(현장 피드백 CSV 평가)

최종 모델 = baseline 0.25 + BiLSTM 0.25 + KLUE 0.5 앙상블 (`models/v5_final.json`).
조치 시점 규칙(기준표 v2): 안전 위험 있음 또는 설비 정지 → 상(즉시) / 1회성이 아니고 품질·성능 저하·악화 → 중(당일) / 그 외 하(정기).

---

## 3. web/ 구조

```
web/
├── app.py            서버 시작 (create_app, start_workers). use_reloader=False 유지 (카메라 이중 실행 방지)
├── config.py         ★ 사용자 개인 설정 (카메라 번호, 모델 경로, 기준값, EM_MODEL_DIR, ADMIN_PIN 등) — 내용 변경 금지
├── state.py          공유 상태 (스레드 잠금)
├── camera_worker.py  카메라 워커 공통 틀 (팀원은 judge()만 채움)
├── streaming.py      MJPEG 영상 전송
├── db.py             SQLite: events, equipment, tickets, inspections, notes, feedback
├── rules.py          연동 규칙: 불량률 알림, 라인 연동 정지(컨베이어)
├── incidents.py      이벤트 → 사건(원인→정지→연쇄→복구) 묶기
├── analytics.py      타임라인 구간, OEE, 교대 보고서
├── twin_data.py      디지털 트윈 고도화: 레이어 · 리플레이 데이터
├── modules/          core(대시보드·트윈·분석·점검 API), gate, quality, safety, maintenance
├── templates/        base, dashboard, twin, gate_login, quality, safety, maintenance, feedback, timeline, oee, report, system
├── static/           style.css(v2 다크 테마 토큰), app.js(공통 함수), twin.js/css, process.css, incident.js
├── tools/check_model.py
├── model_files/      ★ 팀원 YOLO 모델 (.pt) — 삭제 금지
├── defect_images/    불량 판정 순간 사진 (검사번호.jpg, 최근 1000장만 자동 유지) — 디지털 트윈 불량함 메뉴에서 사용
└── smartline.db      ★ 운영 기록 (OEE·타임라인·리플레이의 원천) — 삭제 금지
```

화면은 1초마다 `/api/state`를 받아 갱신한다(`app.js`의 `poll`). 상태 문자열 → 색은 `app.js`의 `tone()`.
시간 표시 규칙: 걸린 시간은 1분 미만 "45초", 그 이상은 분만 "20분" (`app.js`의 `mins`).

### 실행
```powershell
cd D:\EM_model
uv run web/app.py          # http://localhost:5000  (Mac은 AirPlay 때문에 5001 사용)
```
안전 게이트가 로그인이라 카메라 없이 테스트할 때는 관리자 우회(ADMIN_PIN)로 들어간다.

---

### 화면 단순화 (발표용)
메뉴는 6개만 보인다: 관제 대시보드 · 디지털 트윈 · 품질 검사 · 설비 안전 제어 · 설비 보전 · OEE.
타임라인 · 교대 보고서 · AI 학습 페이지는 **메뉴에서만 숨겼고 코드와 주소는 살아 있다** (AI 학습은 설비 보전 페이지의 'AI 정확도 보기' 링크로 들어감).
디지털 트윈은 '⏪ 리플레이'만 보이고 레이어·가정 시뮬레이션 버튼은 숨겼다.
설정: `modules/core.py`의 `HIDE_PAGES`, `TWIN_FEATURES` 기본값 (config.py에 같은 이름으로 넣으면 덮어씀). 숨긴 기능의 코드는 지우지 말 것.

## 4. 절대 지키기

1. **삭제 전에 반드시 목록을 보여주고 사용자 승인을 받는다.** 한 번에 몰아서 지우지 말고 묶음별로.
2. 작업 시작 전에 되돌릴 수 있게 한다: git이 없으면 `git init` 후 현재 상태를 커밋(사용자 동의 후), 또는 폴더 백업.
3. **절대 삭제 금지**
   - `data/` 아래 원본·최종 시험·dev·field 데이터 전부 (특히 test_final 관련 파일과 lock 파일 — 최종 시험은 '봉인' 원칙)
   - `models/` 의 v5 최종 모델과 `v5_final.json`
   - `web/config.py`, `web/model_files/`, `web/smartline.db`
   - `docs/`, `prompts/`, `src/` 의 스크립트 (실험 재현 기록)
4. **동작과 화면 디자인은 바꾸지 않는다.** '최적화'는 안 쓰는 파일·코드 정리와 명백한 버그 수정까지. 발표가 코앞이라 구조를 크게 바꾸는 리팩터링은 하지 않는다.
5. 팀원 모듈의 설정값(클래스 이름 `PRODUCT_CLASS`, `DEFECT_CLASSES`, `DANGER_CLASSES`, `REQUIRED` 등)은 팀원 모델에 맞춘 것이라 바꾸지 않는다.
6. 주석은 지금처럼 쉬운 한국어로 유지한다.

---

## 5. 정리(삭제) 후보 — 확인 후 사용자에게 제안할 것

- 모든 `__pycache__/` 폴더 (언제 지워도 안전, 실행하면 다시 생김)
- `web/twin_update/` 폴더, `twin_update.zip` (디지털 트윈 업데이트를 덮어쓴 뒤 남은 것)
- 웹 백업 폴더·zip (`web_backup_v2`, `web_backup_v3` 등) — 사용자가 원하면 하나만 남기기
- 미리보기 이미지(`preview_*.png`, `twin_*.png`)가 프로젝트 안에 있다면 별도 폴더로
- 안 쓰는 import, 같은 import 중복 (예: `modules/core.py`에 `from flask import request`, `from state import ...`가 여러 줄로 나뉘어 있음)
- 더 이상 쓰지 않는 템플릿·정적 파일: 각 파일이 어디서 참조되는지 grep으로 확인 후 제안 (예: `gate.html`은 이미 삭제됨)

## 6. 알려진 문제

- **VS Code가 파일 이동 시 import 경로를 자동으로 바꾼 적이 있다.** `import web.twin_update.db as db` 같은 줄이 있으면 `import db`로 되돌린다.
  확인: `Select-String -Path web\*.py, web\modules\*.py -Pattern "twin_update"`
- `app.py` 첫 docstring에 `D:\EM_model`이 들어 있어 `SyntaxWarning: "\E"`가 뜬다 → docstring 앞에 `r`을 붙인다 (`r"""...`).
- `config.py`의 `EM_MODEL_DIR`이 EM_model 폴더를 정확히 가리켜야 설비 보전 분류기가 더미가 아닌 실제 모델로 뜬다
  (web이 EM_model 안에 있으면 `os.environ.get("EM_MODEL_DIR", os.path.dirname(BASE))` 권장).

## 7. 수정 후 점검 순서

1. 문법 확인: `uv run python -m py_compile web\*.py web\modules\*.py src\*.py`
2. 서버 실행 후 페이지 열기: `/`, `/twin`, `/quality`, `/safety`, `/maintenance`, `/timeline`, `/oee`, `/report`, `/feedback`, `/system`
3. API 확인: `/api/state`, `/api/twin/layers`, `/api/twin/replay`, `/api/oee`, `/api/system`
4. `/system`(시스템 점검) 페이지에서 카메라·모델·클래스 이름·분류기 상태 확인
5. 무엇을 지웠고 무엇을 고쳤는지 표로 보고