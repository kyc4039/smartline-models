# 설비 보전 AI v5: 데이터 재구축

v4까지의 실험에서 확인한 데이터 문제(라벨 먼저 생성 → 말투가 긴급도를 대신함, 단일 생성기, 기준 변경, 현장 어휘 부족)를
해결하기 위해 **기준표 v2**로 데이터를 처음부터 다시 만든다.

## 폴더 구조

```
maintenance_v5/
├── docs/       기준표, 실험 기록
├── prompts/    단계별 생성 프롬프트
├── data/
│   ├── raw/    생성기에서 받은 원본 (수정하지 않음)
│   └── ...     정제·분할된 데이터
├── src/        코드
└── models/     학습된 모델 (버전별 하위 폴더)
```

## 데이터셋 구성

| 데이터셋 | 생성기 | 분량 | 용도 |
|---|---|---|---|
| 최종 시험 | GPT 36 + Gemini 36 + 직접 작성 20 | 92문장 | 마지막에 한 번만 평가 |
| 선택용 (dev) | GPT 36 + Gemini 36 (최종 시험과 같은 방식으로 새로 생성) | 72문장 | 모델·설정 선택 |
| 학습용 상황 | Claude | 약 300개 | 상황 속성 → 규칙으로 라벨 계산 |
| 학습용 문장 | Claude | 약 1,200문장 | 상황 × 말투, 15%는 상황 단위로 val 분리 |

**원칙: 학습 생성기(Claude)는 평가 데이터에 쓰지 않는다.**

## 진행 순서

| 단계 | 내용 | 산출물 | 상태 |
|---|---|---|---|
| 1 | 기준표 v2 확정 | `docs/기준표_v2.md` | 완료 |
| 2 | 최종 시험 생성 후 봉인 | `data/test_final.csv` (72문장) | 완료 |
| 2-1 | 직접 작성 20문장 (학습 데이터를 보기 전에) | `data/test_final.csv`에 포함 (총 92문장) | 완료 |
| 3 | 선택용 데이터 생성 (최종 시험과 같은 방식, 새 대화창) | `data/dev.csv` (72문장) | 완료 |
| 4 | 현장 용어 목록 | `docs/현장용어.md` | 완료 |
| 5 | 학습용 상황 설계 + 라벨 자동 계산 (`prompts/02`, `src/build_scenarios.py`, `src/check_scenarios.py`) | `data/scenarios.csv` (300개) | 완료 |
| 6 | 학습용 문장화 (`src/make_batches.py`, `prompts/03`) | `data/raw/train_01~12.csv`, `train_fix.csv` | 완료 |
| 7 | 품질 검사, 상황 단위 분할 (`src/build_train.py`) | `data/train.csv` 1,020 / `data/val.csv` 180 | 완료 |
| 8 | 학습 → dev 선택 → 최종 시험 1회 평가, v1과 비교 | `models/v5_*` | 진행 |

## 최종 시험 봉인 규칙

- 생성 후 **형식만 확인**하고 문장은 읽지 않는다.
- 학습 데이터 생성(6단계)이 끝난 뒤 라벨을 검수한다.
- 모델 선택에는 절대 사용하지 않고, 8단계 마지막에 **한 번만** 평가한다.

## 직접 작성 20문장 가이드 (2-1단계)

- 학습 데이터와 최종 시험 AI 문장을 **보기 전에** 쓴다.
- 담당 분야 4 × 조치 시점 3 조합에서 골고루 (조합당 1~2문장)
- 기준표 v2로 속성과 라벨을 직접 붙인다.
- AI에게 쓰게 하거나 다듬게 하지 않는다. 휴대폰 음성 입력을 쓰면 자연스러운 메모가 된다.
- 형식은 AI 시험 데이터와 같게 하고 style 열은 비워도 된다.

## 8단계 실행 순서

```powershell
uv add kiwipiepy scikit-learn tensorflow torch transformers joblib
uv run python src/train_baseline.py
uv run python src/train_bilstm.py
uv run python src/train_klue.py          # CPU에서 약 10분, 첫 실행 시 모델 다운로드(약 450MB)
uv run python src/select_dev.py          # dev로 조합 선택 → models/v5_final.json
uv run python src/final_test.py          # 최종 시험 단 한 번
```

선택 규칙 (2026-10-01 확정): 즉시 조치 탐지율 ≥ 0.9 → 조치 시점 정확도 → 오경보 적음 → 담당 정확도