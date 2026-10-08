# legacy_v1 (이전 작업: em_model_v1)

원본 위치: D:\AI_prj · 정리 날짜: 2026-10-01
(원본 README가 인코딩 문제로 깨져 10/08에 다시 썼습니다.)

## 들어 있는 것

| 경로 | 내용 | v5에서의 용도 |
|---|---|---|
| models_v4/ | em_model_v1 최종 모델 기록 (기준 모델 + BiLSTM, 앙상블) | v1 vs v5 비교 실험 |
| data/gen/, data/train*.csv, data/val.csv | v1 학습 데이터 (Claude 생성, 옛 기준) | 같은 모델 구조로 데이터만 바꿔 비교 |
| data/test_llm.csv | GPT 59문장 (옛 기준 라벨) | 선택용 데이터 후보 (새 기준으로 다시 라벨 필요) |
| data/test_human.csv | Gemini 43 + Claude 11 (옛 기준 라벨) | 선택용 데이터 후보 (Claude 11 제외, 다시 라벨 필요) |
| src/ | v1 코드 | v5 코드 작성 때 참고 |

## 버전 이름 안내
- em_model_v1: 이전 프로젝트 전체
- 그 안의 v1~v4: em_model_v1 안의 실험 반복 번호 (models_v4 = 마지막 실험의 최종 모델)
- v5: 기준표 v2로 데이터를 처음부터 다시 만든 현재 버전 (상위 폴더)
