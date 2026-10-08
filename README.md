# SMART LINE 모델 기록

[SMART LINE MES](https://github.com/kyc4039/smartline-mes)에 들어간 AI 모델의 학습 과정 기록입니다.
나중에 다시 쓰거나 개선할 때 확인하는 용도라서, 실행 코드 · 데이터 · 프롬프트 · 결과 지표 · 보고서를 남깁니다.
학습된 가중치 파일은 크기가 크고 아직 개선 중이라 올리지 않았습니다.

| 폴더 | 모델 | 담당 | 기록 위치 |
|---|---|---|---|
| [maintenance/](maintenance/) | ④ 설비보전: 고장 보고 문장 → 담당 분야 · 조치 시점 (TF-IDF · BiLSTM · KLUE 앙상블) | 김영창 | 이 저장소 |
| — | ① 안전게이트 (YOLOv8n) | 김태원 | 팀 구글 드라이브 |
| — | ② 프레스 안전 (YOLO11s) | 유장오 | 팀 구글 드라이브 |
| — | ③ 품질검사 (YOLOv8) | 백임수 | 팀 구글 드라이브 |

## 설비보전 AI 한눈에

- 데이터: 기준표 v2로 상황 300개를 설계 → 말투 4종으로 문장화 → 학습 1,020 / 검증 180
- 평가: GPT · Gemini · 직접 작성 92문장 최종 시험 (학습 생성기와 다른 생성기로 만들고 마지막에 한 번만 평가)
- 결과와 과정: [maintenance/reports/설비보전AI_학습보고서.pdf](maintenance/reports/설비보전AI_학습보고서.pdf), [maintenance/docs/실험기록.md](maintenance/docs/실험기록.md)

## 실행

```bash
cd maintenance
uv sync
uv run python src/predict.py        # 학습된 모델이 models/에 있을 때
```

웹(MES)에서 쓰려면 smartline-mes에서 `uv run python tools/export_maint_model.py`로
`maintenance/models`의 판정용 파일을 `smartline-mes/model_files/maintenance`로 복사합니다.
