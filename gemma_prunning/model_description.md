# LoRA 기반 모델 경량화 파이프라인

## 개요

이 프로젝트는 LoRA(Low-Rank Adaptation) 기법을 활용하여 대규모 언어 모델을 효율적으로 파인튜닝하고, 점진적 프루닝을 통해 모델을 경량화하는 파이프라인입니다.

## 주요 특징

- Apple Silicon (MPS) 최적화: M1/M2/M3 칩 GPU 가속 지원
- 메모리 효율적: LoRA를 통한 최소한의 파라미터만 학습
- 점진적 프루닝: 2%~20%까지 세밀한 단계별 경량화
- 복구 훈련: 프루닝 후 성능 회복 메커니즘
- 종합 평가: 시간/메모리/정확도 자동 측정

## 파이프라인 구조

전체 프로세스는 4단계로 구성됩니다:

1. STEP 1: LoRA 파인튜닝
2. STEP 2: 점진적 프루닝 (2%, 5%, 8%, 12%, 15%, 20%)
3. STEP 3: 복구 훈련
4. STEP 4: 모델 성능 비교

## 단계별 실행
- LoRA 파인튜닝
python prunning_model.py 1

- 점진적 프루닝
python prunning_model.py 2

- 복구 훈련
python prunning_model.py 3

# 모든 모델 평가
python prunning_model.py eval all