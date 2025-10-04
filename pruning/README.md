# Gemma-3-1B 모델 기반 데이터 중요도 기반 프루닝 실험 문서

## 1. 개요

본 문서는 Google의 `Gemma-3-1B-IT` 모델을 대상으로, 도메인 특화 데이터셋(`tomato.json`)을 활용하여 **gradient 기반 중요도 계산을 통한 L1 Unstructured Pruning**을 수행하는 과정을 기술한다.  

프루닝의 목적은 모델 파라미터의 일부를 제거함으로써 메모리 및 추론 속도를 개선하는 것이다. 본 코드는 M3 Pro(MPS) 환경에서 실행되었으나, CUDA GPU 환경 대비 연산 효율이 낮으므로 실행 시간이 크게 소요될 수 있다.

---

## 2. 데이터셋 구성

데이터는 JSON 형식으로 제공되며, 각 샘플은 `instruction`과 `output` 키를 포함한다.  

모델 입력은 아래와 같은 템플릿을 사용하여 대화 형태로 정규화한다.

\---

## 3. 실험 환경

## - **모델**: google/gemma-3-1b-it  

- 디바이스**: Apple M3 Pro (Metal Performance Shaders, MPS)  **
- 프레임워크**: PyTorch 2.x, HuggingFace Transformers  **
- Batch Size**: 기본값 1 (MPS 환경에서 2 이상 시 메모리 초과 발생)  **
- 최대 입력 길이 (max_length)**: 256  

## 4. 중요도 계산 방법

각 Linear layer의 `weight` 파라미터에 대해 gradient의 절대값을 누적하여 중요도를 정의한다.  

절차는 다음과 같다:

1. 데이터셋을 batch 단위로 입력하여 모델의 loss를 계산한다.
2. `loss.backward()`를 통해 gradient를 얻는다.
3. 각 weight 파라미터의 gradient 절대값을 중요도로 축적한다.
4. 전체 데이터셋 순회 후, 파라미터별 중요도 지도가 완성된다.

**주의**: backward 연산 시 활성화 값(activation)이 모두 저장되므로, batch size 증가 시 메모리 사용량이 기하급수적으로 늘어난다.

---

## 5. 프루닝 기법

프루닝은 `torch.nn.utils.prune`의 `global_unstructured`를 사용하였다.  

- **방법**: L1 Unstructured Pruning  

- **대상**: 모든 Linear layer의 weight  

- **제거율**: 30% (사용자가 `pruning_ratio`로 조정 가능)

프루닝 후에는 `prune.remove()`를 통해 mask를 제거하고, 파라미터를 실제로 축소한다.