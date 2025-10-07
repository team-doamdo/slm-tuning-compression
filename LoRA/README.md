## **1. 개요**

본 문서는 Gemma-3-1B-IT 모델을 대상으로, 도메인 특화 데이터셋(tomato_data.json)을 활용하여 **LoRA (Low-Rank Adaptation) 기법**으로 어댑터를 학습한 과정을 기술한다.

LoRA의 목적은 **모델 파라미터 대부분을 고정시키고 일부 저랭크 행렬만 학습**함으로써, 적은 메모리와 연산으로 도메인 특화 성능을 개선하는 것이다.

실험은 M3 Pro(MPS) 환경에서 수행되었으며, CPU 대비 연산 효율은 낮지만 GPU 대비 구현 및 테스트가 용이하다.



## **2. 데이터셋 구성**

데이터는 JSON 형식으로 제공되며, 각 샘플은 다음과 같은 키를 포함한다.

| **Key**     | **Description**                |
| ----------- | ------------------------------ |
| instruction | 모델에 입력할 질문/지시문      |
| output      | 기대되는 모델 출력 (정답/응답) |

델 입력은 아래와 같은 템플릿을 사용하여 **질문-응답(prompt-response) 대화 형태**로 정규화한다.

Prompt: {instruction}
Answer: {output}



## **3. 실험 환경**

| **항목**    | **내용**                                                     |
| ----------- | ------------------------------------------------------------ |
| 모델        | google/gemma-3-1b-it                                         |
| 디바이스    | Apple M3 Pro (MPS)                                           |
| 프레임워크  | PyTorch 2.x, HuggingFace Transformers                        |
| LoRA 설정   | r=16, alpha=32, dropout=0.1, target_modules=[“q_proj”,“v_proj”,“k_proj”,“o_proj”,“up_proj”,“down_proj”,“gate_proj”] |
| 학습 배치   | 2 (MPS 환경에서 더 큰 배치는 메모리 초과 가능)               |
| 입력 길이   | max_length=512 (prompt), max_length=256 (response)           |
| 학습 epoch  | 3                                                            |
| 학습률      | 3e-4                                                         |
| 데이터 정리 | DataCollatorForSeq2Seq 사용, 자동 padding                    |

| **항목**    | **내용**                                                     |
| ----------- | ------------------------------------------------------------ |
| 모델        | google/gemma-3-1b-it                                         |
| 디바이스    | Apple M3 Pro (MPS)                                           |
| 프레임워크  | PyTorch 2.x, HuggingFace Transformers                        |
| LoRA 설정   | r=16, alpha=32, dropout=0.1, target_modules=[“q_proj”,“v_proj”,“k_proj”,“o_proj”,“up_proj”,“down_proj”,“gate_proj”] |
| 학습 배치   | 2 (MPS 환경에서 더 큰 배치는 메모리 초과 가능)               |
| 입력 길이   | max_length=512 (prompt), max_length=256 (response)           |
| 학습 epoch  | 3                                                            |
| 학습률      | 3e-4                                                         |
| 데이터 정리 | DataCollatorForSeq2Seq 사용, 자동 padding                    |

## **4. LoRA 적용 방법**

LoRA는 **기존 모델의 대부분 weight를 고정**하고, 일부 Linear layer에 **저랭크 가중치 행렬(Lora weight)**만 학습한다.

1. 프루닝 모델(또는 기본 Gemma-3-1B 모델) 로드
2. LoRA Config 정의: 학습할 레이어(target_modules)와 rank(r), alpha, dropout 설정
3. get_peft_model 함수를 통해 LoRA 어댑터를 모델에 결합
4. 학습 가능한 파라미터 확인 (model.print_trainable_parameters())

이 방식으로 전체 파라미터의 약 1~2%만 학습 가능, 나머지는 고정되어 효율적 학습 가능



## **5. 학습 프로세스**

1. JSON 데이터셋 로드 후, prompt-response 형태로 변환

2. tokenizer를 사용하여 input_ids와 labels 생성

3. DataCollatorForSeq2Seq를 활용하여 **자동 패딩 및 tensor 변환**

4. Trainer 객체를 사용하여 학습 수행

   - gradient_accumulation_steps=8로 작은 배치를 사실상 큰 배치처럼 처리
   - fp16=False (MPS/CPU 환경에서는 mixed precision 지원 제한)

5. 학습 결과 LoRA 가중치만 저장 (model.save_pretrained("./lora_tomato"))

   

## 6. 학습 결과

- 학습 완료 후 LoRA 어댑터는 ./lora_tomato 폴더에 저장
- 학습 가능 파라미터 수: 약 11,038,496 (~1.2% of total)
- 데이터셋 샘플 수: 4,902
- 학습 시간: M3 Pro 환경에서 약 1시간 30분 (epoch=3 기준)
- loss 및 학습 로그를 통해 학습 진행 상황 확인 가능

## 7. 결론 

- LoRA 기법을 통해 Gemma-3-1B 모델을 **도메인 특화 데이터셋**에 맞게 효율적으로 적응 가능
- 전체 모델 파라미터를 학습하지 않고, 작은 저랭크 행렬만 업데이트하여 **메모리와 연산 효율** 극대화
- MPS 환경에서도 실험 가능하지만, 대규모 학습은 CUDA GPU 환경 권장

