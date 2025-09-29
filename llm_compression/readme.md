## 📌 코드 구성

1. **환경 설정**
   - `bitsandbytes`, `transformers`, `peft`, `accelerate`, `wandb` 설치
   - Hugging Face & W&B 로그인 (환경변수 사용)
   - Google Drive 마운트

2. **모델 로드**
   - `google/gemma-3-1b-pt` 불러오기
   - `AutoModelForCausalLM`, `AutoTokenizer` 사용

3. **프루닝 적용**
   - MLP 레이어(`gate_proj`, `up_proj`, `down_proj`) 가중치 10% 비구조적 프루닝

4. **LoRA 적용**
   - MLP 레이어에 LoRA 어댑터 추가

5. **데이터셋 로드 & 전처리**
   - `tomato_qa_3000.txt` JSON 데이터셋 로드
   - 학습/테스트 데이터 분리
   - 질문-답변 형식으로 토큰화

6. **학습**
   - batch size 8, accumulation step 2 적용
   - learning rate 5e-5
   - 5 epoch 학습 진행

7. **모델 저장**
   - `/content/drive/MyDrive/results` 에 저장

8. **추론 성능 비교**
   - 원본 모델 vs 프루닝+LoRA 모델 추론 시간 측정
   - 동일 입력에 대한 응답 출력 비교