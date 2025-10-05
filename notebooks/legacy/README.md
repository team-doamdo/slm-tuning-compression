
---

## 📓 노트북 설명
### `fine_tuning_pruning.ipynb`
#### 주요 기능
1. **모델 로드**
   - `google/gemma-3-1b-pt` 기반 언어 모델  
   - `transformers`, `torch` 사용  

2. **Baseline 추론 (`quick_infer`)**
   - CPU 환경에서 토큰 생성 시간 측정  
   - 프루닝 전후 성능 비교를 위한 baseline 제공  

3. **구조적 프루닝 (`structured_prune_mlp`)**
   - 각 레이어 MLP projection weight norm 기반 중요도 평가  
   - Top-k 방식으로 뉴런 선택 후 선형 계층 교체  
   - keep ratio(예: 0.7) 조절 가능  

4. **LoRA 기반 파인튜닝**
   - [PEFT](https://huggingface.co/docs/peft) 활용  
   - `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj` 대상 모듈에 LoRA 적용  
   - Hugging Face `Trainer` API + W&B 로깅 지원  

5. **데이터셋 로드**
   - 커스텀 JSONL 파일(`dataset.jsonl`) → train/eval split (80:20)  
   - Instruction-following 형식: `"Q: ...\nA: ..."`  

6. **학습 관리**
   - `TrainingArguments`로 배치 사이즈, gradient accumulation, learning rate, epochs, early stopping 등 설정  
   - Best 모델 저장 및 `wandb` 실험 추적  

---

## 📑 데이터셋 설명
### `dataset.jsonl`
- Instruction/Output 쌍으로 이루어진 커스텀 Tomato QA 데이터셋 (3000개 샘플)  
- 예시:
```json
{"instruction": "What is the ideal daytime temperature for tomato plants?", "output": "The ideal daytime temperature is around 21-27°C (70-80°F)."}
