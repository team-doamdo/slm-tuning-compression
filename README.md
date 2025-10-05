# Smart Farm LLM Project

## 1. Current Status

- Sensor data (temperature, humidity, CO₂, etc.) are currently **manually monitored by humans**.  
- Goal: Run a **lightweight LLM (sLM) locally** so that AI can analyze and interpret data without direct human observation.  

---

## 2. Core Objectives

- Clearly define **what type of LLM to build**.  
- Not just storing data in a DB → the **LLM should directly process sensor data and return insights**.  
- Focus on building a **lightweight model first**, then explore service-oriented applications.  

---

## 3. Candidate Service Ideas

1. **Automated Report Generation**  
   - Example: “Show me how many greenhouses experienced unusual temperature fluctuations today.”  
   - Automatically create farming logs and periodic reports.  

2. **Anomaly Detection & Alerts**  
   - Example: “Greenhouse #10 shows abnormal temperature patterns. Please check.”  
   - AI identifies patterns that humans can easily miss.  

3. **Crop Growth Cycle Analysis**  
   - Analyze time-series data based on crop growth cycles (e.g., 2 weeks, 4 weeks).  
   - The model retains historical context for a given cycle, then **summarizes and stores insights**.  

---

## 4. Technical Considerations

- **Model Optimization / Compression**  
  - Techniques: Pruning, LoRA, quantization.  
  - Must be deployable on **Raspberry Pi or mobile devices**.  

- **Database Structure**  
  - Avoid plain SQLite.  
  - Use **time-series databases** better suited for sensor data.  

- **Prompt Engineering**  
  - Lightweight models may produce unreliable responses.  
  - Design **scenario-based prompts** to improve reliability.  

- **History Management**  
  - Support queries like: “Analyze the past 2 weeks.”  
  - Store historical data with **summarization and compression** for efficient memory usage.  

---

## 5. Execution Strategy

1. **Model Selection**  
   - Prioritize models with strong **Korean language support**.  
   - Good at handling **time-series analysis**.  
   - Candidates: LLaMA family, Gemma (via Hugging Face).  

2. **Data Preparation**  
   - Use **real-world or open datasets** for training and testing.  
   - Avoid unnecessary dummy data.  

3. **Experiment Workflow**  
   - Start with optimization & testing on PC.  
   - Validate deployment feasibility on **Raspberry Pi**.  
   - Consider **mobile deployment** as the final step.  

---

## 6. Project Direction

- The **primary goal is not building an app**, but creating a **meaningful lightweight LLM** for smart farming.  
- Applications include report generation, anomaly alerts, and time-series insights.  
- Evaluation criteria:  
  - Accuracy on test queries  
  - Effectiveness in anomaly detection  
  - Quality of auto-generated reports
 

## 7. Project Structure

**⚠️ 주의**: 프루닝된 모델은 레이어별로 크기가 다르므로 일반 Hugging Face 방식으로 로드 불가

→ **모델 로드** 시 반드시 `load_pruned_model()` 사용

```
smartfarm_pruning/
│
├── data/
│   ├── raw/                          # 원본 데이터 (6개 카테고리별 JSON 파일)
│   └── split/                        # 계층적 분할된 데이터셋 (카테고리 비율 유지)
│       ├── pruning_activation.json   # 활성화 기반 중요도 측정용 (40%)
│       ├── finetuning_lora.json      # LoRA 파인튜닝 학습용 (40%)
│       └── final_validation.json     # 최종 성능 평가용 (20%, 모든 실험 공통)
│
├── models/
│   ├── original/
│   │   └── gemma-3-1b-pt/            # Gemma-3 1B 원본 모델 
│   └── pruned/                       # 프루닝된 모델 저장소 (파인튜닝 팀 인계용)
│       ├── pruned_activation/        # Activation 기반 프루닝 모델
│       └── pruned_magnitude/         # Magnitude 기반 프루닝 모델
│
├── results/
│   ├── pruning/                                   # 프루닝 단계 결과물
│   │   ├── activation/                            # Activation 기반 프루닝 결과
│   │   │   ├── activation_heads_to_prune.json     # 제거 대상 Attention Head 목록
│   │   │   ├── activation_neurons_to_prune.json   # 제거 대상 FFN 뉴런 목록
│   │   │   └── activation_neuron_stats.json       # 뉴런 활성화 통계 (중간 저장)
│   │   │
│   │   ├── magnitude/                             # Magnitude 기반 프루닝 결과
│   │   │   ├── magnitude_heads_to_prune.json      # 제거 대상 Attention Head 목록
│   │   │   └── magnitude_neurons_to_prune.json    # 제거 대상 FFN 뉴런 목록
│   │   │
│   │   └── evaluation/                            # 프루닝 모델 성능 평가 결과
│   │       ├── eval_origin.json                   # 원본 모델 기준 성능
│   │       └── eval_{실험번호}.json                 # 실험별 성능
│   │
│   ├── finetuning/                   # 파인튜닝 단계 결과물 
│   │
│   └── quantization/                 # 양자화 단계 결과물 
│
├── src/
│   ├── pruning/                          # 프루닝 구현 코드
│   │   ├── activation/                   # Activation 기반 프루닝
│   │   │   ├── activation_attention.py   # Attention Head 중요도 측정 
│   │   │   ├── activation_ffn.py         # FFN 뉴런 중요도 측정 
│   │   │   └── activation_integrate.py   # 프루닝 실행 및 모델 저장 (Activation 통합)
│   │   │
│   │   ├── magnitude/                    # Magnitude 기반 프루닝
│   │   │   ├── magnitude_attention.py    # Attention Head 중요도 측정 
│   │   │   ├── magnitude_ffn.py          # FFN 뉴런 중요도 측정 
│   │   │   └── magnitude_integrate.py    # 프루닝 실행 및 모델 저장 (Magnitude 통합)
│   │   │
│   │   └── readme.md                 # 프루닝 방법론 및 실험 설정 문서
│   │ 
│   ├── finetuning/                   # 파인튜닝 구현 코드
│   │
│   ├── quantization/                 # 양자화 구현 코드 
│   │
│   └── utils/                        # 공통 유틸리티 (모든 단계에서 사용)
│       ├── model_loader.py           # 모델 로드/저장 (원본/프루닝/LoRA/양자화 모델 자동 감지)
│       ├── data_loader.py            # JSON 데이터 로드/저장 및 전처리
│       └── model_evaluator.py        # 통합 성능 평가 (BLEU/ROUGE/정확도/속도 측정)
│
├── scripts/
│   ├── split_data_stratified.py      # 데이터셋 계층적 분할 (카테고리별 비율 유지)
│   └── test_pruned_model.py          # 프루닝 모델 간단 검증 (생성 테스트)
│
├── requirements.txt                  # 프로젝트 의존성 패키지 목록
│
└── readme.md                         # 프로젝트 전체 개요 및 사용법

```


---

## License

This project is for **experimental and research purposes only**.  
