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
slm-tuning-compression
├── data
│   └── (학습한 데이터셋)
├── results
│   └── (모델 평가/실험 결과)
├── scripts
│   
├── src
│   ├── 프루닝, 로라, 양자화 진행코드
│   │   
│   ├── model_test
│   │   └── (LoRA, 프루닝, 양자화 모델 테스트 코드)
│   └── utils
│       └── (모델 로더, 공용 유틸 코드)
├── README.md
├── requirements.txt
└── .gitignore

```


---

## License

This project is for **experimental and research purposes only**.  
