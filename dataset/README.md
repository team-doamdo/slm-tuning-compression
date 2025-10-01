# 토마토 스마트팜 SLM 데이터셋 구축 지침

## json 파일 규칙

```json
{
  "instruction": "[STATUS: ] 질문 (1-2문장)",
  "output": "[STATUS: ] 답변 (1-2문장)",
  "category": "카테고리명",
  "subcategory": "세부카테고리명",
  "source": "출처"
}
```

## 데이터 필드 상세 규격

### 1. **instruction** (필수)

센서 상태 접두사가 포함된 사용자 질문 - 1-2문장


**카테고리 1-4 (Disease, Environmental, Equipment, Troubleshooting)**

- 모든 질문 앞에 `[STATUS: OK]` 추가
- 예시
    - `"[STATUS: OK] My tomato leaves have brown spots with yellow halos. What should I do?”`
    - `"[STATUS: OK] When should I start pruning tomato suckers?"`

**카테고리 5 (Sensor Status Query)**

- 다양한 센서값 포함 (정상/비정상 모두)
- 형식: `[STATUS: METRIC_STATUS (actual_value vs threshold)]`
- 예시
    - `"[STATUS: TEMP_HIGH(35)] Current greenhouse conditions?"`
    - `"[STATUS: HUM_LOW(35)] What adjustments needed?"`

**카테고리 6 (Sensor Alert with Query)**

- 센서 이상값 + 원래 질문
- 예시
    - `"[STATUS: TEMP_HIGH(35)] My tomato leaves have brown spots. What should I do?”`
    - `"[STATUS: CO2_LOW(250)] How much water do tomato plants need?"`

---

### 2. **output** (필수)

AI 답변 - 구체적 수치/방법 포함

**카테고리 1-4 (일반 답변):**

```
"[STATUS: OK] This appears to be early blight caused by Alternaria solani. Remove affected leaves immediately, improve air circulation, and apply copper-based fungicide every 7-10 days."
```

**카테고리 5 (센서 상태 답변):**

```
"[WARNING: TEMP_HIGH(35). Immediately increase ventilation and activate cooling systems.] Provide shading or apply misting to rapidly reduce greenhouse temperature."
```

**카테고리 6 (경고 + 원래 답변):**

```
"[WARNING: TEMP_HIGH(35). Immediately increase ventilation and activate cooling systems.] This appears to be early blight. Apply copper fungicide every 7-10 days."
```

**[ 센서 이상 패턴 - 카테고리 5,6 ]**

| 센서 유형 | 상태 코드 | 형식 예시 |
| --- | --- | --- |
| Temperature High | `TEMP_HIGH` | `[STATUS: TEMP_HIGH(35)]` |
| Temperature Low | `TEMP_LOW` | `[STATUS: TEMP_LOW(10)]` |
| Humidity High | `HUM_HIGH` | `[STATUS: HUM_HIGH(95)]` |
| Humidity Low | `HUM_LOW` | `[STATUS: HUM_LOW(35)]` |
| CO2 High | `CO2_HIGH` | `[STATUS: CO2_HIGH(2000)]` |
| CO2 Low | `CO2_LOW` | `[STATUS: CO2_LOW(150)]` |
| Light Low | `LIGHT_LOW` | `[STATUS: LIGHT_LOW(10000)]` |

**[ 상황별 응답 예시 ]**

- 정상 범위
    - temperature: 22-24
    - humidity: 60-70
    - CO2: 800-1000ppm
    - light: 45,000–70,000 lux
1. Temperature High
    
    `[WARNING: TEMP_HIGH({current_temperature}). Immediately increase ventilation and activate cooling systems.]`
    
2. Temperature Low
`[WARNING: TEMP_LOW({current_temperature}). Activate heating systems and close ventilation to maintain optimal temperature.]`
3. Humidity High
    
    `[WARNING: HUM_HIGH({current_humidity}). Reduce irrigation and activate dehumidifiers immediately.]`
    
4. Humidity Low
    
    `[WARNING: HUM_LOW({current_humidity}). Increase irrigation and activate misting systems.]`
    
5. CO2 High
    
    `[WARNING: CO2_HIGH({current_CO2}). Increase ventilation to lower CO2 concentration.]`
    
6. CO2 Low
    
    `[WARNING: CO2_LOW({current_CO2}). Activate CO2 supplementation system immediately to maintain photosynthesis.]`
    
7. Light Low
    
    `[WARNING: LIGHT_LOW({current_light}). Activate supplemental lighting immediately to prevent stretching and yield loss.]`
    

---

### 3. **category** (필수)

**카테고리별 분배**

1. **Disease Management** (질병 및 병해충 관리) - 1,500개
2. **Environmental Control** (환경 제어) - 1,250개
3. **Equipment & Sensors** (장비 및 센서 관리) - 750개
4. **Troubleshooting** (문제해결 및 진단) - 500개
5. **Sensor Status Query** (센서 상태 질문) - 500개
6. **Sensor Alert with Query** (센서 경고 + 질문) - 500개

6개 카테고리 중 택 1

| # | 카테고리명 | 개수 | 센서값 형태 | 설명 |
| --- | --- | --- | --- | --- |
| 1 | Disease Management | 1,500 | `[STATUS: OK]` | 병해충 관리 |
| 2 | Environmental Control | 1,250 | `[STATUS: OK]` | 환경 제어 |
| 3 | Equipment & Sensors | 750 | `[STATUS: OK]` | 장비/센서 |
| 4 | Troubleshooting | 500 | `[STATUS: OK]` | 문제해결 |
| 5 | **Sensor Status Query** | **500** | 이상값 | **센서 상태 질문**  |
| 6 | **Sensor Alert with Query** | **500** | **이상값** | **센서 경고 + 질문** |
