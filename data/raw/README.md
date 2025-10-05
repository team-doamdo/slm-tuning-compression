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


## 4. **subcategory** (필수)

### disease_management
- `diagnosis` - 병해충 진단
- `treatment` - 치료 방법
- `prevention` - 예방 전략
- `pest_control` - 해충 관리
- `organic` - 유기농 방제

### environmental_control
- `temperature` - 온도 관리
- `humidity` - 습도 제어
- `light` - 광 관리
- `co2` - CO2 조절
- `ventilation` - 환기 시스템

### equipment_sensors
- `sensor_setup` - 센서 설치
- `calibration` - 센서 보정
- `maintenance` - 장비 유지보수
- `automation` - 자동화 시스템

### troubleshooting
- `physiological` - 생리장해
- `stress` - 환경 스트레스
- `emergency` - 긴급 상황
- `diagnosis` - 문제 진단

### sensor_status_query
- `temperature_check` - 온도 상태
- `humidity_check` - 습도 상태
- `co2_check` - CO2 상태
- `light_check` - 조도 상태

### sensor_alert_with_query
- `temp_alert` - 온도 경고
- `humidity_alert` - 습도 경고
- `co2_alert` - CO2 경고
- `light_alert` - 조도 경고

---

## 5. **source** (필수)

공신력 있는 출처 명시:
- `"Cornell Extension"`
- `"USDA Plant Disease Guide"`
- `"UC Davis Research"`
- `"Wageningen University"`
- `"Journal of Agricultural Engineering 2024"`

---

## 데이터 예시

### 카테고리 1-4 예시
```json
{
  "instruction": "[STATUS: OK] My tomato leaves have brown spots with yellow halos. What should I do?",
  "output": "[STATUS: OK] This appears to be early blight caused by Alternaria solani. Remove affected leaves immediately, improve air circulation, and apply copper-based fungicide every 7-10 days.",
  "category": "disease_management",
  "subcategory": "diagnosis",
  "source": "Cornell Extension"
}
```

### 카테고리 5 예시
```json
{
  "instruction": "[STATUS: TEMP_HIGH(35)] What immediate actions should I take?",
  "output": "[WARNING: TEMP_HIGH(35). Immediately increase ventilation and activate cooling systems.] Provide shading or apply misting to rapidly reduce greenhouse temperature.",
  "category": "sensor_status_query",
  "subcategory": "temperature_check",
  "source": "USDA Guidelines"
}
```

### 카테고리 6 예시
```json
{
  "instruction": "[STATUS: HUM_HIGH(95)] When should I start pruning tomato suckers?",
  "output": "[WARNING: HUM_HIGH(95). Reduce irrigation and activate dehumidifiers immediately.] Start when suckers are 5-7cm long, remove all below first flower cluster.",
  "category": "sensor_alert_with_query",
  "subcategory": "humidity_alert",
  "source": "UC Davis Research"
}
```

---

## 토마토 스마트팜 정상 범위 참고값

### Temperature
- **정상 범위**: 20-25°C
- **출처**: 농촌진흥청 작물 재배 가이드, Deep learning-driven IoT solution for smart tomato farming

### Humidity
- **정상 범위**: 65-75%
- **출처**: Deep learning-driven IoT solution for smart tomato farming 논문

### CO2
- **정상 범위**: 800-1000ppm

### Light
센서에서는 lux 단위로 출력(순간 광도) → DLI(하루 총광량)로 환산해서 정상 범위 여부 판단

#### 순간 광도 (Lux 기준)
- **하한선**: 15,000 lux (이상이어야 최소 생육 유지)
- **권장치**: 45,000-70,000 lux (맑은 날 직사광 수준, DLI 충족에 유리)

#### DLI (Daily Light Integral)
- **정상 범위**: 15-20 mol/m²/day
