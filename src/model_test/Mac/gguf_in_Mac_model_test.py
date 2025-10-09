import json
import subprocess
import time
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# ---------- 설정 ----------
MODEL_NAME = "run s_q5:latest"
DATASET_JSON = "/Users/tenedict/Desktop/p_teest/final_test_validation_dataset.json"  # 데이터셋 경로
TIMEOUT_SEC = 2  # 2초 제한

# ---------- Ollama 실행 함수 ----------
def query_ollama(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("→ Timeout! Skipping this question.")
        return None  # None 반환 → 실패로 처리
    except Exception as e:
        print("→ Ollama query error:", e)
        return None

# ---------- 데이터 불러오기 ----------
with open(DATASET_JSON, "r") as f:
    dataset = json.load(f)

# ---------- BLEU, ROUGE 초기화 ----------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

total_bleu = 0.0
total_rouge1 = 0.0
total_rouge2 = 0.0
total_rougeL = 0.0
success_count = 0
keyword_match_total = 0
total_time = 0.0
count_time_included = 0  # 2초 이하로 완료된 케이스만 포함

# ---------- 평가 반복 ----------
for idx, item in enumerate(dataset, 1):
    question = item["instruction"]
    reference = item["output"]

    start = time.time()
    answer = query_ollama(question)
    end = time.time()
    
    elapsed = end - start
    if answer is not None:
        success = bool(answer.strip())
        if success:
            success_count += 1
        
        # 키워드 매치
        ref_keywords = set(reference.lower().split())
        ans_keywords = set(answer.lower().split())
        keyword_match = len(ref_keywords & ans_keywords) / max(1, len(ref_keywords))
        keyword_match_total += keyword_match
        
        # BLEU
        bleu_score = sentence_bleu([reference.split()], answer.split())
        total_bleu += bleu_score
        
        # ROUGE
        scores = scorer.score(reference, answer)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
        
        # 2초 이하 응답시간만 합산
        if elapsed <= TIMEOUT_SEC:
            total_time += elapsed
            count_time_included += 1
    else:
        success = False
        bleu_score = 0.0
        scores = {'rouge1': type('', (), {'fmeasure': 0.0})(),
                  'rouge2': type('', (), {'fmeasure': 0.0})(),
                  'rougeL': type('', (), {'fmeasure': 0.0})() }
        keyword_match = 0.0
    
    # ---------- 진행 상황 출력 ----------
    print(f"[{idx}/{len(dataset)}] Question: {question}")
    print(f"Answer: {answer}")
    print(f"BLEU: {bleu_score:.4f}, ROUGE1: {scores['rouge1'].fmeasure:.4f}, ROUGE2: {scores['rouge2'].fmeasure:.4f}, ROUGEL: {scores['rougeL'].fmeasure:.4f}")
    print(f"Keyword match: {keyword_match:.4f}, Success: {success}, Time: {elapsed:.2f}s\n")

# ---------- 최종 요약 ----------
dataset_size = len(dataset)
summary = {
    "summary": {
        "dataset_size": dataset_size,
        "success_rate": success_count / dataset_size,
        "exact_match_accuracy": 0.0,
        "keyword_match_score": keyword_match_total / dataset_size,
        "bleu_rouge": {
            "bleu": total_bleu / dataset_size,
            "rouge1": total_rouge1 / dataset_size,
            "rouge2": total_rouge2 / dataset_size,
            "rougeL": total_rougeL / dataset_size
        },
        "avg_response_time": (total_time / count_time_included) if count_time_included > 0 else 0.0,
        "model_parameters": 855362944,
        "model_memory_mb": 3294.0
    }
}

# ---------- 결과 저장 ----------
with open("eval_gguf_in_Mac_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("========== Evaluation Complete ==========")
print(json.dumps(summary, indent=2))