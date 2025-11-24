# 터미널 켜서 실행해야함. 실행 방법은 아래 커맨드 차례대로 입력
# cd /home/rasberry/python
# source venv/bin/activate
# python3 a.py

import ollama
import time
import json
from typing import List, Dict, Any

# === 설정 부분 ===
MODEL_NAME = "1" 
INPUT_FILE = "tomato_dataset_test.json"
OUTPUT_FILE = "output_results.json"
# ==================

def load_prompts(file_path: str) -> List[str]:
    """JSON 파일에서 프롬프트 리스트를 불러옵니다."""
    instructions: List[str] = []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    instruction = item.get("instruction")
                    if instruction:
                        instructions.append(instruction)
            else:
                print("⚠️ 오류: JSON 파일의 최상위 구조가 리스트가 아닙니다.")
                
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{file_path}'을 찾을 수 없습니다. 파일을 생성했는지 확인하세요.")
    except json.JSONDecodeError:
        print(f"오류: 입력 파일 '{file_path}'의 JSON 형식이 잘못되었습니다.")
        
    return instructions

def save_results(file_path: str, results: List[Dict[str, Any]]):
    """테스트 결과를 JSON 파일로 저장합니다."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False로 설정하여 한글이 깨지지 않도록 합니다.
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n모든 테스트 결과가 '{file_path}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")

def run_ollama_tests(model: str, prompt_list: List[str]) -> List[Dict[str, Any]]:
    """
    Ollama 테스트를 실행하고, 질문, 답변, 추론 시간을 측정하여 딕셔너리 리스트로 반환합니다.
    """
    test_results: List[Dict[str, Any]] = []

    if not prompt_list:
        print("프롬프트 목록이 비어 있어 테스트를 실행하지 않습니다.")
        return test_results

    print(f"--- Ollama 모델: {model} 테스트 시작 (총 {len(prompt_list)}개 프롬프트) ---")
    
    try:
        client = ollama.Client()
    except Exception as e:
        print(f"⚠️ Ollama 클라이언트 초기화 오류: {e}")
        return test_results

    for i, prompt in enumerate(prompt_list):
        print(f"\n[{i+1}/{len(prompt_list)}] 질문: {prompt}")

        start_time = time.time()
        
        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                stream=False
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            answer = response['response'].strip()

            # 결과를 딕셔너리 형태로 저장
            test_results.append({
                "id": i + 1,
                "question": prompt,
                "answer": answer,
                "inference_time_sec": round(inference_time, 4)
            })
            
            answer_preview = answer[:50].replace('\n', ' ')
            if len(answer) > 50:
                 answer_preview += '...'
            
            print(f"✅ 답변 완료: {answer_preview} (추론 시간: {inference_time:.4f}초)")

        except Exception as e:
            error_message = f"Ollama 통신 오류: {e}"
            print(f"⚠️ {error_message}")
            test_results.append({
                "id": i + 1,
                "question": prompt,
                "answer": error_message,
                "inference_time_sec": 0.0
            })

    return test_results

# 메인 실행 블록
if __name__ == "__main__":
    # 1. JSON 파일에서 프롬프트 불러오기
    prompts_to_run = load_prompts(INPUT_FILE)
    
    # 2. Ollama 테스트 실행
    final_results = run_ollama_tests(MODEL_NAME, prompts_to_run)
    
    # 3. 결과를 JSON 파일로 저장
    save_results(OUTPUT_FILE, final_results)
    
    print("\n테스트 스크립트 실행 완료.")