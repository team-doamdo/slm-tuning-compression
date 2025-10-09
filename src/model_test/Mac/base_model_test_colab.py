from huggingface_hub import login
import psutil, os
import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from google.colab import files

# Hugging Face 로그인
hf_token = ""
login(token=hf_token)
print("✅ Hugging Face 로그인 완료")

# --------------------------
# 설정
# --------------------------
models = [
    # "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",  # 더 안정적인 TinyLlama 모델
    # "google/gemma-3-1b-it",  # 가벼운 대화형 모델
    # "microsoft/Phi-4-mini-instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct", #베이스 모델
    # "deepseek-ai/deepseek-coder-1.3b-instruct"
]

prompts = [
    "What is 28 + 93?",
    "Which country has the largest population?",
    "If you put a plant in a cave with no sunlight at all, why would it not survive for long?",
    "Context: On January 10, 2025, the New York City Metropolitan Transportation Authority (MTA) announced that it will add 265 new zero-emission electric buses to city bus routes. Task: Summarize in one sentence.",
    "2, 4, 8, 16, ? What is the next number?",
    "If January 1 is Monday and January 8 is also Monday, what day is January 15?",
    "The following temperature sensor readings: [22, 23, 22, 45, 23, 22]. Which value is an outlier?",
    "Here are the test scores: [85, 88, 90, 300, 87, 89]. Which score is an outlier?",
    "Please write a short poem.",
    "Rewrite this sentence for a child: 'The Earth revolves around the Sun, causing seasons.'"
]

output_file = "model_eval_results.txt"

# --------------------------
# 함수
# --------------------------
def measure_memory():
    return psutil.virtual_memory().used / (1024 ** 2)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_safely(model_name):
    try:
        print(f"🔄 모델 로딩 중: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model, tokenizer
    except Exception as e:
        print(f"❌ 모델 로딩 실패 {model_name}: {str(e)}")
        return None, None

def test_model(model_name, model, tokenizer, prompts, f):
    f.write(f"\n{'='*50}\n모델: {model_name}\n{'='*50}\n\n")
    print(f"\n{'='*50}\n모델: {model_name}\n{'='*50}\n")

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=150,
        pad_token_id=tokenizer.pad_token_id
    )

    total_time = 0
    successful_tests = 0

    for i, prompt in enumerate(prompts, 1):
        f.write(f"\n--- 프롬프트 {i} ---\n질문: {prompt}\n")
        print(f"\n--- 프롬프트 {i} ---\n질문: {prompt}\n")

        try:
            start_mem = measure_memory()
            start_time = time.time()
            output = gen_pipeline(prompt, max_new_tokens=150)
            end_time = time.time()
            end_mem = measure_memory()

            generated_text = output[0]["generated_text"]
            answer = generated_text[len(prompt):].strip()

            f.write(f"답변:\n{answer}\n⏱️ 소요시간: {end_time - start_time:.2f}초\n💾 메모리 사용: {end_mem - start_mem:.2f} MB\n")
            print(f"답변:\n{answer}\n⏱️ 소요시간: {end_time - start_time:.2f}초\n💾 메모리 사용: {end_mem - start_mem:.2f} MB\n")

            total_time += (end_time - start_time)
            successful_tests += 1
        except Exception as e:
            f.write(f"❌ 에러 발생: {str(e)}\n")
            print(f"❌ 에러 발생: {str(e)}")

        f.write(f"{'-'*30}\n")
        print(f"{'-'*30}\n")

    summary = f"\n📊 {model_name} 요약:\n성공한 테스트: {successful_tests}/{len(prompts)}\n"
    if successful_tests > 0:
        summary += f"평균 응답시간: {total_time/successful_tests:.2f}초\n"
    f.write(summary)
    print(summary)

# --------------------------
# 메인
# --------------------------
def main():
    print("🚀 언어모델 벤치마크 테스트 시작\n" + "="*50)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("언어모델 벤치마크 테스트 결과\n" + "="*50 + "\n")
        f.write(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU 사용: 아니오\n")
        f.write(f"테스트 모델 수: {len(models)}\n")
        f.write(f"프롬프트 수: {len(prompts)}\n\n")

        for model_idx, model_name in enumerate(models, 1):
            model, tokenizer = load_model_safely(model_name)
            if model and tokenizer:
                test_model(model_name, model, tokenizer, prompts, f)
                print(f"✅ {model_name} 테스트 완료\n")
            else:
                f.write(f"\n❌ {model_name}: 로딩 실패로 스킵\n")
                print(f"❌ {model_name} 스킵됨\n")
            del model, tokenizer
            clear_memory()

    try:
        files.download(output_file)
        print(f"📥 결과 파일 다운로드 완료: {output_file}")
    except:
        print(f"📄 파일 확인: {output_file}")

if __name__ == "__main__":
    main()