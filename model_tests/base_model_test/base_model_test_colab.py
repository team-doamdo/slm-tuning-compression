from huggingface_hub import login
import psutil, os


# 🤫 Hugging Face 웹에서 발급한 토큰 넣기
hf_token = ""

# 로그인
login(token=hf_token)
print("✅ Hugging Face 로그인 완료")
# Google Colab용 언어 모델 벤치마크 테스트

import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from google.colab import files
import os

# --------------------------
# 설정
# --------------------------

# 코랩 환경에 맞게 가벼운 모델들로 선별
models = [
    # "google/gemma-2-2b-it",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",  # 더 안정적인 TinyLlama 모델
    # "google/gemma-3-1b-it",  # 가벼운 대화형 모델
    # "microsoft/Phi-4-mini-instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct", #베이스 모델
    "deepseek-ai/deepseek-coder-1.3b-instruct"
]

# 실제 테스트 프롬프트 예시
prompts = [
    "What is 28 + 93?",  # basic_arithmetic
    "Which country has the largest population?",  # factual_knowledge
    "If you put a plant in a cave with no sunlight at all, why would it not survive for long?",  # scientific_reasoning
    "Context: On January 10, 2025, the New York City Metropolitan Transportation Authority (MTA) announced that it will add 265 new zero-emission electric buses to city bus routes. Together with the 60 buses introduced last year and the 205 that will begin operating by the end of this year, a total of 530 electric buses will be running throughout New York City. These new 40-foot buses use a regenerative braking system that recovers up to 90% of energy during braking and can reduce annual carbon dioxide emissions by up to 90 tons per bus. MTA Chair Janno Lieber stated, \"With the introduction of these electric buses and infrastructure upgrades at bus depots across the city, New Yorkers will be able to breathe cleaner air.\" In addition, a new charging infrastructure will be installed at the Jamaica Bus Depot to lay the foundation for future zero-emission bus operations. Task: Summarize the passage above into one sentence in English.",  # text_comprehension
    "2, 4, 8, 16, ? What is the next number in the sequence?",  # pattern_recognition_1
    "If January 1 is Monday and January 8 is also Monday, what day of the week is January 15?",  # pattern_recognition_2
    "The following temperature sensor readings are given: [22, 23, 22, 45, 23, 22]. Which value seems to be an outlier, and why?",  # outlier_detection_1
    "Here are the test scores: [85, 88, 90, 300, 87, 89]. Which score is an outlier, and why?",  # outlier_detection_2
    "Please write a short poem.",  # text_generation
    "Rewrite this sentence so that a child can understand it: \"The Earth revolves around the Sun, causing the change of seasons.\""  # text_editing
]

output_file = "model_eval_results.txt"

# --------------------------
# 함수들
# --------------------------

def setup_gpu():
    """CPU 전용 설정"""
    print("ℹ️ CPU 모드로 실행됩니다.")
    return False


# def measure_memory():
#     """CPU 메모리 사용량 측정 (MB 단위)"""
#     process = psutil.Process(os.getpid())
#     cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
#     return cpu_mem


def measure_memory():
    """전체 시스템 메모리 사용량 측정 (MB 단위)"""
    mem = psutil.virtual_memory()
    # mem.used는 현재 사용 중인 메모리 (MB)
    return mem.used / (1024 ** 2)

def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_safely(model_name):
    """안전하게 모델 로드"""
    try:
        print(f"🔄 모델 로딩 중: {model_name}")

        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # pad_token이 없는 경우 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 모델 로드 (메모리 절약을 위해 float16 사용)
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
    """개별 모델 테스트"""
    header = f"\n{'='*50}\n모델: {model_name}\n{'='*50}\n\n"
    f.write(header)
    print(header)  # 콘솔에 바로 출력

    # 파이프라인 생성
    try:
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id
        )
    except Exception as e:
        error_msg = f"❌ 파이프라인 생성 실패: {str(e)}\n"
        f.write(error_msg)
        print(error_msg)
        return

    total_time = 0
    successful_tests = 0

    for i, prompt in enumerate(prompts, 1):
        section_header = f"\n--- 프롬프트 {i} ---\n질문: {prompt}\n"
        f.write(section_header)
        print(section_header)

        try:
            start_mem = measure_memory()
            start_time = time.time()

            # 텍스트 생성
            output = gen_pipeline(prompt, max_new_tokens=150)

            end_time = time.time()
            end_mem = measure_memory()

            generated_text = output[0]["generated_text"]
            answer = generated_text[len(prompt):].strip()

            answer_text = f"답변:\n{answer}\n⏱️ 소요시간: {end_time - start_time:.2f}초\n💾 메모리 사용: {end_mem - start_mem:.2f} MB\n"
            f.write(answer_text)
            print(answer_text)

            total_time += (end_time - start_time)
            successful_tests += 1

        except Exception as e:
            error_msg = f"❌ 에러 발생: {str(e)}\n"
            f.write(error_msg)
            print(error_msg)

        f.write(f"{'-'*30}\n")
        print(f"{'-'*30}\n")

    # 모델 요약
    summary_text = f"\n📊 {model_name} 요약:\n성공한 테스트: {successful_tests}/{len(prompts)}\n"
    if successful_tests > 0:
        summary_text += f"평균 응답시간: {total_time/successful_tests:.2f}초\n"
    f.write(summary_text)
    print(summary_text)

    for i, prompt in enumerate(prompts, 1):
        f.write(f"\n--- 프롬프트 {i} ---\n")
        f.write(f"질문: {prompt}\n\n")

        try:
            start_mem = measure_memory()
            start_time = time.time()

            # 텍스트 생성
            output = gen_pipeline(prompt, max_new_tokens=150)

            end_time = time.time()
            end_mem = measure_memory()

            # 결과 처리
            generated_text = output[0]["generated_text"]
            # 원본 프롬프트 제거하고 답변만 추출
            answer = generated_text[len(prompt):].strip()

            f.write(f"답변:\n{answer}\n\n")
            f.write(f"⏱️ 소요시간: {end_time - start_time:.2f}초\n")
            f.write(f"💾 메모리 사용: {end_mem - start_mem:.2f} MB\n")

            total_time += (end_time - start_time)
            successful_tests += 1
        except Exception as e:
            f.write(f"❌ 에러 발생: {str(e)}\n")

        f.write(f"\n{'-'*30}\n")

    # 모델 요약
    f.write(f"\n📊 {model_name} 요약:\n")
    f.write(f"성공한 테스트: {successful_tests}/{len(prompts)}\n")
    if successful_tests > 0:
        f.write(f"평균 응답시간: {total_time/successful_tests:.2f}초\n")
    f.write(f"\n")

# --------------------------
# 메인 실행부
# --------------------------

def main():
    print("🚀 언어모델 벤치마크 테스트 시작")
    print("="*50)

    # GPU 설정 확인
    gpu_available = setup_gpu()

    # 결과 파일 생성
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("언어모델 벤치마크 테스트 결과\n")
        f.write("="*50 + "\n")
        f.write(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU 사용: {'예' if gpu_available else '아니오'}\n")
        f.write(f"테스트 모델 수: {len(models)}\n")
        f.write(f"프롬프트 수: {len(prompts)}\n\n")

        for model_idx, model_name in enumerate(models, 1):
            print(f"\n[{model_idx}/{len(models)}] 테스트 중: {model_name}")

            # 모델 로드
            model, tokenizer = load_model_safely(model_name)

            if model is not None and tokenizer is not None:
                # 모델 테스트
                test_model(model_name, model, tokenizer, prompts, f)
                print(f"✅ {model_name} 테스트 완료")
            else:
                f.write(f"\n❌ {model_name}: 로딩 실패로 스킵\n\n")
                print(f"❌ {model_name} 스킵됨")

            # 메모리 정리
            del model, tokenizer
            clear_memory()

            print(f"진행률: {model_idx}/{len(models)} ({model_idx/len(models)*100:.1f}%)")

    print(f"\n🎉 모든 테스트 완료!")
    print(f"결과 파일: {output_file}")

    # 결과 파일 다운로드 (코랩에서)
    try:
        files.download(output_file)
        print(f"📥 결과 파일이 다운로드되었습니다.")
    except:
        print(f"📄 결과를 확인하려면 파일 탭에서 '{output_file}'을 확인하세요.")

# 실행
if __name__ == "__main__":
    main()