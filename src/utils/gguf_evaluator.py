import os
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not installed!")
    print("Install: pip install llama-cpp-python")
    exit(1)

from transformers import AutoTokenizer
from src.utils.data_loader import load_json, save_json


class GGUFEvaluator:
    """GGUF 모델 전용 평가 클래스 (UniversalModelEvaluator와 동일한 인터페이스)"""
    
    def __init__(self, model_path, tokenizer_path="models/original/gemma-3-1b-pt", use_gpu=True):
        print(f"Loading GGUF model: {model_path}")
        
        # llama.cpp 모델 로드
        n_gpu_layers = -1 if use_gpu else 0
        
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False
        )
        
        # Tokenizer 로드 (원본 모델에서)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_path = model_path
        self.device = 'cuda' if use_gpu else 'cpu'
        
        print(f"✓ GGUF model loaded")
        print(f"Evaluator device: {self.device}")
        
        # BLEU/ROUGE 메트릭 로드 (model_evaluator.py와 동일)
        self.bleu = None
        self.rouge = None
        
        try:
            import evaluate
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
            print("✓ BLEU/ROUGE loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: BLEU/ROUGE loading failed - {str(e)}")
            print("  Evaluation will continue without BLEU/ROUGE metrics")
    
    def get_model_info(self):
        """모델 기본 정보 (model_evaluator.py와 동일한 형식)"""
        import os
        
        file_size_mb = os.path.getsize(self.model_path) / (1024**2)
        
        # Q4_K_M 기준 파라미터 수 추정
        # 4bit quantization ≈ 0.5 bytes per parameter
        estimated_params = int((file_size_mb * 1024 * 1024) / 0.5)
        
        return {
            "total_parameters": estimated_params,
            "model_memory_mb": file_size_mb,
            "measurement_method": "GGUF file size (4-bit quantized)",
            "model_dtype": "Q4_K_M (4-bit quantization)",
            "device": str(self.device),
            "model_format": "GGUF"
        }
    
    def calculate_bleu_rouge_scores(self, generated_texts, expected_texts):
        """BLEU/ROUGE 점수 계산 (model_evaluator.py와 동일)"""
        if self.bleu is None or self.rouge is None:
            return None
        
        try:
            # BLEU 계산
            bleu_result = self.bleu.compute(
                predictions=generated_texts,
                references=[[ref] for ref in expected_texts]
            )
            
            # ROUGE 계산
            rouge_result = self.rouge.compute(
                predictions=generated_texts,
                references=expected_texts
            )
            
            return {
                "bleu": bleu_result['bleu'],
                "rouge1": rouge_result['rouge1'],
                "rouge2": rouge_result['rouge2'],
                "rougeL": rouge_result['rougeL']
            }
        except Exception as e:
            print(f"⚠️ BLEU/ROUGE calculation failed: {e}")
            return None
    
    def evaluate_generation(self, test_prompts, max_new_tokens=20):
        """생성 성능 평가 (model_evaluator.py와 동일한 형식)"""
        
        print(f"\n생성 성능 평가 ({len(test_prompts)}개 프롬프트)")
        
        results = []
        total_time = 0
        successful_generations = 0
        
        for prompt in tqdm(test_prompts, desc="생성 테스트"):
            try:
                # 생성
                start_time = time.time()
                
                output = self.model(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=0.0,  # do_sample=False와 동일
                    top_p=1.0,
                    echo=False,
                    stop=[self.tokenizer.eos_token],
                    repeat_penalty=1.1  # repetition_penalty와 동일
                )
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                generated_text = output['choices'][0]['text'].strip()
                
                # 결과 저장
                is_successful = len(generated_text.strip()) > 0
                if is_successful:
                    successful_generations += 1
                
                results.append({
                    "prompt": prompt,
                    "generated": generated_text,
                    "generation_time": generation_time,
                    "success": is_successful,
                    "response_length": len(generated_text)
                })
                
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "generated": "",
                    "generation_time": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # 통계 계산
        avg_time = total_time / len(test_prompts) if len(test_prompts) > 0 else 0
        success_rate = successful_generations / len(test_prompts) if len(test_prompts) > 0 else 0
        avg_response_length = np.mean([r['response_length'] for r in results if r['success']]) if successful_generations > 0 else 0
        
        return {
            "results": results,
            "statistics": {
                "total_prompts": len(test_prompts),
                "successful_generations": successful_generations,
                "success_rate": success_rate,
                "average_generation_time": avg_time,
                "average_response_length": avg_response_length,
                "total_evaluation_time": total_time
            }
        }
    
    def calculate_exact_match_accuracy(self, generated_texts, expected_texts):
        """Exact Match 정확도 계산 (model_evaluator.py와 동일)"""
        exact_matches = 0
        for gen, exp in zip(generated_texts, expected_texts):
            if gen.strip().lower() == exp.strip().lower():
                exact_matches += 1
        
        return exact_matches / len(generated_texts) if len(generated_texts) > 0 else 0.0
    
    def evaluate_on_dataset(self, dataset):
        """데이터셋 기반 평가 (model_evaluator.py와 완전히 동일한 형식)"""
        
        print(f"\n데이터셋 평가 (전체 {len(dataset)}개 샘플)")
        
        # 전체 샘플 사용
        prompts = []
        expected_answers = []
        
        for item in dataset:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if instruction and output:
                prompts.append(instruction)
                expected_answers.append(output)
        
        print(f"유효한 샘플: {len(prompts)}개")
        
        # 생성 평가
        generation_results = self.evaluate_generation(prompts, max_new_tokens=50)
        
        # 생성된 텍스트 추출
        generated_texts = [r['generated'] for r in generation_results['results']]
        
        # 1. 키워드 매칭 점수
        keyword_scores = []
        for i in range(len(prompts)):
            generated = generated_texts[i].lower()
            expected = expected_answers[i].lower()
            
            expected_words = set(expected.split())
            generated_words = set(generated.split())
            
            if len(expected_words) > 0:
                overlap = len(expected_words & generated_words)
                keyword_score = overlap / len(expected_words)
                keyword_scores.append(keyword_score)
            else:
                keyword_scores.append(0.0)
        
        # 2. Exact Match 정확도
        exact_match_acc = self.calculate_exact_match_accuracy(generated_texts, expected_answers)
        avg_keyword_score = np.mean(keyword_scores) if keyword_scores else 0.0
        
        # 3. BLEU/ROUGE 점수 계산
        bleu_rouge_scores = self.calculate_bleu_rouge_scores(generated_texts, expected_answers)
        
        # 통계 출력 (model_evaluator.py와 동일)
        print("\n" + "="*70)
        print("데이터셋 평가 통계")
        print("="*70)
        print(f"  생성 성공률: {generation_results['statistics']['success_rate']:.1%}")
        print(f"  Exact Match 정확도: {exact_match_acc:.1%}")
        print(f"  키워드 매칭 점수: {avg_keyword_score:.1%}")
        
        if bleu_rouge_scores:
            print(f"  BLEU 점수: {bleu_rouge_scores['bleu']:.3f}")
            print(f"  ROUGE-1: {bleu_rouge_scores['rouge1']:.3f}")
            print(f"  ROUGE-2: {bleu_rouge_scores['rouge2']:.3f}")
            print(f"  ROUGE-L: {bleu_rouge_scores['rougeL']:.3f}")
        
        print(f"  평균 응답 시간: {generation_results['statistics']['average_generation_time']:.3f}초")
        print("="*70)
        
        return {
            "generation_results": generation_results,
            "quality_metrics": {
                "exact_match_accuracy": exact_match_acc,
                "keyword_match_score": avg_keyword_score,
                "bleu_rouge": bleu_rouge_scores,
                "keyword_scores": keyword_scores
            }
        }


def evaluate_gguf_model(
    model_path,
    evaluate_id,
    test_data_path=None,
    save_path=None,
    use_gpu=True
):
    """
    GGUF 모델 평가 함수 (evaluate_model_universal과 동일한 인터페이스)
    
    Args:
        model_path: GGUF 모델 파일 경로
        evaluate_id: 평가 식별자
        test_data_path: 테스트 데이터 경로
        save_path: 결과 저장 경로
        use_gpu: GPU 사용 여부
    
    Returns:
        평가 결과 딕셔너리
    """
    
    print("=" * 70)
    print("GGUF 모델 평가 시작")
    print("=" * 70)
    
    device = 'cuda' if use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    # 1. 모델 로드
    try:
        evaluator = GGUFEvaluator(
            model_path=model_path,
            use_gpu=use_gpu
        )
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. 모델 기본 정보
    model_info = evaluator.get_model_info()
    print(f"\n모델 정보:")
    print(f"  파라미터: {model_info['total_parameters']:,}")
    print(f"  메모리: {model_info['model_memory_mb']:.2f} MB")
    print(f"  dtype: {model_info['model_dtype']}")
    
    # 3. 데이터셋 평가
    dataset_results = None
    if test_data_path and os.path.exists(test_data_path):
        try:
            dataset = load_json(test_data_path)
            dataset_results = evaluator.evaluate_on_dataset(dataset)
            
            # 결과 저장 (model_evaluator.py와 완전히 동일한 형식)
            final_results = {
                "evaluate_id": evaluate_id,
                "model_path": model_path,
                "model_info": model_info,
                "dataset_evaluation": dataset_results,
                "summary": {
                    "dataset_size": len(dataset),
                    "success_rate": dataset_results['generation_results']['statistics']['success_rate'],
                    "exact_match_accuracy": dataset_results['quality_metrics']['exact_match_accuracy'],
                    "keyword_match_score": dataset_results['quality_metrics']['keyword_match_score'],
                    "bleu_rouge": dataset_results['quality_metrics']['bleu_rouge'],
                    "avg_response_time": dataset_results['generation_results']['statistics']['average_generation_time'],
                    "model_parameters": model_info['total_parameters'],
                    "model_memory_mb": model_info['model_memory_mb']
                }
            }
            
            if save_path:
                save_json(final_results, save_path)
                print(f"\n✓ 결과 저장: {save_path}")
            
            print(f"\n✓ 평가 완료")
            return final_results
            
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print("\n✓ 평가 완료")
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GGUF 모델 평가 스크립트')
    parser.add_argument('model_path', type=str, help='GGUF 모델 파일 경로')
    parser.add_argument('result_filename', type=str, help='결과 파일명 (evaluate_id로 사용)')
    parser.add_argument('--test_data', type=str, default='data/split/final_validation.json',
                        help='테스트 데이터 경로')
    parser.add_argument('--no_gpu', action='store_true', help='GPU 사용 안 함')
    
    args = parser.parse_args()
    
    # 결과 저장 경로 생성
    save_path = f"results/{args.result_filename}.json"
    
    # 모델 평가 실행
    results = evaluate_gguf_model(
        model_path=args.model_path,
        evaluate_id=args.result_filename,
        test_data_path=args.test_data,
        save_path=save_path,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    main()
