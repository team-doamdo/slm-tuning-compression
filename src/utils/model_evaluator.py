import torch
import json
import numpy as np
import time
import psutil
import os
import sys
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: peft not installed. LoRA models won't be supported.")
    print("Install with: pip install peft")
    PeftModel = None
    PEFT_AVAILABLE = False

# 프로젝트 경로 추가
project_root = "/content/drive/MyDrive/smartfarm_pruning"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.model_loader import load_model, load_pruned_model
from src.utils.data_loader import load_json, save_json


def auto_load_model(model_path, device=None):
    """자동으로 원본/프루닝/LoRA 모델 감지해서 로드"""

    # device 자동 감지
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"모델 자동 로드: {model_path}")
    print(f"사용 device: {device}")
    
    # LoRA adapter인지 확인 (adapter_config.json 존재 여부)
    adapter_config_path = os.path.join(model_path, 'adapter_config.json')
    
    if os.path.exists(adapter_config_path):
        print("LoRA adapter로 감지됨")
        
        # adapter_config에서 base_model_name_or_path 읽기
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_path = adapter_config.get('base_model_name_or_path', 'models/pruned_activation/')
        
        print(f"  Base model: {base_model_path}")
        print(f"  LoRA adapter: {model_path}")
        
        # Base 모델 로드 (프루닝된 모델)
        from peft import PeftModel
        
        base_model, tokenizer = load_pruned_model(base_model_path, device=device, dtype=torch.float32)
        
        # LoRA adapter 로드
        print("\n LoRA adapter 로드 중...")
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float32
        )
        
        # Merge
        print(" LoRA merge 중...")
        model = model.merge_and_unload()
        print(" LoRA merge 완료!")
        
        return model, tokenizer
    
    # 프루닝 모델인지 확인
    pruned_structure_path = os.path.join(model_path, 'pruned_structure.json')
    
    if os.path.exists(pruned_structure_path):
        print("프루닝 모델로 감지됨")
        return load_pruned_model(model_path, device=device, dtype=torch.float32)
    else:
        print("원본 모델로 감지됨")
        return load_model(model_path, device=device, dtype=torch.float32)


class UniversalModelEvaluator:
    """범용 모델 평가 클래스"""
    
    def __init__(self, model, tokenizer, device=None):
      # device 자동 감지
      if device is None:
          device = 'cuda' if torch.cuda.is_available() else 'cpu'

      self.model = model
      self.tokenizer = tokenizer
      self.device = device
      self.model.eval()

      print(f"Evaluator device: {self.device}")
      
      # BLEU/ROUGE 메트릭 로드
      self.bleu = None
      self.rouge = None
      
      # Python 인터프리터 재시작 없이 import 시도
      import sys
      
      # 현재 디렉토리를 sys.path에서 제거 (충돌 방지)
      current_dir = os.path.dirname(os.path.abspath(__file__))
      if current_dir in sys.path:
          sys.path.remove(current_dir)
      
      try:
          # 직접 import 시도
          import evaluate
          self.bleu = evaluate.load("bleu")
          self.rouge = evaluate.load("rouge")
          print("BLEU/ROUGE loaded successfully")
      except Exception as e:
          print(f"Warning: BLEU/ROUGE loading failed - {str(e)}")
          import traceback
          traceback.print_exc()
      
      # 경로 복원
      if current_dir not in sys.path:
          sys.path.insert(0, current_dir)

    def calculate_bleu_rouge_scores(self, generated_texts, expected_texts):
        """BLEU/ROUGE 점수 계산"""
        if self.bleu is None or self.rouge is None:
            return None
        
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

    def get_model_info(self):
      """모델 기본 정보 - CPU/GPU 모두 정확한 메모리 측정"""
      total_params = sum(p.numel() for p in self.model.parameters())
      
      # 메모리 사용량 측정
      if torch.cuda.is_available() and self.device == 'cuda':
          # GPU: CUDA 메모리 직접 측정
          torch.cuda.empty_cache()
          model_memory_mb = torch.cuda.memory_allocated() / (1024**2)
          measurement_method = "CUDA allocated"
          
      else:
          # CPU: 정확한 메모리 측정 
          try:
              import psutil
              import os
              import gc
              
              # 가비지 컬렉션으로 정확도 향상
              gc.collect()
              
              # PyTorch 텐서 메모리 계산
              model_tensor_memory = 0
              
              # Parameters 메모리
              for param in self.model.parameters():
                  model_tensor_memory += param.numel() * param.element_size()
              
              # Buffers 메모리 (batch norm의 running_mean 등)
              for buffer in self.model.buffers():
                  model_tensor_memory += buffer.numel() * buffer.element_size()
              
              model_memory_mb = model_tensor_memory / (1024**2)
              measurement_method = "PyTorch tensors (params + buffers)"
              
              # 프로세스 전체 메모리 (참고용)
              process = psutil.Process(os.getpid())
              process_memory_mb = process.memory_info().rss / (1024**2)
              
              # 디버깅 정보 출력
              print(f"  모델 메모리 (텐서): {model_memory_mb:.2f} MB")
              print(f"  프로세스 전체 메모리: {process_memory_mb:.2f} MB")
              
          except ImportError:
              # psutil이 설치 안 된 경우 fallback
              print("  ⚠️ Warning: psutil not installed. Using parameter-only estimation.")
              print("  ⚠️ Install: pip install psutil")
              
              # Fallback: 파라미터만 계산
              param_memory = sum(p.element_size() * p.numel() 
                              for p in self.model.parameters())
              model_memory_mb = param_memory / (1024**2)
              measurement_method = "Parameters only (estimated)"
      
      return {
          "total_parameters": total_params,
          "model_memory_mb": model_memory_mb,
          "measurement_method": measurement_method,  # 측정 방식 추가
          "model_dtype": str(self.model.dtype),
          "device": str(self.device)
      }

    def evaluate_generation(self, test_prompts, max_new_tokens=20):
        """생성 성능 평가"""
        
        print(f"생성 성능 평가 ({len(test_prompts)}개 프롬프트)")
        
        results = []
        total_time = 0
        successful_generations = 0
        
        for prompt in tqdm(test_prompts, desc="생성 테스트"):
            try:
                # 토크나이징
                start_time = time.time()
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # 생성
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                # 디코드
                input_length = inputs["input_ids"].shape[1]
                generated_ids = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 결과 저장
                is_successful = len(generated_text.strip()) > 0
                if is_successful:
                    successful_generations += 1
                
                results.append({
                    "prompt": prompt,
                    "generated": generated_text.strip(),
                    "generation_time": generation_time,
                    "success": is_successful,
                    "response_length": len(generated_text.strip())
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
        avg_time = total_time / len(test_prompts)
        success_rate = successful_generations / len(test_prompts)
        avg_response_length = np.mean([r['response_length'] for r in results if r['success']])
        
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
        """Exact Match 정확도 계산"""
        exact_matches = 0
        for gen, exp in zip(generated_texts, expected_texts):
            if gen.strip().lower() == exp.strip().lower():
                exact_matches += 1
        
        return exact_matches / len(generated_texts) if len(generated_texts) > 0 else 0.0

    def evaluate_on_dataset(self, dataset):
      """데이터셋 기반 평가 (전체 데이터 사용)"""
      
      print(f"데이터셋 평가 (전체 {len(dataset)}개 샘플)")
      
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
      
      # 통계 출력
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


def evaluate_model_universal(
    model_path,
    evaluate_id,
    test_data_path=None,
    save_path=None,
    custom_prompts=None,
    device=None
):
    """
    범용 모델 평가 함수
    
    Args:
        model_path: 모델 경로 (원본/프루닝 자동 감지)
        evaluate_id: 평가 식별자 (결과 파일에 저장)
        test_data_path: 테스트 데이터 경로 (선택)
        save_path: 결과 저장 경로 (선택)
        custom_prompts: 커스텀 프롬프트 리스트 (선택)
    
    Returns:
        평가 결과 딕셔너리
    """
    
    print("=" * 70)
    print("범용 모델 평가 시작")
    print("=" * 70)
    
    # Device 자동 감지
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. 모델 로드 (자동 감지)
    try:
        model, tokenizer = auto_load_model(model_path, device=device)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None
    
    # 2. 평가기 초기화
    evaluator = UniversalModelEvaluator(model, tokenizer)
    
    # 3. 모델 기본 정보
    model_info = evaluator.get_model_info()
    print(f"\n모델 정보:")
    print(f"  파라미터: {model_info['total_parameters']:,}")
    print(f"  메모리: {model_info['model_memory_mb']:.2f} MB")
    print(f"  dtype: {model_info['model_dtype']}")
    
    # 데이터셋 평가 
    dataset_results = None
    if test_data_path and os.path.exists(test_data_path):
        try:
            dataset = load_json(test_data_path)
            dataset_results = evaluator.evaluate_on_dataset(dataset)
            
            # 결과 저장
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
                print(f"\n결과 저장: {save_path}")
            
            print(f"\n평가 완료")
            return final_results
            
        except Exception as e:
            print(f"평가 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print("\n평가 완료")
    return None


# 실행 예시
if __name__ == "__main__":
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='모델 평가 스크립트')
    parser.add_argument('model_path', type=str, help='모델 경로')
    parser.add_argument('result_filename', type=str, help='결과 파일명 (evaluate_id로 사용)')
    parser.add_argument('--test_data', type=str, default='data/split/final_validation.json',
                        help='테스트 데이터 경로')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', None],
                        help='디바이스 (cuda/cpu/None=자동감지)')
    
    args = parser.parse_args()
    
    # 결과 저장 경로 생성
    save_path = f"results/{args.result_filename}.json"
    
    # 모델 평가 실행
    results = evaluate_model_universal(
        model_path=args.model_path,
        evaluate_id=args.result_filename,
        test_data_path=args.test_data,
        save_path=save_path,
        device=args.device
    )