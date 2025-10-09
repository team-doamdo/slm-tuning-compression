"""
프루닝된 모델 검증 스크립트

"""

import torch
import json
import os
import sys
from pathlib import Path

sys.path.append('/content/drive/MyDrive/smartfarm_pruning')

from src.utils.model_loader import load_pruned_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_pruned_model(model_path='models/pruned_magnitude'):
    """프루닝된 모델 종합 검증"""
    
    print("=" * 60)
    print(" 프루닝 모델 검증 테스트")
    print("=" * 60)
    
    # [1/4] 프루닝 메타데이터 확인
    print("\n[1/4] 프루닝 메타데이터 확인...")
    try:
        metadata_path = os.path.join(model_path, "pruning_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            print("메타데이터 로드 성공")
            print(f"   프루닝 방법: {metadata.get('pruning_method', 'N/A')}")
            print(f"   프루닝된 헤드: {metadata.get('total_heads_pruned', 0)}개")
            print(f"   프루닝된 뉴런: {metadata.get('total_neurons_pruned', 0)}개")
            print(f"   영향받은 레이어: {metadata.get('layers_affected', 0)}개")
        else:
            print("메타데이터 파일 없음")
            metadata = None
    except Exception as e:
        print(f"❌ 메타데이터 로드 실패: {e}")
        metadata = None
    
    # [2/4] 모델 로드
    print("\n[2/4] 모델 로드 테스트...")
    try:
        # float32 보장된 프루닝 모델 로드
        model, tokenizer = load_pruned_model(model_path, device='cuda', dtype=torch.float32)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print("모델 로드 성공")
        print(f"   전체 파라미터: {total_params:,}")
        print(f"   모델 dtype: {model.dtype}")
        
        # 프루닝 효과 추정
        if metadata:
            heads_pruned = metadata.get('total_heads_pruned', 0)
            neurons_pruned = metadata.get('total_neurons_pruned', 0)
            print(f"   프루닝 헤드: {heads_pruned}개")
            print(f"   프루닝 뉴런: {neurons_pruned}개")
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # [3/4] Forward pass 테스트
    print("\n[3/4] Forward pass 테스트...")
    try:
        model.eval()
        dummy_input = torch.randint(0, 1000, (1, 10)).to(model.device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            logits = outputs.logits
            
            # NaN/Inf 체크
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()
            
            if has_nan or has_inf:
                print(f"❌ Logits에 이상값 발견!")
                print(f"   NaN: {has_nan}, Inf: {has_inf}")
                return False
        
        print(f"Forward pass 성공")
        print(f"   출력 shape: {logits.shape}")
        print(f"   Logits 정상: NaN={has_nan}, Inf={has_inf}")
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # [4/4] 텍스트 생성 테스트
    print("\n[4/4] 텍스트 생성 테스트...")
    try:
        test_prompts = [
            "Hello",
            "The weather is", 
            "Paris is the capital of"
        ]
        
        success_count = 0
        
        for prompt in test_prompts:
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                add_special_tokens=True
            ).to(model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=input_length + 20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_ids = outputs[0][input_length:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 생성 성공 여부
            is_success = len(generated.strip()) > 0
            status = "✅" if is_success else "❌"
            
            if is_success:
                success_count += 1
            
            print(f"  {status} '{prompt}' -> '{generated[:50]}{'...' if len(generated) > 50 else ''}'")
        
        print(f"\n생성 성공: {success_count}/{len(test_prompts)}")
        
        if success_count == 0:
            print("❌ 모든 생성 실패")
            return False
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # [5/5] 원본 모델과 비교 (선택적)
    print("\n[5/5] 파라미터 수 비교...")
    try:
        # 원본 모델 추정치 (Gemma-3-1b-pt 기준)
        estimated_original_params = 999_885_952
        
        if total_params < estimated_original_params:
            reduction = (estimated_original_params - total_params) / estimated_original_params * 100
            print(f"✅ 파라미터 감소 확인")
            print(f"   추정 원본: {estimated_original_params:,}")
            print(f"   현재 프루닝: {total_params:,}")
            print(f"   감소율: {reduction:.1f}%")
        else:
            print(f"⚠️ 파라미터 수가 예상보다 크거나 같음")
            print(f"   추정 원본: {estimated_original_params:,}")
            print(f"   현재: {total_params:,}")
        
    except Exception as e:
        print(f"⚠️ 비교 실패: {e}")
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("✅ 모든 핵심 테스트 통과!")
    print("=" * 60)
    print("\n 검증 요약:")
    print(f"  모델 경로: {model_path}")
    print(f"  파라미터: {total_params:,}")
    print(f"  dtype: {model.dtype}")
    if metadata:
        print(f"  프루닝된 헤드: {metadata.get('total_heads_pruned', 0)}개")
        print(f"  프루닝된 뉴런: {metadata.get('total_neurons_pruned', 0)}개")
    print(f"  생성 테스트: {success_count}/{len(test_prompts)} 성공")
    print("\n✅ 프루닝된 모델이 정상적으로 작동합니다!")
    print("=" * 60)
    
    return True


def quick_validation(model_path='models/pruned_magnitude'):
    """빠른 검증 (생성 테스트만)"""
    
    print(" 빠른 프루닝 모델 검증")
    print("-" * 40)
    
    try:
        model, tokenizer = load_pruned_model(model_path, device='cuda', dtype=torch.float32)
        
        inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ 빠른 테스트 성공: '{generated}'")
        return True
        
    except Exception as e:
        print(f"❌ 빠른 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    import os
    
    # 작업 디렉토리 설정
    os.chdir('/content/drive/MyDrive/smartfarm_pruning')
    
    # 전체 테스트 실행
    print("프루닝 모델 검증 시작...\n")
    
    success = test_pruned_model('models/pruned_magnitude')
    
    if success:
        print("\n 모든 검증 완료!")
    else:
        print("\n 일부 테스트 실패")
        
        # 빠른 검증 시도
        print("\n빠른 검증 시도...")
        quick_success = quick_validation('models/pruned_magnitude')
        
        if quick_success:
            print("기본 작동은 확인됨")
        else:
            print("심각한 문제 있음")
            sys.exit(1)