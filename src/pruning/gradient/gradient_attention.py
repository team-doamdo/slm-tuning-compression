import torch
import numpy as np
from tqdm import tqdm
import sys
import os
import gc

# 작업 디렉토리를 프로젝트 루트로 변경
os.chdir('/content/drive/MyDrive/smartfarm_pruning')

# 현재 디렉토리를 Python 경로에 추가
sys.path.append('.')

# 프로젝트 루트를 Python 경로에 추가
project_root = "/content/drive/MyDrive/smartfarm_pruning"
if project_root not in sys.path:
    sys.path.append(project_root)

# utils 함수들 import
from src.utils.model_loader import load_model
from src.utils.data_loader import save_json, load_json

# ========== 필수 통일 사항 ==========
# 1. 프루닝 설정
PRUNE_RATIO = 0.0  

# 2. 데이터 처리 (동일한 조건으로 측정)
BATCH_SIZE = 4
MAX_LENGTH = 512
MAX_SAMPLES = None  # 테스트 시 같은 수 사용 (None = 전체)

# 3. 경로 (동일한 모델과 데이터 사용)
MODEL_PATH = "models/original/gemma-3-1b-it"
DATA_PATH = "data/split/pruning_activation.json"
OUTPUT_FILE = "results/gradient_heads_to_prune_it.json"

# 4. 기술 설정 (동일한 정밀도)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32  # float16 사용 금지
ATTENTION_IMPLEMENTATION = "eager"  # 안정성

# 5. 메모리 관리 (동일한 주기)
CLEAR_CACHE_EVERY = 10
GC_COLLECT_EVERY = 10

# 6. 디버깅 (테스트 시 동일하게)
DEBUG_MODE = False  # True 시 MAX_SAMPLES=10 자동 설정
# ====================================


def measure_head_gradient(model, tokenizer, data_samples):
    """
    Gradient 기반으로 attention heads 중요도 측정
    
    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        data_samples: 중요도 측정용 데이터 샘플들
        
    Returns:
        head_outputs: Dict[layer_idx][head_idx] = [gradient_values]
    """
    
    print("Attention heads gradient 측정 중...")
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_q_heads
    
    print(f"  분석 대상: {num_layers}개 레이어 × {num_q_heads}개 헤드")
    print(f"  데이터 샘플: {len(data_samples)}개")
    print(f"  배치 크기: {BATCH_SIZE}, 최대 길이: {MAX_LENGTH}")
    
    # 각 헤드의 gradient 값 수집
    head_outputs = {layer_idx: {head_idx: [] for head_idx in range(num_q_heads)} 
                    for layer_idx in range(num_layers)}
    
    # Hook: attention output에서 각 헤드별 gradient 측정
    def create_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            try:
                # grad_output[0]는 gradient [batch, seq_len, hidden_size]
                if grad_output and grad_output[0] is not None:
                    gradient = grad_output[0]
                    
                    batch_size, seq_len, hidden_size = gradient.shape
                    
                    # [batch, seq_len, num_heads, head_dim]로 reshape
                    grad_per_head = gradient.view(batch_size, seq_len, num_q_heads, head_dim)
                    
                    # 각 헤드의 평균 gradient 크기 계산
                    for head_idx in range(num_q_heads):
                        head_grad = grad_per_head[:, :, head_idx, :].abs().mean().item()
                        head_outputs[layer_idx][head_idx].append(head_grad)
                        
            except Exception as e:
                pass
        return hook
    
    # Hook 등록
    hooks = []
    for layer_idx in range(num_layers):
        try:
            layer = model.model.layers[layer_idx].self_attn.o_proj
            hook = layer.register_full_backward_hook(create_hook(layer_idx))
            hooks.append(hook)
        except Exception as e:
            print(f"⚠ 레이어 {layer_idx} hook 등록 실패: {e}")
    
    # Forward + Backward pass
    model.train()  # gradient 계산을 위해 train mode
    
    for idx, sample in enumerate(tqdm(data_samples, desc="데이터 처리")):
        try:
            # 입력 텍스트 구성
            if isinstance(sample, dict):
                if 'instruction' in sample and 'output' in sample:
                    text = f"{sample['instruction']} {sample['output']}"
                elif 'input' in sample and 'output' in sample:
                    text = f"{sample['input']} {sample['output']}"
                elif 'question' in sample and 'answer' in sample:
                    text = f"{sample['question']} {sample['answer']}"
                else:
                    text = str(sample)
            else:
                text = str(sample)
            
            # 토크나이징
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True,
                add_special_tokens=True
            ).to(model.device)
            
            # labels 생성 (language modeling loss)
            labels = inputs["input_ids"].clone()
            
            # Forward pass
            model.zero_grad()
            outputs = model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            
            # Backward pass (gradient 계산)
            loss.backward()
            
            # 메모리 관리
            if (idx + 1) % CLEAR_CACHE_EVERY == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if (idx + 1) % GC_COLLECT_EVERY == 0:
                gc.collect()
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠ 샘플 {idx} 처리 실패: {e}")
            continue
    
    # Hook 제거
    for hook in hooks:
        hook.remove()
    
    # 수집 확인
    collected_heads = sum(len([v for v in heads.values() if len(v) > 0]) 
                         for heads in head_outputs.values())
    print(f"Gradient 측정 완료")
    print(f"   수집된 헤드: {collected_heads}개 (예상: {num_layers * num_q_heads}개)")
    
    return head_outputs


def calculate_head_importance(gradients):
    """
    측정된 gradient로 중요도 계산
    
    Args:
        gradients: measure_head_gradient()의 출력
        
    Returns:
        head_scores: Dict[layer_idx][head_idx] = score
    """
    
    print("Gradient 기반 중요도 계산 중...")
    
    head_scores = {}
    
    for layer_idx, layer_gradients in gradients.items():
        layer_head_scores = {}
        
        for head_idx, head_gradient_list in layer_gradients.items():
            if len(head_gradient_list) > 0:
                # 평균 gradient 값을 중요도로 사용
                importance = np.mean(head_gradient_list)
                
                # 유효성 검증
                if np.isnan(importance) or np.isinf(importance):
                    layer_head_scores[head_idx] = 0.0
                else:
                    layer_head_scores[head_idx] = importance
            else:
                layer_head_scores[head_idx] = 0.0
        
        head_scores[layer_idx] = layer_head_scores
    
    total_heads = sum(len(scores) for scores in head_scores.values())
    valid_heads = sum(len([s for s in scores.values() if s > 0]) for scores in head_scores.values())
    
    print(f"중요도 계산 완료: {total_heads}개 헤드 중 {valid_heads}개 유효")
    
    return head_scores


def select_heads_to_prune(head_scores, ratio=None):
    """
    중요도 낮은 N% 선택
    
    Args:
        head_scores: calculate_head_importance()의 출력
        ratio: 프루닝할 비율 (None이면 PRUNE_RATIO 사용)
        
    Returns:
        heads_to_prune: List[(layer, head)] 프루닝할 헤드들
    """
    
    if ratio is None:
        ratio = PRUNE_RATIO
    
    print(f"중요도 정렬 및 하위 {ratio:.1%} 선택")
    
    if not head_scores:
        print("유효한 헤드 점수가 없습니다!")
        return []
    
    # 모든 헤드를 (중요도, 레이어, 헤드) 형태로 수집
    all_heads = []
    zero_heads = []
    
    for layer_idx, layer_heads in head_scores.items():
        for head_idx, importance in layer_heads.items():
            if importance > 0:
                all_heads.append((importance, layer_idx, head_idx))
            else:
                zero_heads.append((0.0, layer_idx, head_idx))
    
    print(f"  유효한 헤드: {len(all_heads)}개")
    print(f"  무효한 헤드: {len(zero_heads)}개")
    
    if len(all_heads) == 0:
        print("프루닝할 유효한 헤드가 없습니다!")
        return []
    
    # 중요도 기준 오름차순 정렬 (낮은 중요도가 앞에)
    all_heads.sort(key=lambda x: x[0])
    
    # 무효한 헤드를 먼저 프루닝하고 그 다음에 낮은 중요도 헤드
    combined_heads = zero_heads + all_heads
    
    total_heads = len(combined_heads)
    num_to_prune = int(total_heads * ratio)
    
    print(f"  전체 헤드: {total_heads}개")
    print(f"  프루닝 예정: {num_to_prune}개")
    
    # 하위 N% 선택
    heads_to_prune = []
    for i in range(min(num_to_prune, len(combined_heads))):
        importance, layer_idx, head_idx = combined_heads[i]
        heads_to_prune.append((layer_idx, head_idx))
    
    # 통계 출력
    pruned_by_layer = {}
    for layer_idx, head_idx in heads_to_prune:
        pruned_by_layer[layer_idx] = pruned_by_layer.get(layer_idx, 0) + 1
    
    print(f"\n레이어별 프루닝 분포:")
    for layer_idx in sorted(head_scores.keys()):
        pruned = pruned_by_layer.get(layer_idx, 0)
        total_in_layer = len(head_scores[layer_idx])
        percentage = pruned / total_in_layer * 100 if total_in_layer > 0 else 0
        print(f"  레이어 {layer_idx:2d}: {pruned:2d}/{total_in_layer} ({percentage:5.1f}%)")
    
    return heads_to_prune


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Attention Heads Gradient 중요도 분석")
    print("   (원본 모델 직접 프루닝)")
    print("=" * 60)
    
    # 디버그 모드 설정
    if DEBUG_MODE:
        MAX_SAMPLES = 10
        print("⚠ DEBUG_MODE 활성화 - 10개 샘플만 사용")
    
    print(f"\n설정:")
    print(f"  프루닝 비율: {PRUNE_RATIO:.1%}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  최대 길이: {MAX_LENGTH}")
    print(f"  샘플 수: {'전체' if MAX_SAMPLES is None else MAX_SAMPLES}")
    print(f"  디바이스: {DEVICE}")
    print(f"  데이터 타입: {DTYPE}")
    print()
    
    try:
        # 1. 모델 로드
        print(f"모델 로드: {MODEL_PATH}")
        model, tokenizer = load_model(MODEL_PATH)
        
        # 2. 데이터 로드
        print(f"데이터 로드: {DATA_PATH}")
        data_samples = load_json(DATA_PATH)
        
        # MAX_SAMPLES 제한
        if MAX_SAMPLES is not None:
            data_samples = data_samples[:MAX_SAMPLES]
        
        print(f"  샘플 수: {len(data_samples)}개")
        
        # 3. Gradient 측정
        gradients = measure_head_gradient(model, tokenizer, data_samples)
        
        # 4. 중요도 계산
        scores = calculate_head_importance(gradients)
        
        if not scores:
            print("중요도 계산 실패!")
            exit(1)
        
        # 5. 프루닝 대상 선택
        targets = select_heads_to_prune(scores)
        
        if not targets:
            print("프루닝 대상 선택 실패!")
            exit(1)
        
        # 6. 결과 저장
        print(f"\n결과 저장: {OUTPUT_FILE}")
        
        result_data = []
        for layer_idx, head_idx in targets:
            result_data.append({
                "layer": layer_idx,
                "head": head_idx
            })
        
        save_json(result_data, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Attention heads gradient 분석 완료!")
        print(f"원본 모델: {MODEL_PATH}")
        print(f"측정 데이터: {len(data_samples)}개 샘플")
        print(f"프루닝 비율: {PRUNE_RATIO:.1%}")
        print(f"프루닝 헤드 수: {len(targets)}개")
        print(f"결과 파일: {OUTPUT_FILE}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
