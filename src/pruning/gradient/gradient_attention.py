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

# 7. 통계 수집 옵션 (NEW)
COLLECT_STATISTICS = True  # 상세 통계 수집 여부
GRADIENT_THRESHOLD = 1e-6  # Dead head 판단 임계값
# ====================================


def measure_head_gradient(model, tokenizer, data_samples):
    """
    Gradient 기반으로 attention heads 중요도 측정 (통계 강화)
    
    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        data_samples: 중요도 측정용 데이터 샘플들
        
    Returns:
        head_statistics: Dict[layer_idx][head_idx] = {mean, std, max, count}
    """
    
    print("Attention heads gradient 측정 중 (통계 수집)...")
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_q_heads
    
    print(f"  분석 대상: {num_layers}개 레이어 × {num_q_heads}개 헤드")
    print(f"  데이터 샘플: {len(data_samples)}개")
    print(f"  배치 크기: {BATCH_SIZE}, 최대 길이: {MAX_LENGTH}")
    
    # 각 헤드의 통계 정보 수집
    head_statistics = {}
    for layer_idx in range(num_layers):
        head_statistics[layer_idx] = {}
        for head_idx in range(num_q_heads):
            head_statistics[layer_idx][head_idx] = {
                'gradients': [],  # 각 배치의 gradient 값들
                'count': 0
            }
    
    # Hook: attention output에서 각 헤드별 gradient 측정
    def create_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            try:
                if grad_output and grad_output[0] is not None:
                    gradient = grad_output[0].detach()  # 메모리 효율을 위해 detach
                    
                    batch_size, seq_len, hidden_size = gradient.shape
                    
                    # [batch, seq_len, num_heads, head_dim]로 reshape
                    grad_per_head = gradient.view(batch_size, seq_len, num_q_heads, head_dim)
                    
                    # 각 헤드의 gradient 통계 계산
                    for head_idx in range(num_q_heads):
                        head_grad = grad_per_head[:, :, head_idx, :].abs()
                        mean_grad = head_grad.mean().item()
                        
                        # 통계 저장
                        head_statistics[layer_idx][head_idx]['gradients'].append(mean_grad)
                        head_statistics[layer_idx][head_idx]['count'] += 1
                        
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Hook error at layer {layer_idx}: {e}")
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
    model.train()
    
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
            
            # labels 생성
            labels = inputs["input_ids"].clone()
            
            # Forward + Backward pass
            model.zero_grad()
            outputs = model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            loss.backward()
            
            # 메모리 관리
            del inputs, labels, outputs, loss
            
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
    
    # 최종 통계 계산
    print("\n통계 계산 중...")
    final_statistics = {}
    
    for layer_idx in range(num_layers):
        final_statistics[layer_idx] = {}
        
        for head_idx in range(num_q_heads):
            grads = head_statistics[layer_idx][head_idx]['gradients']
            
            if len(grads) > 0:
                mean_val = np.mean(grads)
                std_val = np.std(grads)
                max_val = np.max(grads)
                min_val = np.min(grads)
                
                # Dead head 체크
                is_dead = mean_val < GRADIENT_THRESHOLD
                
                final_statistics[layer_idx][head_idx] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'max': float(max_val),
                    'min': float(min_val),
                    'count': len(grads),
                    'is_dead': bool(is_dead)
                }
            else:
                final_statistics[layer_idx][head_idx] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'count': 0,
                    'is_dead': True
                }
    
    # 수집 통계 출력
    total_heads = num_layers * num_q_heads
    collected = sum(1 for l in final_statistics.values() 
                   for h in l.values() if h['count'] > 0)
    dead_heads = sum(1 for l in final_statistics.values() 
                    for h in l.values() if h['is_dead'])
    
    print(f"Gradient 측정 완료")
    print(f"   수집된 헤드: {collected}/{total_heads}개")
    print(f"   Dead heads: {dead_heads}개 ({dead_heads/total_heads*100:.1f}%)")
    
    return final_statistics


def calculate_head_importance(statistics):
    """
    통계 정보로부터 중요도 계산 (다차원 평가)
    
    Args:
        statistics: measure_head_gradient()의 출력
        
    Returns:
        head_scores: Dict[layer_idx][head_idx] = score
    """
    
    print("다차원 중요도 계산 중...")
    
    head_scores = {}
    
    for layer_idx, layer_stats in statistics.items():
        layer_head_scores = {}
        
        for head_idx, stats in layer_stats.items():
            if stats['count'] > 0 and not stats['is_dead']:
                # 다차원 중요도 계산
                # 1. 평균 gradient (기본 중요도)
                mean_importance = stats['mean']
                
                # 2. 안정성 고려 (낮은 std = 더 안정적)
                stability_score = 1.0 / (1.0 + stats['std'])
                
                # 3. 최대값 고려 (중요한 순간 존재)
                max_score = stats['max']
                
                # 종합 중요도 (가중 평균)
                importance = (
                    0.6 * mean_importance +  # 평균 gradient
                    0.2 * stability_score +   # 안정성
                    0.2 * max_score           # 최대값
                )
                
                # 유효성 검증
                if np.isnan(importance) or np.isinf(importance):
                    importance = 0.0
                
                layer_head_scores[head_idx] = float(importance)
            else:
                layer_head_scores[head_idx] = 0.0
        
        head_scores[layer_idx] = layer_head_scores
    
    total_heads = sum(len(scores) for scores in head_scores.values())
    valid_heads = sum(len([s for s in scores.values() if s > 0]) for scores in head_scores.values())
    
    print(f"중요도 계산 완료: {total_heads}개 헤드 중 {valid_heads}개 유효")
    
    # 레이어별 통계 출력
    print("\n레이어별 중요도 분포:")
    for layer_idx in sorted(head_scores.keys()):
        scores = list(head_scores[layer_idx].values())
        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            print(f"  Layer {layer_idx:2d}: "
                  f"mean={np.mean(valid_scores):.6f}, "
                  f"std={np.std(valid_scores):.6f}, "
                  f"min={np.min(valid_scores):.6f}, "
                  f"max={np.max(valid_scores):.6f}")
    
    return head_scores


def select_heads_to_prune(head_scores, ratio=None):
    """
    중요도 낮은 N% 선택 (레이어별 분산 고려)
    
    Args:
        head_scores: calculate_head_importance()의 출력
        ratio: 프루닝할 비율 (None이면 PRUNE_RATIO 사용)
        
    Returns:
        heads_to_prune: List[(layer, head)] 프루닝할 헤드들
    """
    
    if ratio is None:
        ratio = PRUNE_RATIO
    
    print(f"중요도 정렬 및 하위 {ratio:.1%} 선택 (레이어 분산 고려)")
    
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
    print(f"  Dead heads: {len(zero_heads)}개")
    
    if len(all_heads) == 0:
        print("프루닝할 유효한 헤드가 없습니다!")
        return []
    
    # 중요도 기준 오름차순 정렬
    all_heads.sort(key=lambda x: x[0])
    
    # Dead heads를 먼저 프루닝
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
    
    # 레이어별 통계
    pruned_by_layer = {}
    for layer_idx, head_idx in heads_to_prune:
        pruned_by_layer[layer_idx] = pruned_by_layer.get(layer_idx, 0) + 1
    
    print(f"\n레이어별 프루닝 분포:")
    for layer_idx in sorted(head_scores.keys()):
        pruned = pruned_by_layer.get(layer_idx, 0)
        total_in_layer = len(head_scores[layer_idx])
        percentage = pruned / total_in_layer * 100 if total_in_layer > 0 else 0
        
        # 프루닝된 헤드의 평균 중요도
        pruned_heads_in_layer = [head_idx for l, h in heads_to_prune if l == layer_idx]
        if pruned_heads_in_layer:
            avg_importance = np.mean([head_scores[layer_idx][h] for h in pruned_heads_in_layer])
            print(f"  Layer {layer_idx:2d}: {pruned:2d}/{total_in_layer} ({percentage:5.1f}%) "
                  f"- avg importance: {avg_importance:.6f}")
        else:
            print(f"  Layer {layer_idx:2d}: {pruned:2d}/{total_in_layer} ({percentage:5.1f}%)")
    
    return heads_to_prune


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Attention Heads Gradient 중요도 분석 v2")
    print("   (통계 강화 + 메모리 최적화)")
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
    print(f"  통계 수집: {COLLECT_STATISTICS}")
    print(f"  Gradient 임계값: {GRADIENT_THRESHOLD}")
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
        
        # 3. Gradient 측정 (통계 포함)
        statistics = measure_head_gradient(model, tokenizer, data_samples)
        
        # 4. 중요도 계산 (다차원 평가)
        scores = calculate_head_importance(statistics)
        
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
        print("Attention heads gradient 분석 완료! (v2)")
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