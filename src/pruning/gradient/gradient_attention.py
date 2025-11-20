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

# 7. 고급 설정 (NEW in v3)
COLLECT_STATISTICS = True
GRADIENT_THRESHOLD = 1e-6

# 레이어별 중요도 스케일링 (도메인 특화 지식 보존)
LAYER_IMPORTANCE_SCALE = {
    'early': 0.8,    # 초기 레이어 (0-8): 범용 패턴
    'middle': 1.0,   # 중간 레이어 (9-17): 기준
    'late': 1.5      # 후기 레이어 (18-25): 도메인 특화
}

# 중요도 계산 가중치 (미세 조정)
IMPORTANCE_WEIGHTS = {
    'mean_gradient': 0.5,      # 평균 gradient
    'max_gradient': 0.3,       # 최대 gradient (중요 순간)
    'stability': 0.1,          # 안정성 (낮은 분산)
    'activity': 0.1            # 활성도 (non-zero 비율)
}

# 프루닝 제약 조건
MIN_HEADS_PER_LAYER = 2    # 레이어당 최소 헤드 수
MAX_PRUNE_PER_LAYER = 0.5  # 레이어당 최대 프루닝 비율
# ====================================


def measure_head_gradient(model, tokenizer, data_samples):
    """
    Gradient 기반 attention heads 중요도 측정 (최종 최적화)
    
    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        data_samples: 중요도 측정용 데이터 샘플들
        
    Returns:
        head_statistics: Dict[layer_idx][head_idx] = {통계 정보}
    """
    
    print("Attention heads gradient 측정 중 (최적화 v3)...")
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_q_heads
    
    print(f"  분석 대상: {num_layers}개 레이어 × {num_q_heads}개 헤드")
    print(f"  데이터 샘플: {len(data_samples)}개")
    print(f"  배치 크기: {BATCH_SIZE}, 최대 길이: {MAX_LENGTH}")
    print(f"  레이어별 스케일링: early={LAYER_IMPORTANCE_SCALE['early']}, "
          f"middle={LAYER_IMPORTANCE_SCALE['middle']}, late={LAYER_IMPORTANCE_SCALE['late']}")
    
    # 통계 초기화 (효율적인 구조)
    head_statistics = {}
    for layer_idx in range(num_layers):
        head_statistics[layer_idx] = {}
        for head_idx in range(num_q_heads):
            head_statistics[layer_idx][head_idx] = {
                'sum': 0.0,           # gradient 합
                'sq_sum': 0.0,        # gradient 제곱합 (분산 계산용)
                'max': 0.0,           # 최대 gradient
                'zero_count': 0,      # zero gradient 횟수
                'count': 0            # 총 측정 횟수
            }
    
    # Hook 설정
    def create_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            try:
                if grad_output and grad_output[0] is not None:
                    gradient = grad_output[0].detach()
                    
                    batch_size, seq_len, hidden_size = gradient.shape
                    grad_per_head = gradient.view(batch_size, seq_len, num_q_heads, head_dim)
                    
                    for head_idx in range(num_q_heads):
                        head_grad = grad_per_head[:, :, head_idx, :].abs()
                        mean_grad = head_grad.mean().item()
                        max_grad = head_grad.max().item()
                        
                        stats = head_statistics[layer_idx][head_idx]
                        stats['sum'] += mean_grad
                        stats['sq_sum'] += mean_grad ** 2
                        stats['max'] = max(stats['max'], max_grad)
                        stats['zero_count'] += 1 if mean_grad < GRADIENT_THRESHOLD else 0
                        stats['count'] += 1
                        
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
            
            labels = inputs["input_ids"].clone()
            
            # Forward + Backward
            model.zero_grad()
            outputs = model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            loss.backward()
            
            # 메모리 효율 관리
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
    print("\n최종 통계 계산 중...")
    final_statistics = {}
    
    for layer_idx in range(num_layers):
        final_statistics[layer_idx] = {}
        
        for head_idx in range(num_q_heads):
            stats = head_statistics[layer_idx][head_idx]
            
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sq_sum'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(variance, 0))
                zero_ratio = stats['zero_count'] / stats['count']
                
                final_statistics[layer_idx][head_idx] = {
                    'mean': float(mean),
                    'std': float(std),
                    'max': float(stats['max']),
                    'zero_ratio': float(zero_ratio),
                    'count': stats['count'],
                    'is_dead': bool(mean < GRADIENT_THRESHOLD)
                }
            else:
                final_statistics[layer_idx][head_idx] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'zero_ratio': 1.0,
                    'count': 0,
                    'is_dead': True
                }
    
    # 통계 출력
    total_heads = num_layers * num_q_heads
    collected = sum(1 for l in final_statistics.values() 
                   for h in l.values() if h['count'] > 0)
    dead_heads = sum(1 for l in final_statistics.values() 
                    for h in l.values() if h['is_dead'])
    
    print(f"Gradient 측정 완료")
    print(f"   수집된 헤드: {collected}/{total_heads}개")
    print(f"   Dead heads: {dead_heads}개 ({dead_heads/total_heads*100:.1f}%)")
    
    return final_statistics


def calculate_head_importance(statistics, num_layers):
    """
    다차원 중요도 계산 + 레이어별 스케일링
    
    Args:
        statistics: measure_head_gradient()의 출력
        num_layers: 총 레이어 수
        
    Returns:
        head_scores: Dict[layer_idx][head_idx] = scaled_importance
    """
    
    print("레이어별 스케일링을 적용한 중요도 계산 중...")
    
    head_scores = {}
    
    for layer_idx, layer_stats in statistics.items():
        layer_head_scores = {}
        
        # 레이어 위치에 따른 스케일 결정
        if layer_idx <= num_layers // 3:
            layer_scale = LAYER_IMPORTANCE_SCALE['early']
            layer_type = 'early'
        elif layer_idx <= 2 * num_layers // 3:
            layer_scale = LAYER_IMPORTANCE_SCALE['middle']
            layer_type = 'middle'
        else:
            layer_scale = LAYER_IMPORTANCE_SCALE['late']
            layer_type = 'late'
        
        for head_idx, stats in layer_stats.items():
            if stats['count'] > 0 and not stats['is_dead']:
                # 1. 평균 gradient 중요도
                mean_score = stats['mean']
                
                # 2. 최대 gradient (중요 순간 감지)
                max_score = stats['max']
                
                # 3. 안정성 (낮은 분산 = 안정적)
                stability_score = 1.0 / (1.0 + stats['std'])
                
                # 4. 활성도 (non-zero 비율)
                activity_score = 1.0 - stats['zero_ratio']
                
                # 가중 평균
                base_importance = (
                    IMPORTANCE_WEIGHTS['mean_gradient'] * mean_score +
                    IMPORTANCE_WEIGHTS['max_gradient'] * max_score +
                    IMPORTANCE_WEIGHTS['stability'] * stability_score +
                    IMPORTANCE_WEIGHTS['activity'] * activity_score
                )
                
                # 레이어별 스케일링 적용
                scaled_importance = base_importance * layer_scale
                
                # 유효성 검증
                if np.isnan(scaled_importance) or np.isinf(scaled_importance):
                    scaled_importance = 0.0
                
                layer_head_scores[head_idx] = float(scaled_importance)
            else:
                layer_head_scores[head_idx] = 0.0
        
        head_scores[layer_idx] = layer_head_scores
        
        # 레이어별 통계
        valid_scores = [s for s in layer_head_scores.values() if s > 0]
        if valid_scores and VERBOSE:
            print(f"  Layer {layer_idx:2d} ({layer_type:6s}, scale={layer_scale:.1f}): "
                  f"mean={np.mean(valid_scores):.6f}, "
                  f"std={np.std(valid_scores):.6f}")
    
    total_heads = sum(len(scores) for scores in head_scores.values())
    valid_heads = sum(len([s for s in scores.values() if s > 0]) for scores in head_scores.values())
    
    print(f"중요도 계산 완료: {total_heads}개 헤드 중 {valid_heads}개 유효")
    
    return head_scores


def select_heads_to_prune(head_scores, num_layers, ratio=None):
    """
    고급 프루닝 전략: 레이어별 제약 + 중요도 기반 선택
    
    Args:
        head_scores: calculate_head_importance()의 출력
        num_layers: 총 레이어 수
        ratio: 프루닝할 비율
        
    Returns:
        heads_to_prune: List[(layer, head)] 프루닝할 헤드들
    """
    
    if ratio is None:
        ratio = PRUNE_RATIO
    
    print(f"고급 프루닝 전략 적용 (ratio={ratio:.1%})")
    print(f"   제약: 최소 {MIN_HEADS_PER_LAYER}개/레이어, "
          f"최대 {MAX_PRUNE_PER_LAYER:.0%} 프루닝/레이어")
    
    if not head_scores:
        print("유효한 헤드 점수가 없습니다!")
        return []
    
    # 헤드 수집
    all_heads = []
    zero_heads = []
    layer_head_counts = {}
    
    for layer_idx, layer_heads in head_scores.items():
        layer_head_counts[layer_idx] = len(layer_heads)
        
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
    
    # 중요도 정렬
    all_heads.sort(key=lambda x: x[0])
    combined_heads = zero_heads + all_heads
    
    total_heads = len(combined_heads)
    target_prune_count = int(total_heads * ratio)
    
    print(f"  전체 헤드: {total_heads}개")
    print(f"  목표 프루닝: {target_prune_count}개")
    
    # 레이어별 제약을 고려한 프루닝
    heads_to_prune = []
    layer_prune_counts = {i: 0 for i in range(num_layers)}
    
    for importance, layer_idx, head_idx in combined_heads:
        # 레이어별 제약 확인
        current_layer_heads = layer_head_counts[layer_idx]
        pruned_in_layer = layer_prune_counts[layer_idx]
        remaining = current_layer_heads - pruned_in_layer
        
        # 1. 최소 헤드 수 체크
        if remaining <= MIN_HEADS_PER_LAYER:
            continue
        
        # 2. 최대 프루닝 비율 체크
        max_prunable = int(current_layer_heads * MAX_PRUNE_PER_LAYER)
        if pruned_in_layer >= max_prunable:
            continue
        
        # 프루닝 추가
        heads_to_prune.append((layer_idx, head_idx))
        layer_prune_counts[layer_idx] += 1
        
        # 목표 달성 시 종료
        if len(heads_to_prune) >= target_prune_count:
            break
    
    # 상세 통계
    print(f"\n프루닝 결과:")
    print(f"  실제 프루닝: {len(heads_to_prune)}개 ({len(heads_to_prune)/total_heads:.1%})")
    print(f"\n레이어별 분포:")
    
    for layer_idx in sorted(layer_prune_counts.keys()):
        pruned = layer_prune_counts[layer_idx]
        total = layer_head_counts[layer_idx]
        percentage = pruned / total * 100 if total > 0 else 0
        
        if pruned > 0:
            # 프루닝된 헤드의 평균 중요도
            pruned_heads = [h for l, h in heads_to_prune if l == layer_idx]
            avg_importance = np.mean([head_scores[layer_idx][h] for h in pruned_heads])
            print(f"  Layer {layer_idx:2d}: {pruned:2d}/{total} ({percentage:5.1f}%) "
                  f"- avg importance: {avg_importance:.6f}")
    
    return heads_to_prune


# 전역 변수
VERBOSE = True  # 상세 출력


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Attention Heads Gradient 중요도 분석 v3")
    print("   (레이어별 스케일링 + 고급 프루닝 전략)")
    print("=" * 60)
    
    if DEBUG_MODE:
        MAX_SAMPLES = 10
        print("⚠ DEBUG_MODE 활성화 - 10개 샘플만 사용")
    
    print(f"\n설정:")
    print(f"  프루닝 비율: {PRUNE_RATIO:.1%}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  최대 길이: {MAX_LENGTH}")
    print(f"  샘플 수: {'전체' if MAX_SAMPLES is None else MAX_SAMPLES}")
    print(f"  디바이스: {DEVICE}")
    print(f"  레이어 스케일: early={LAYER_IMPORTANCE_SCALE['early']}, "
          f"middle={LAYER_IMPORTANCE_SCALE['middle']}, late={LAYER_IMPORTANCE_SCALE['late']}")
    print(f"  중요도 가중치: {IMPORTANCE_WEIGHTS}")
    print()
    
    try:
        # 1. 모델 로드
        print(f"모델 로드: {MODEL_PATH}")
        model, tokenizer = load_model(MODEL_PATH)
        num_layers = model.config.num_hidden_layers
        
        # 2. 데이터 로드
        print(f"데이터 로드: {DATA_PATH}")
        data_samples = load_json(DATA_PATH)
        
        if MAX_SAMPLES is not None:
            data_samples = data_samples[:MAX_SAMPLES]
        
        print(f"  샘플 수: {len(data_samples)}개")
        
        # 3. Gradient 측정
        statistics = measure_head_gradient(model, tokenizer, data_samples)
        
        # 4. 레이어별 스케일링을 적용한 중요도 계산
        scores = calculate_head_importance(statistics, num_layers)
        
        if not scores:
            print("중요도 계산 실패!")
            exit(1)
        
        # 5. 고급 프루닝 전략 적용
        targets = select_heads_to_prune(scores, num_layers)
        
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
        print("Attention heads gradient 분석 완료! (v3)")
        print(f"원본 모델: {MODEL_PATH}")
        print(f"측정 데이터: {len(data_samples)}개 샘플")
        print(f"프루닝 비율: {PRUNE_RATIO:.1%}")
        print(f"프루닝 헤드 수: {len(targets)}개")
        print(f"결과 파일: {OUTPUT_FILE}")
        print("\n주요 개선사항:")
        print("  - 레이어별 중요도 스케일링 (도메인 특화 지식 보존)")
        print("  - 다차원 중요도 평가 (mean, max, stability, activity)")
        print("  - 레이어별 프루닝 제약 (최소/최대 헤드 수)")
        print("  - 메모리 효율 최적화")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()