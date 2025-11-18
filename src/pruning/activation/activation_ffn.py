import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import gc

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.model_loader import load_model
from src.utils.data_loader import load_json, save_json
from transformers import AutoTokenizer


# ========== Configuration ==========
# 프루닝 강도 관련
PRUNE_RATIO = 0.25  # 가장 중요
# 범위: 0.05 ~ 0.40
# - 0.05 (5%): 매우 보수적, 성능 유지 우선
# - 0.10 (10%): 안전한 선택
# - 0.20 (20%): 균형잡힌 선택 (현재)
# - 0.30 (30%): 공격적, 크기 감소 우선
# - 0.40 (40%): 매우 공격적, 파인튜닝 필수

MAX_PRUNE_PER_LAYER = 0.45  # 레이어당 최대 제거 비율
# 범위: 0.10 ~ 0.70
# - 0.10: 매우 보수적, 레이어 기능 보존
# - 0.30: 균형잡힌 선택
# - 0.50: 현재 설정, 절반까지 허용
# - 0.70: 공격적, 특정 레이어 크게 축소 가능

MIN_NEURONS_PER_LAYER = 400  # 레이어당 최소 유지 뉴런
# 범위: 50 ~ 2000
# - 50: 극도로 작게 허용 (위험)
# - 100: 현재 설정 (최소한의 기능)
# - 500: 안전한 선택
# - 1000: 보수적
# - 2000: 매우 보수적

# 중요도 계산 가중치 (합이 1.0이 되도록 설정)
IMPORTANCE_WEIGHTS = {
    'mean_activation': 0.8,
    'max_activation': 0.1,
    'activity_ratio': 0.1
}

# 조정 예시:
# - 안정성 중시: {'mean': 0.8, 'max': 0.1, 'activity': 0.1}
# - 중요 패턴 중시: {'mean': 0.4, 'max': 0.5, 'activity': 0.1}
# - Dead neuron 제거 중시: {'mean': 0.3, 'max': 0.2, 'activity': 0.5}

# Data Processing
BATCH_SIZE = 4  # GPU 메모리에 따라
# - 1: 매우 적은 메모리 (T4 GPU)
# - 4: 현재 설정 (균형)
# - 8: 충분한 메모리 (A100)
# - 16: 대용량 메모리

MAX_SAMPLES = None  # 중요도 측정용 샘플 수
# - 100: 빠른 테스트
# - 400: 현재 데이터 전체
# - 1000: 더 정확한 측정
# - None: 모든 데이터 사용

MAX_LENGTH = 512  # 토큰 길이
# - 256: 짧은 문장, 빠른 처리
# - 512: 현재 설정 (균형)
# - 1024: 긴 문맥 고려

# 활성화 측정 방식
ACTIVATION_THRESHOLD = 1e-6  # Dead neuron 판단 임계값
USE_SWIGLU = True  # SwiGLU 전체 측정 여부 (True 권장)

# 레이어별 차별화
LAYER_IMPORTANCE_SCALE = {
    'early': 0.2,
    'middle': 1.0,
    'late': 2.2
}

# Paths
MODEL_PATH = "models/original/gemma-3-4b-it"
DATA_PATH = "data/split/pruning_activation.json"
OUTPUT_PATH = "results/activation_neurons_to_prune_it.json"
STATS_OUTPUT_PATH = "results/activation_neuron_stats.json"  # 선택적 통계 저장

# Technical Settings  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32  # float16은 NaN 위험
ATTENTION_IMPLEMENTATION = "eager"  # "eager" 또는 "sdpa"

# Memory Management
CLEAR_CACHE_EVERY = 10  # N 배치마다 GPU 캐시 정리
GC_COLLECT_EVERY = 10   # N 배치마다 가비지 컬렉션

# Debug Settings
VERBOSE = True  # 상세 출력 여부
SAVE_INTERMEDIATE_STATS = True  # 중간 통계 저장 여부  
DEBUG_MODE = False  # 디버그 모드 (소량 데이터로 테스트)

# Debug mode overrides
if DEBUG_MODE:
    MAX_SAMPLES = 10
    VERBOSE = True
    print("DEBUG MODE: Using only 10 samples")
# ====================================


def measure_neuron_activation(model, tokenizer, data_samples):
    """
    실제 데이터로 각 뉴런의 활성화 측정
    
    Args:
        model: Gemma 모델
        tokenizer: 토크나이저
        data_samples: 평가 데이터
        
    Returns:
        neuron_statistics: 뉴런별 활성화 통계
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 샘플 수 제한
    if MAX_SAMPLES:
        data_samples = data_samples[:MAX_SAMPLES]
        if VERBOSE:
            print(f"Using {MAX_SAMPLES} samples (limited by MAX_SAMPLES)")
    
    layers = model.model.layers
    num_layers = len(layers)
    
    # 레이어별 통계 초기화
    neuron_stats = {}
    for layer_idx in range(num_layers):
        intermediate_size = layers[layer_idx].mlp.gate_proj.weight.shape[0]
        neuron_stats[layer_idx] = {
            'sum': torch.zeros(intermediate_size).to(device),
            'sq_sum': torch.zeros(intermediate_size).to(device),
            'max': torch.zeros(intermediate_size).to(device),
            'zero_count': torch.zeros(intermediate_size).to(device),
            'count': 0
        }
    
    # Hook 설정
    hooks = []
    captured_activations = {}
    
    def create_hook(layer_idx, proj_type):
        def hook_fn(module, input, output):
            if output is not None:
                captured_activations[f"{layer_idx}_{proj_type}"] = output.detach()
        return hook_fn
    
    # Hook 등록
    for layer_idx, layer in enumerate(layers):
        if USE_SWIGLU:
            # gate와 up projection 모두 hook
            hook_gate = layer.mlp.gate_proj.register_forward_hook(
                create_hook(layer_idx, 'gate')
            )
            hook_up = layer.mlp.up_proj.register_forward_hook(
                create_hook(layer_idx, 'up')
            )
            hooks.extend([hook_gate, hook_up])
        else:
            # gate projection만 hook
            hook_gate = layer.mlp.gate_proj.register_forward_hook(
                create_hook(layer_idx, 'gate')
            )
            hooks.append(hook_gate)
    
    # 데이터 처리
    num_batches = (len(data_samples) + BATCH_SIZE - 1) // BATCH_SIZE
    
    if VERBOSE:
        print(f"Processing {len(data_samples)} samples in {num_batches} batches")
        print(f"Batch size: {BATCH_SIZE}, Max length: {MAX_LENGTH}")
        print(f"SwiGLU measurement: {USE_SWIGLU}")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Measuring activations", disable=not VERBOSE):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(data_samples))
            batch_samples = data_samples[batch_start:batch_end]
            
            # 텍스트 준비
            texts = []
            for sample in batch_samples:
                text = f"{sample['instruction']} {sample['output']}"
                texts.append(text)
            
            try:
                # 토큰화
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH
                ).to(device)
                
                # Forward pass
                captured_activations.clear()
                _ = model(**inputs)
                
                # 활성화 통계 업데이트
                for layer_idx in range(num_layers):
                    if USE_SWIGLU:
                        # SwiGLU activation 계산
                        gate_key = f"{layer_idx}_gate"
                        up_key = f"{layer_idx}_up"
                        
                        if gate_key in captured_activations and up_key in captured_activations:
                            gate_out = captured_activations[gate_key]
                            up_out = captured_activations[up_key]
                            # SwiGLU: gate * silu(up)
                            activation = gate_out * F.silu(up_out)
                        else:
                            continue
                    else:
                        # Gate projection만 사용
                        gate_key = f"{layer_idx}_gate"
                        if gate_key in captured_activations:
                            activation = captured_activations[gate_key]
                        else:
                            continue
                    
                    # 통계 계산 (배치와 시퀀스 차원에 대한 평균)
                    neuron_act = activation.abs().mean(dim=[0, 1])  # [intermediate_size]
                    neuron_max = activation.abs().max(dim=0)[0].max(dim=0)[0]  # [intermediate_size]
                    
                    stats = neuron_stats[layer_idx]
                    stats['sum'] += neuron_act
                    stats['sq_sum'] += neuron_act ** 2
                    stats['max'] = torch.maximum(stats['max'], neuron_max)
                    stats['zero_count'] += (neuron_act < ACTIVATION_THRESHOLD).float()
                    stats['count'] += 1
                
            except Exception as e:
                if VERBOSE:
                    print(f"Warning: Batch {batch_idx} processing error: {e}")
                continue
            
            # 메모리 정리
            if batch_idx % CLEAR_CACHE_EVERY == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if batch_idx % GC_COLLECT_EVERY == 0:
                gc.collect()
    
    # Hook 제거
    for hook in hooks:
        hook.remove()
    
    # 최종 통계 계산
    if VERBOSE:
        print("\nComputing final statistics...")
    
    final_stats = {}
    for layer_idx, stats in neuron_stats.items():
        if stats['count'] > 0:
            mean_act = stats['sum'] / stats['count']
            var_act = (stats['sq_sum'] / stats['count']) - (mean_act ** 2)
            std_act = torch.sqrt(torch.clamp(var_act, min=1e-8))
            dead_ratio = stats['zero_count'] / stats['count']
            
            final_stats[f"layer_{layer_idx}"] = {
                'mean': mean_act.cpu().numpy(),
                'std': std_act.cpu().numpy(), 
                'max': stats['max'].cpu().numpy(),
                'dead_ratio': dead_ratio.cpu().numpy()
            }
            
            if VERBOSE:
                print(f"  Layer {layer_idx}: "
                      f"mean={mean_act.mean():.4f}, "
                      f"std={std_act.mean():.4f}, "
                      f"dead={(dead_ratio > 0.9).sum().item()}/{len(dead_ratio)}")
    
    return final_stats


def calculate_neuron_importance(neuron_statistics):
    """
    활성화 통계로부터 중요도 점수 계산
    
    Args:
        neuron_statistics: 각 레이어의 뉴런별 통계
        
    Returns:
        neuron_scores: 정규화된 중요도 점수
    """
    if VERBOSE:
        print("\nCalculating importance scores...")
    
    neuron_scores = {}
    
    for layer_key, stats in neuron_statistics.items():

        layer_idx = int(layer_key.split('_')[1])

        mean_act = np.array(stats['mean'])
        max_act = np.array(stats['max'])
        dead_ratio = np.array(stats['dead_ratio'])
        
        # 각 요소 정규화
        mean_norm = mean_act / (mean_act.max() + 1e-8)
        max_norm = max_act / (max_act.max() + 1e-8)
        activity = 1.0 - dead_ratio
        
        # 가중치 기반 중요도 계산
        importance = (
            IMPORTANCE_WEIGHTS['mean_activation'] * mean_norm +
            IMPORTANCE_WEIGHTS['max_activation'] * max_norm +
            IMPORTANCE_WEIGHTS['activity_ratio'] * activity
        )

        # 레이어별 스케일 적용
        if layer_idx <= 8:
            layer_scale = LAYER_IMPORTANCE_SCALE['early']
        elif layer_idx <= 17:
            layer_scale = LAYER_IMPORTANCE_SCALE['middle']
        else:
            layer_scale = LAYER_IMPORTANCE_SCALE['late']

        # 스케일 적용 (높은 스케일 = 더 중요 = 프루닝 덜 됨)
        importance = importance * layer_scale
        
        neuron_scores[layer_key] = importance.tolist()
        
        if VERBOSE:
            print(f"  Layer {layer_idx}: "
                  f"importance range [{importance.min():.4f}, {importance.max():.4f}] "
                  f"(scale: {layer_scale})")

    return neuron_scores


def select_neurons_to_prune(neuron_scores):
    """
    중요도가 낮은 하위 PRUNE_RATIO% 뉴런 선택
    
    Args:
        neuron_scores: 각 레이어의 뉴런별 중요도 점수
        
    Returns:
        neurons_to_prune: 제거할 뉴런 목록
    """
    if VERBOSE:
        print(f"\nSelecting neurons to prune (ratio: {PRUNE_RATIO})...")
    
    # 모든 뉴런 수집
    all_neurons = []
    layer_neuron_counts = {}
    
    for layer_key, scores in neuron_scores.items():
        layer_idx = int(layer_key.split('_')[1])
        layer_neuron_counts[layer_idx] = len(scores)
        
        for neuron_idx, score in enumerate(scores):
            all_neurons.append({
                'layer': layer_idx,
                'neuron': neuron_idx,
                'score': score
            })
    
    # 중요도 정렬 (오름차순)
    all_neurons.sort(key=lambda x: x['score'])
    
    # 프루닝 대상 선택 (레이어별 최소 뉴런 수 고려)
    total_neurons = len(all_neurons)
    target_prune_count = int(total_neurons * PRUNE_RATIO)
    
    neurons_to_prune = []
    layer_prune_counts = {i: 0 for i in range(len(neuron_scores))}
    
    for neuron in all_neurons:
        layer_idx = neuron['layer']
        current_layer_neurons = layer_neuron_counts[layer_idx]
        pruned_in_layer = layer_prune_counts[layer_idx]
        
        # 레이어별 제약 확인
        remaining_after_prune = current_layer_neurons - pruned_in_layer - 1

        # 1. 최소 뉴런 수 확인
        if remaining_after_prune < MIN_NEURONS_PER_LAYER:
            continue
            
        # 2. 최대 프루닝 비율 확인 
        max_prunable_in_layer = int(current_layer_neurons * MAX_PRUNE_PER_LAYER)
        if pruned_in_layer >= max_prunable_in_layer:
            continue
        
        neurons_to_prune.append({
            'layer': layer_idx,
            'neuron': neuron['neuron']
        })
        layer_prune_counts[layer_idx] += 1
        
        if len(neurons_to_prune) >= target_prune_count:
            break
    
    # 통계 출력
    print(f"\nPruning Statistics:")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Target prune count: {target_prune_count}")
    print(f"  Actually pruning: {len(neurons_to_prune)}")
    print(f"  Actual ratio: {len(neurons_to_prune)/total_neurons:.1%}")
    
    if VERBOSE:
        print("\nNeurons to prune per layer:")
        for layer_idx in sorted(layer_prune_counts.keys()):
            total_in_layer = layer_neuron_counts[layer_idx]
            prune_count = layer_prune_counts[layer_idx]
            if prune_count > 0:
                prune_pct = (prune_count / total_in_layer) * 100
                print(f"  Layer {layer_idx}: {prune_count}/{total_in_layer} ({prune_pct:.1f}%)")
    
    return neurons_to_prune


def save_statistics(neuron_statistics, neuron_scores, output_path):
    """중간 통계 저장 (선택적)"""
    if not SAVE_INTERMEDIATE_STATS:
        return
    
    stats_data = {
        'config': {
            'prune_ratio': PRUNE_RATIO,
            'batch_size': BATCH_SIZE,
            'max_samples': MAX_SAMPLES,
            'use_swiglu': USE_SWIGLU,
            'importance_weights': IMPORTANCE_WEIGHTS
        },
        'layer_statistics': {},
        'layer_scores': {}
    }
    
    # 통계를 JSON 직렬화 가능한 형태로 변환
    for layer_key, stats in neuron_statistics.items():
        layer_idx = int(layer_key.split('_')[1])
        stats_data['layer_statistics'][layer_idx] = {
            'mean_activation': float(np.mean(stats['mean'])),
            'max_activation': float(np.mean(stats['max'])),
            'dead_ratio': float(np.mean(stats['dead_ratio'])),
            'num_dead': int((np.array(stats['dead_ratio']) > 0.9).sum())
        }
    
    for layer_key, scores in neuron_scores.items():
        layer_idx = int(layer_key.split('_')[1])
        stats_data['layer_scores'][layer_idx] = {
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'mean_score': float(np.mean(scores))
        }
    
    save_json(stats_data, output_path)
    if VERBOSE:
        print(f"Statistics saved to {output_path}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("FFN Activation-based Pruning")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Prune ratio: {PRUNE_RATIO:.1%}")
    print(f"  Min neurons per layer: {MIN_NEURONS_PER_LAYER}")
    print(f"  Use SwiGLU: {USE_SWIGLU}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    
    # 1. 데이터 로드
    print(f"\n[1/5] Loading data from {DATA_PATH}...")
    data_samples = load_json(DATA_PATH)
    print(f"Loaded {len(data_samples)} samples")
    
    # 2. 모델 로드
    print(f"\n[2/5] Loading model from {MODEL_PATH}...")
    result = load_model(MODEL_PATH, device=DEVICE, dtype=DTYPE)
    
    if isinstance(result, tuple):
        model, tokenizer = result
    else:
        model = result
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 활성화 측정
    print(f"\n[3/5] Measuring neuron activations...")
    neuron_statistics = measure_neuron_activation(model, tokenizer, data_samples)
    
    # 메모리 정리
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 4. 중요도 계산
    print(f"\n[4/5] Calculating importance scores...")
    neuron_scores = calculate_neuron_importance(neuron_statistics)
    
    # 5. 프루닝 대상 선택
    neurons_to_prune = select_neurons_to_prune(neuron_scores)
    
    # 6. 결과 저장
    print(f"\n[5/5] Saving results...")
    save_json(neurons_to_prune, OUTPUT_PATH)
    print(f"Results saved to {OUTPUT_PATH}")
    
    # 선택적: 통계 저장
    if SAVE_INTERMEDIATE_STATS:
        save_statistics(neuron_statistics, neuron_scores, STATS_OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Pruning target selection completed!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total neurons to prune: {len(neurons_to_prune)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
