import torch
import numpy as np
from tqdm import tqdm
import sys
import os

# 작업 디렉토리를 프로젝트 루트로 변경
os.chdir('/content/drive/MyDrive/smartfarm_pruning')

# 현재 디렉토리를 Python 경로에 추가
sys.path.append('.')

# 프로젝트 루트를 Python 경로에 추가
project_root = "/content/drive/MyDrive/smartfarm_pruning"
if project_root not in sys.path:
    sys.path.append(project_root)

# utils 함수들 import (지침에 따라)
from src.utils.model_loader import load_model
from src.utils.data_loader import save_json

# 프루닝 설정
PRUNE_RATIO = 0.05  # 5% 프루닝
MODEL_PATH = "models/original/gemma-3-1b-pt"
OUTPUT_FILE = "results/magnitude_heads_to_prune.json"


def calculate_head_importance(model):
    """
    모든 레이어의 attention heads 중요도 계산
    
    Args:
        model: 로드된 모델
        
    Returns:
        head_scores: Dict[layer_idx][head_idx] = score
    """
    
    print("🔍 Attention heads 중요도 계산 중...")
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)
    head_dim = config.hidden_size // num_q_heads
    
    print(f"  분석 대상: {num_layers}개 레이어 × {num_q_heads}개 헤드 = {num_layers * num_q_heads}개")
    
    head_scores = {}
    model.eval()
    
    with torch.no_grad():
        for layer_idx in tqdm(range(num_layers), desc="레이어 순회"):
            
            try:
                # 각 헤드의 Q, K, V 가중치 추출
                attention = model.model.layers[layer_idx].self_attn
                
                q_weight = attention.q_proj.weight.data
                k_weight = attention.k_proj.weight.data
                v_weight = attention.v_proj.weight.data
                
                # GQA 구조 처리: K,V를 Q 헤드 수에 맞게 확장
                if num_kv_heads < num_q_heads:
                    repeat_factor = num_q_heads // num_kv_heads
                    k_expanded = []
                    v_expanded = []
                    
                    for kv_head_idx in range(num_kv_heads):
                        k_start = kv_head_idx * head_dim
                        k_end = (kv_head_idx + 1) * head_dim
                        v_start = kv_head_idx * head_dim
                        v_end = (kv_head_idx + 1) * head_dim
                        
                        k_head = k_weight[k_start:k_end, :]
                        v_head = v_weight[v_start:v_end, :]
                        
                        for _ in range(repeat_factor):
                            k_expanded.append(k_head)
                            v_expanded.append(v_head)
                    
                    k_weight_expanded = torch.cat(k_expanded, dim=0)
                    v_weight_expanded = torch.cat(v_expanded, dim=0)
                else:
                    k_weight_expanded = k_weight
                    v_weight_expanded = v_weight
                
            except Exception as e:
                print(f"⚠️ 레이어 {layer_idx} 스킵: {e}")
                continue
            
            # 각 헤드별 절댓값 평균 계산
            layer_head_scores = {}
            
            for head_idx in range(num_q_heads):
                try:
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim
                    
                    q_head = q_weight[start:end, :]
                    k_head = k_weight_expanded[start:end, :]
                    v_head = v_weight_expanded[start:end, :]
                    
                    # 절댓값 평균 계산
                    q_magnitude = torch.mean(torch.abs(q_head)).item()
                    k_magnitude = torch.mean(torch.abs(k_head)).item()
                    v_magnitude = torch.mean(torch.abs(v_head)).item()
                    
                    # 유효성 검증
                    if (np.isnan(q_magnitude) or np.isnan(k_magnitude) or np.isnan(v_magnitude) or
                        np.isinf(q_magnitude) or np.isinf(k_magnitude) or np.isinf(v_magnitude)):
                        layer_head_scores[head_idx] = 0.0
                        continue
                    
                    # Q, K, V 평균으로 전체 중요도
                    importance = (q_magnitude + k_magnitude + v_magnitude) / 3.0
                    layer_head_scores[head_idx] = importance
                    
                except Exception as e:
                    layer_head_scores[head_idx] = 0.0
            
            head_scores[layer_idx] = layer_head_scores
    
    total_heads = sum(len(scores) for scores in head_scores.values())
    valid_heads = sum(len([s for s in scores.values() if s > 0]) for scores in head_scores.values())
    
    print(f" 분석 완료: {total_heads}개 헤드 중 {valid_heads}개 유효")
    
    return head_scores


def select_heads_to_prune(head_scores, ratio=0.05):
    """
    중요도 낮은 N% 선택
    
    Args:
        head_scores: calculate_head_importance()의 출력
        ratio: 프루닝할 비율 (기본값: 30%)
        
    Returns:
        heads_to_prune: List[(layer, head)] 프루닝할 헤드들
    """
    
    print(f" 중요도 정렬 및 하위 {ratio:.1%} 선택")
    
    if not head_scores:
        print("❌ 유효한 헤드 점수가 없습니다!")
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
        print("❌ 프루닝할 유효한 헤드가 없습니다!")
        return []
    
    # 중요도 기준 오름차순 정렬 (낮은 중요도가 앞에)
    all_heads.sort(key=lambda x: x[0])
    
    # 무효한 헤드를 먼저 프루닝하고, 그 다음에 낮은 중요도 헤드
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
    
    print(f"\n 레이어별 프루닝 분포:")
    for layer_idx in sorted(head_scores.keys()):
        pruned = pruned_by_layer.get(layer_idx, 0)
        total_in_layer = len(head_scores[layer_idx])
        percentage = pruned / total_in_layer * 100
        print(f"  레이어 {layer_idx:2d}: {pruned:2d}/{total_in_layer} ({percentage:5.1f}%)")
    
    return heads_to_prune


# 테스트 코드 
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Attention Heads Magnitude 중요도 분석")
    print("=" * 60)
    
    try:
        # 1. 모델 로드 (utils 함수 사용)
        print(f" 모델 로드: {MODEL_PATH}")
        model, tokenizer = load_model(MODEL_PATH)
        
        # 2. 중요도 계산
        scores = calculate_head_importance(model)
        
        if not scores:
            print("❌ 중요도 계산 실패!")
            exit(1)
        
        # 3. 프루닝 대상 선택
        targets = select_heads_to_prune(scores, PRUNE_RATIO)
        
        if not targets:
            print("❌ 프루닝 대상 선택 실패!")
            exit(1)
        
        # 4. 결과 저장 
        print(f"\n 결과 저장: {OUTPUT_FILE}")
        
        # [{"layer": x, "head": y}, ...]
        result_data = []
        for layer_idx, head_idx in targets:
            result_data.append({
                "layer": layer_idx,
                "head": head_idx
            })
        
        save_json(result_data, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print(" Attention heads 분석 완료!")
        print(f" 프루닝 비율: {PRUNE_RATIO:.1%}")
        print(f" 프루닝 헤드 수: {len(targets)}개")
        print(f" 결과 파일: {OUTPUT_FILE}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()