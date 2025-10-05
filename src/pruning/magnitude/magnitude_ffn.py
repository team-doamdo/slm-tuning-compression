import torch
import json
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.model_loader import load_model
from src.utils.data_loader import save_json

# ===== 프루닝 비율 설정 =====
PRUNE_RATIO = 0.05  


def calculate_neuron_importance(model):
    """
    모든 레이어의 FFN neurons의 Magnitude 기반 중요도 계산
    
    방법: L2 Norm + Combined Projection
    - gate_proj와 up_proj를 곱셈으로 결합 (SwiGLU 구조 반영)
    - L2 norm으로 중요도 계산 (큰 가중치에 민감)
    
    Args:
        model: Gemma 모델 객체
        
    Returns:
        neuron_scores: Dict[str, List[float]] - 각 레이어의 뉴런별 중요도 점수
    """
    neuron_scores = {}
    
    # Gemma 모델의 레이어 접근
    layers = model.model.layers
    
    print(f"Total layers: {len(layers)}")
    print(f"Model config - hidden_size: {model.config.hidden_size}, intermediate_size: {model.config.intermediate_size}")
    
    for layer_idx, layer in enumerate(layers):
        print(f"Processing layer {layer_idx}...")
        
        # FFN 모듈 접근
        mlp = layer.mlp
        
        # gate_proj와 up_proj의 가중치 가져오기
        gate_weight = mlp.gate_proj.weight.data  # [intermediate_size, hidden_size]
        up_weight = mlp.up_proj.weight.data      # [intermediate_size, hidden_size]
        
        print(f"  gate_weight shape: {gate_weight.shape}")
        print(f"  up_weight shape: {up_weight.shape}")
        
        # SwiGLU 구조 반영: gate와 up의 곱셈
        # output = gate_proj(x) * SiLU(up_proj(x))에서 두 projection이 곱해짐
        combined_weight = gate_weight * up_weight
        
        # L2 norm 계산: sqrt(mean(weight^2))
        # dim=1로 각 뉴런(행)에 대해 hidden_size 차원으로 계산
        importance = torch.sqrt(torch.mean(combined_weight ** 2, dim=1))
        
        # CPU로 이동 후 리스트로 변환
        importance_list = importance.cpu().tolist()
        
        neuron_scores[f"layer_{layer_idx}"] = importance_list
        
        print(f"Layer {layer_idx}: {len(importance_list)} neurons (expected: {model.config.intermediate_size})")
    
    return neuron_scores


def select_neurons_to_prune(
    neuron_scores,
    ratio: float = 0.05,
    *,
    per_layer_floor: float = 0.02,   # 레이어별 최소 비율 가드 
    per_layer_cap: float = 0.08,     # 레이어별 최대 비율 가드 
    min_intermediate: int = 2048     # 레이어별 최소 잔존 뉴런 수 (SwiGLU 안정성 가드)
):
    """
    중요도(작을수록 덜 중요하다고 가정)에 기반해 FFN 뉴런을 '레이어별 quota' 방식으로 선택한다.
    - 호출부 시그니처 유지: neuron_scores, ratio 만 넣어도 동작 (나머지는 기본값)
    - ratio는 '글로벌 목표 비율'이 아니라 '레이어별 기본 비율'로 적용한다.
      (쏠림 방지: per_layer_floor/per_layer_cap으로 레이어별 컷 비율에 상/하한을 둠)
    - 각 레이어는 k = clamp(round(n * ratio), floor, cap) 만큼만 제거
    - k를 적용해도 잔존 뉴런(n - k)이 min_intermediate보다 작아지면 k를 줄여서 하한 보장
    - 반환 형식: [{"layer": layer_idx, "neuron": neuron_idx}, ...] (기존 JSON과 동일)

    파라미터
    --------
    neuron_scores : Dict[Union[int,str], List[float]]
        레이어별 뉴런 중요도 점수. 예:
        {
          0: [ ... ],
          "layer_1": [ ... ],
          "2": [ ... ],
        }
        키는 정수, "2", "layer_2" 등 혼용 가능 (내부에서 정규화)

    ratio : float, default 0.05
        레이어별 기본 프루닝 비율. (글로벌 비율이 아님)
        예: 0.05 => 각 레이어에서 약 5% 제거

    per_layer_floor : float, default 0.02
        레이어별 최소 프루닝 비율 가드 (2%)

    per_layer_cap : float, default 0.08
        레이어별 최대 프루닝 비율 가드 (8%)

    min_intermediate : int, default 2048
        프루닝 후에도 레이어의 intermediate(=뉴런 수)가 이 값보다 작아지지 않도록 가드

    반환
    ----
    List[Dict[str, int]]
        {"layer": layer_idx, "neuron": neuron_idx} 의 리스트
    """
    def _parse_layer_idx(k):
        if isinstance(k, int):
            return k
        if isinstance(k, str):
            s = k.strip().lower()
            # 뒤에서부터 숫자만 추출
            num = ""
            for ch in reversed(s):
                if ch.isdigit():
                    num = ch + num
                elif num:
                    break
            if num:
                try:
                    return int(num)
                except Exception:
                    pass
            # 전부 실패하면 0으로 폴백
            try:
                return int(s)
            except Exception:
                return 0
        # 알 수 없는 타입 → 0 폴백
        return 0

    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    results = []

    # 입력 정규화: Dict[int, List[float]] 로 변환
    normalized = {}
    for k, v in neuron_scores.items():
        li = _parse_layer_idx(k)
        # 점수 목록만 신뢰 (None/NaN 필터링)
        scores = [float(x) for x in v if x is not None]
        normalized[li] = scores

    # 레이어별 선택
    for layer_idx, scores in normalized.items():
        n = len(scores)
        if n == 0:
            continue

        # 레이어별 기본 k: round(n * ratio), 그리고 상/하한 가드
        base_k = int(round(n * float(ratio)))
        k_floor = int(round(n * float(per_layer_floor)))
        k_cap   = int(round(n * float(per_layer_cap)))

        k = _clamp(base_k, k_floor, k_cap)

        # 잔존 뉴런 하한 보장: n - k >= min_intermediate
        if (n - k) < min_intermediate:
            k = max(0, n - min_intermediate)

        if k <= 0:
            # 이 레이어에서는 프루닝 생략
            continue

        # 중요도 낮은(=작은) 순으로 k개 선택
        # 점수와 인덱스를 묶고 안정 정렬
        order = sorted(range(n), key=lambda i: (scores[i], i))
        pruned = order[:k]

        # 선택 결과 기록
        for idx in pruned:
            results.append({"layer": int(layer_idx), "neuron": int(idx)})

    return results


if __name__ == "__main__":
    print("Starting FFN Magnitude-based Pruning...")
    print(f"Pruning method: L2 Norm + Combined Projection")
    print(f"Pruning ratio: {PRUNE_RATIO*100}%")
    print("=" * 50)
    
    # 경로 설정
    model_path = "models/original/gemma-3-1b-pt"
    output_path = "results/magnitude_neurons_to_prune.json"
    
    # 1. 모델 로드
    print(f"\n[1/4] Loading model from {model_path}...")
    result = load_model(model_path)
    
    # load_model이 튜플을 반환하는 경우 처리
    if isinstance(result, tuple):
        model, tokenizer = result
    else:
        model = result
    
    print("Model loaded successfully")
    
    # 2. 중요도 계산
    print(f"\n[2/4] Calculating neuron importance...")
    neuron_scores = calculate_neuron_importance(model)
    print("Importance calculation completed")
    
    # 3. 프루닝 대상 선택
    print(f"\n[3/4] Selecting neurons to prune (ratio: {PRUNE_RATIO})...")
    neurons_to_prune = select_neurons_to_prune(neuron_scores)
    print("Selection completed")
    
    # 4. 결과 저장
    print(f"\n[4/4] Saving results to {output_path}...")
    save_json(neurons_to_prune, output_path)
    print("Results saved successfully")
    
    print("\n" + "=" * 50)
    print("FFN Magnitude-based Pruning completed!")