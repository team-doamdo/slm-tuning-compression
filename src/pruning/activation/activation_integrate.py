import torch
import json
import os
from pathlib import Path
import sys

# 프로젝트 루트 경로 
sys.path.append('/content/drive/MyDrive/smartfarm_pruning/1번째')
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import load_json, save_json
from src.utils.model_loader import load_model, save_pruned_model


# ========== Configuration ==========
# 프루닝 비율 설정 (이 값만 수정하면 파일명이 자동으로 변경됨)
PRUNE_RATIO = 0.30  # 5%

# 프루닝 비율에 따른 동적 경로 생성
PRUNE_RATIO_INT = int(PRUNE_RATIO * 100)  # 0.05 -> 5

# Paths
MODEL_PATH = "models/original/gemma-3-4b-it"
HEADS_PATH = f"results/activation_heads_to_prune.json"
NEURONS_PATH = f"results/pruned/activation/activation_neurons_to_prune_{PRUNE_RATIO_INT}.json"
OUTPUT_PATH = f"models/pruned/activation/pruned_activation_{PRUNE_RATIO_INT}"

# FFN 프루닝 파라미터 (activation_ffn.py와 동일하게 유지)
MIN_INTERMEDIATE = 400
EDGE_LAYERS_PROTECT = 2
EDGE_RATIO_SCALE = 0.5
MAX_LAYER_PRUNE_RATIO = 0.45
# ====================================


def get_model_layers(model):
    """모델 구조에 따라 layers 접근 (멀티모달/텍스트 전용 모델 지원)"""
    if hasattr(model, 'language_model'):
        # 멀티모달 모델 (Gemma3ForConditionalGeneration) - 4B 등
        return model.language_model.layers
    else:
        # 텍스트 전용 모델 (Gemma3ForCausalLM) - 1B 등
        return model.model.layers


def get_head_dim(model):
    """모델에서 head_dim 가져오기"""
    if hasattr(model.config, 'head_dim'):
        return model.config.head_dim
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'head_dim'):
        return model.config.text_config.head_dim
    else:
        if hasattr(model.config, 'text_config'):
            config = model.config.text_config
        else:
            config = model.config
        return config.hidden_size // config.num_attention_heads


def load_pruning_targets(heads_path, neurons_path):
    """프루닝 대상 로드"""
    print("Loading pruning targets...")
    
    heads_to_prune = load_json(heads_path)
    neurons_to_prune = load_json(neurons_path)
    
    print(f"  Attention heads to prune: {len(heads_to_prune)}")
    print(f"  FFN neurons to prune: {len(neurons_to_prune)}")
    
    return heads_to_prune, neurons_to_prune


def _normalize_heads_to_prune(heads_to_prune_raw, num_layers: int):
    """
    허용 포맷을 모두 dict[int, list[int]]로 정규화:
      - dict: {layer_idx(int|str): [head_idx, ...]} 또는 {layer_idx: int}
      - list[dict]: [{"layer": L, "head": H}]  or {"layer": L, "heads": [H1, ...]}
      - list[list|tuple]: [[L, H], [L, H], ...]
      - list[int]: [H1, H2, ...] -> 모든 레이어에 동일 적용 (보수적)
    """
    norm = {i: [] for i in range(num_layers)}

    if isinstance(heads_to_prune_raw, dict):
        for k, v in heads_to_prune_raw.items():
            try:
                li = int(k)
            except Exception:
                continue
            if isinstance(v, (list, tuple)):
                norm[li].extend(int(h) for h in v)
            elif isinstance(v, int):
                norm[li].append(int(v))
        return norm

    if isinstance(heads_to_prune_raw, (list, tuple)):
        if len(heads_to_prune_raw) == 0:
            return norm

        first = heads_to_prune_raw[0]

        # list[dict]
        if isinstance(first, dict):
            for item in heads_to_prune_raw:
                if "layer" in item:
                    li = int(item["layer"])
                    if "head" in item:
                        norm[li].append(int(item["head"]))
                    elif "heads" in item:
                        norm[li].extend(int(h) for h in item["heads"])
            return norm

        # list[list|tuple] -> [[L, H], ...]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            for li, hi in heads_to_prune_raw:
                norm[int(li)].append(int(hi))
            return norm

        # list[int] -> 모든 레이어에 동일 적용
        if isinstance(first, int):
            for li in range(num_layers):
                norm[li].extend(int(h) for h in heads_to_prune_raw)
            return norm

    # 인식 불가 포맷 => 빈 dict
    return norm


def prune_attention_heads(model, heads_to_prune, verbose: bool = True):
    """
    Attention heads 프루닝 (4B 멀티모달 모델 지원)
    """
    # head_dim 가져오기 (멀티모달 모델 지원)
    head_dim = get_head_dim(model)
    
    # 모델 구조에 따라 layers 접근
    layers = get_model_layers(model)
    num_layers = len(layers)

    # 입력 포맷 정규화
    heads_norm = _normalize_heads_to_prune(heads_to_prune, num_layers)

    heads_by_layer = {}
    if verbose:
        print(f"[prune_attention_heads] start - layers={num_layers}, head_dim={head_dim}")

    for layer_idx in range(num_layers):
        attn = layers[layer_idx].self_attn

        q_w = attn.q_proj.weight.data
        k_w = attn.k_proj.weight.data
        v_w = attn.v_proj.weight.data
        o_w = attn.o_proj.weight.data

        # 현재 헤드 수
        orig_q_heads = q_w.shape[0] // head_dim
        orig_kv_heads = max(1, k_w.shape[0] // head_dim)
        if orig_q_heads == 0 or orig_kv_heads == 0:
            if verbose:
                print(f"  - layer {layer_idx}: skip (no heads)")
            heads_by_layer[layer_idx] = list(range(orig_q_heads))
            continue

        if verbose:
            print(f"  - layer {layer_idx}: before Q={orig_q_heads}, KV={orig_kv_heads}")

        # 제거 대상
        remove_set = set(int(h) for h in heads_norm.get(layer_idx, []))
        keep_heads = [h for h in range(orig_q_heads) if h not in remove_set]

        if len(keep_heads) == 0:
            keep_heads = [0]
            if verbose:
                print(f"      no heads left; keep head {keep_heads}")

        # 새 qh / kvh / groups 계산
        new_q = len(keep_heads)
        new_kv = orig_kv_heads
        assert new_q >= 1 and new_kv >= 1, f"[L{layer_idx}] invalid heads after prune"
        assert new_q % new_kv == 0, f"[L{layer_idx}] GQA mismatch after prune"

        # Q 인덱스 생성
        keep_q_idx = []
        for h in keep_heads:
            s = h * head_dim
            e = (h + 1) * head_dim
            keep_q_idx.extend(range(s, e))
        keep_q_idx = torch.tensor(keep_q_idx, device=q_w.device, dtype=torch.long)

        # 가중치 슬라이싱
        new_q_w = torch.index_select(q_w, 0, keep_q_idx).clone()
        attn.q_proj.weight = torch.nn.Parameter(new_q_w)
        if attn.q_proj.bias is not None:
            new_q_b = torch.index_select(attn.q_proj.bias.data, 0, keep_q_idx).clone()
            attn.q_proj.bias = torch.nn.Parameter(new_q_b)

        attn.k_proj.weight = torch.nn.Parameter(k_w.clone())
        if attn.k_proj.bias is not None:
            attn.k_proj.bias = torch.nn.Parameter(attn.k_proj.bias.data.clone())

        attn.v_proj.weight = torch.nn.Parameter(v_w.clone())
        if attn.v_proj.bias is not None:
            attn.v_proj.bias = torch.nn.Parameter(attn.v_proj.bias.data.clone())

        new_o_w = torch.index_select(o_w, 1, keep_q_idx).clone()
        attn.o_proj.weight = torch.nn.Parameter(new_o_w)

        # 메타데이터 동기화
        attn.q_proj.out_features = attn.q_proj.weight.shape[0]
        attn.k_proj.out_features = attn.k_proj.weight.shape[0]
        attn.v_proj.out_features = attn.v_proj.weight.shape[0]
        attn.q_proj.in_features = attn.q_proj.weight.shape[1]
        attn.k_proj.in_features = attn.k_proj.weight.shape[1]
        attn.v_proj.in_features = attn.v_proj.weight.shape[1]
        attn.o_proj.in_features = attn.o_proj.weight.shape[1]

        new_q_heads = attn.q_proj.weight.shape[0] // head_dim
        new_kv_heads = attn.k_proj.weight.shape[0] // head_dim
        attn.num_heads = int(new_q_heads)
        attn.num_key_value_heads = int(new_kv_heads)
        attn.num_key_value_groups = int(new_q_heads // new_kv_heads)
        attn.head_dim = head_dim

        if verbose:
            print(f"      keep Q-heads: {keep_heads}")
            print(f"      after: Q_out={attn.q_proj.weight.shape[0]} "
                  f"(= {attn.num_heads}×{head_dim}), "
                  f"KV_out={attn.k_proj.weight.shape[0]} "
                  f"(= {attn.num_key_value_heads}×{head_dim}), "
                  f"groups={attn.num_key_value_groups}")

        heads_by_layer[layer_idx] = keep_heads

    # 레이어별 남은 헤드 수 리포트
    heads_remaining = {}
    for layer_idx in range(num_layers):
        attn = layers[layer_idx].self_attn
        qh = attn.q_proj.weight.shape[0] // head_dim
        kvh = attn.k_proj.weight.shape[0] // head_dim
        heads_remaining[layer_idx] = {"num_q_heads": int(qh), "num_kv_heads": int(kvh)}

    # 최종 무결성 검사
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        o_w = attn.o_proj.weight
        assert q_w.shape[0] % head_dim == 0
        assert k_w.shape[0] % head_dim == 0
        qh = q_w.shape[0] // head_dim
        kvh = k_w.shape[0] // head_dim
        assert qh >= 1 and kvh >= 1
        assert qh % kvh == 0
        assert o_w.shape[1] == q_w.shape[0]

    return model, heads_by_layer, heads_remaining


def prune_ffn_neurons(model, neurons_to_prune, verbose: bool = True):
    """
    FFN neurons 프루닝 (4B 멀티모달 모델 지원)
    """
    if verbose:
        print("\nPruning FFN neurons...")

    raw_by_layer = {}
    for item in neurons_to_prune:
        li = int(item["layer"])
        ni = int(item["neuron"])
        raw_by_layer.setdefault(li, set()).add(ni)

    # 모델 구조에 따라 layers 접근
    layers = get_model_layers(model)
    num_layers = len(layers)
    neurons_by_layer = {}

    if verbose:
        print(f"  Pruning neurons in {len(raw_by_layer)} layers")

    for layer_idx in sorted(raw_by_layer.keys()):
        mlp = layers[layer_idx].mlp
        gate_w = mlp.gate_proj.weight.data
        up_w = mlp.up_proj.weight.data
        down_w = mlp.down_proj.weight.data

        inter = gate_w.shape[0]
        assert up_w.shape[0] == inter and down_w.shape[1] == inter, f"[L{layer_idx}] mismatched FFN shapes"

        # 원래 선택된 제거 집합
        remove_sorted = sorted(raw_by_layer[layer_idx])

        # 엣지 레이어 보호
        if (layer_idx < EDGE_LAYERS_PROTECT) or (layer_idx >= num_layers - EDGE_LAYERS_PROTECT):
            target_remove = int(round(len(remove_sorted) * EDGE_RATIO_SCALE))
            remove_sorted = remove_sorted[:target_remove]
            if verbose and len(raw_by_layer[layer_idx]) != len(remove_sorted):
                print(f"    [edge-protect] L{layer_idx}: remove {len(raw_by_layer[layer_idx])} → {len(remove_sorted)}")

        # 레이어별 최대 제거율 제한
        cap = int(inter * MAX_LAYER_PRUNE_RATIO)
        if len(remove_sorted) > cap:
            if verbose:
                print(f"    [cap] L{layer_idx}: remove {len(remove_sorted)} → {cap} (cap {MAX_LAYER_PRUNE_RATIO*100:.1f}%)")
            remove_sorted = remove_sorted[:cap]

        remove_set = set(remove_sorted)
        keep = [i for i in range(inter) if i not in remove_set]

        if len(keep) < MIN_INTERMEDIATE:
            if verbose:
                print(f"    Layer {layer_idx}: keep {len(keep)} < {MIN_INTERMEDIATE}, adjusting…")
            keep = list(range(min(inter, max(MIN_INTERMEDIATE, len(keep)))))

        if len(keep) == inter:
            neurons_by_layer[layer_idx] = []
            if verbose:
                print(f"    Layer {layer_idx}: {inter} -> {len(keep)} (no-op)")
            continue

        keep_t = torch.tensor(sorted(set(keep)), device=gate_w.device, dtype=torch.long)

        # 슬라이싱
        new_gate_w = torch.index_select(gate_w, 0, keep_t).clone()
        new_up_w = torch.index_select(up_w, 0, keep_t).clone()
        new_down_w = torch.index_select(down_w, 1, keep_t).clone()

        mlp.gate_proj.weight = torch.nn.Parameter(new_gate_w)
        mlp.up_proj.weight = torch.nn.Parameter(new_up_w)
        mlp.down_proj.weight = torch.nn.Parameter(new_down_w)

        # bias 슬라이스
        if mlp.gate_proj.bias is not None:
            mlp.gate_proj.bias = torch.nn.Parameter(
                torch.index_select(mlp.gate_proj.bias.data, 0, keep_t).clone()
            )
        if mlp.up_proj.bias is not None:
            mlp.up_proj.bias = torch.nn.Parameter(
                torch.index_select(mlp.up_proj.bias.data, 0, keep_t).clone()
            )

        # 메타데이터
        mlp.gate_proj.out_features = mlp.gate_proj.weight.shape[0]
        mlp.up_proj.out_features = mlp.up_proj.weight.shape[0]
        mlp.down_proj.in_features = mlp.down_proj.weight.shape[1]
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = mlp.gate_proj.weight.shape[0]

        assert mlp.gate_proj.weight.shape[0] == mlp.up_proj.weight.shape[0]
        assert mlp.down_proj.weight.shape[1] == mlp.gate_proj.weight.shape[0]

        neurons_by_layer[layer_idx] = sorted(list(remove_set))
        if verbose:
            print(f"    Layer {layer_idx}: {inter} -> {len(keep)} neurons")

    if verbose:
        print("  FFN neurons pruning completed")
    return model, neurons_by_layer


def update_config_for_pruned_model(model):
    """프루닝된 모델의 config를 실제 크기로 업데이트"""
    print("\nConfig 업데이트 중...")
    
    layers = get_model_layers(model)
    
    # 멀티모달 모델 지원: text_config 확인
    if hasattr(model.config, 'text_config'):
        config = model.config.text_config
    else:
        config = model.config
    
    # head_dim 계산
    head_dim = get_head_dim(model)
    
    layer_sizes = []
    for layer in layers:
        layer_sizes.append({
            'q_heads': layer.self_attn.q_proj.weight.shape[0] // head_dim,
            'intermediate_size': layer.mlp.gate_proj.weight.shape[0]
        })
    
    avg_intermediate = sum(l['intermediate_size'] for l in layer_sizes) // len(layer_sizes)
    
    # config 업데이트 (멀티모달은 text_config에)
    if hasattr(model.config, 'text_config'):
        model.config.text_config.intermediate_size = avg_intermediate
    else:
        model.config.intermediate_size = avg_intermediate
    
    print(f"  Config updated: intermediate_size = {avg_intermediate} (평균값)")


if __name__ == "__main__":
    print("=" * 60)
    print("Activation-based Pruning Integration")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Prune ratio: {PRUNE_RATIO:.0%}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 60)
    
    # [1/5] 프루닝 대상 로드
    print("\n[1/5] Loading pruning targets...")
    heads_to_prune, neurons_to_prune = load_pruning_targets(HEADS_PATH, NEURONS_PATH)
    
    # [2/5] 모델 로드
    print(f"\n[2/5] Loading model from {MODEL_PATH}...")
    print("  Using float32 for numerical stability")
    print("  Using eager attention implementation")
    
    model, tokenizer = load_model(MODEL_PATH, dtype=torch.float32)
    
    # 모델 타입 확인
    if hasattr(model, 'language_model'):
        print("  Detected multimodal model (Gemma3ForConditionalGeneration)")
    else:
        print("  Detected text-only model (Gemma3ForCausalLM)")
    
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # [3/5] Attention heads 프루닝 
    print("\n[3/5] Pruning attention heads...")
    model, heads_by_layer, heads_remaining = prune_attention_heads(model, heads_to_prune)
    
    # [4/5] FFN neurons 프루닝
    print("\n[4/5] Pruning FFN neurons...")
    model, neurons_by_layer = prune_ffn_neurons(model, neurons_to_prune)
    
    # Config 업데이트 (중요!)
    update_config_for_pruned_model(model)
    
    # [5/5] 모델 저장
    print("\n[5/5] Saving pruned model...")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    save_pruned_model(
        model, 
        tokenizer, 
        OUTPUT_PATH,
        MODEL_PATH, 
        pruning_metadata={
            'method': 'activation',
            'prune_ratio': PRUNE_RATIO,
            'heads': heads_by_layer,
            'neurons': neurons_by_layer,
            'heads_remaining': heads_remaining,
        }
    )
    
    print("\n" + "=" * 60)
    print("Activation-based Pruning Integration completed!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PATH}/")
    print(f"  - model.safetensors (pruned model)")
    print(f"  - config.json (model config)")
    print(f"  - pruning_metadata.json (pruning info)")
    
    # 프루닝 직후 간단한 테스트
    print("\n" + "=" * 60)
    print("Quick validation test")
    print("=" * 60)
    
    try:
        model.eval()
        
        test_prompts = [
            "Hello, I am",
            "The capital of France is"
        ]
        
        print("\nGenerating test outputs...")
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  '{prompt}' -> '{generated_text}'")
        
        print("\nValidation test completed successfully!")
        
    except Exception as e:
        print(f"\nValidation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("\nNext step: Run evaluation script to measure performance")
    print("=" * 60)