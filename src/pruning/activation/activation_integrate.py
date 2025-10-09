import torch
import json
import os
from pathlib import Path
import sys

# 프로젝트 루트 경로 
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import load_json, save_json
from src.utils.model_loader import load_model, save_pruned_model


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
    IN : (model, heads_to_prune)  # heads_to_prune는 다양한 포맷 허용
    OUT: (model, heads_by_layer, heads_remaining)

    변경점(중요):
    - '원래 GQA 그룹 크기(g)'를 유지하려는 제약을 제거.
    - 입력으로 받은 '개별 head 인덱스' 기준으로 Q만 슬라이스한다.
    - KV는 kvh≥1이면 그대로 유지(여기서는 1). 새 qh에 맞춰 num_key_value_groups를 재산출한다.
    - 불변식: qh >= 1, kvh >= 1, qh % kvh == 0, o_in == q_out
    """
    assert hasattr(model, "config") and hasattr(model.config, "head_dim"), "model.config.head_dim 필요"
    head_dim = model.config.head_dim
    num_layers = len(model.model.layers)

    # 1) 입력 포맷 정규화(딕셔너리: layer -> [head_idx,...])
    heads_norm = _normalize_heads_to_prune(heads_to_prune, num_layers)

    heads_by_layer = {}
    if verbose:
        print(f"[prune_attention_heads] start - layers={num_layers}, head_dim={head_dim}")

    for layer_idx in range(num_layers):
        attn = model.model.layers[layer_idx].self_attn

        q_w = attn.q_proj.weight.data    # [Q_out, hidden]
        k_w = attn.k_proj.weight.data    # [K_out, hidden]
        v_w = attn.v_proj.weight.data    # [V_out, hidden]
        o_w = attn.o_proj.weight.data    # [hidden, Q_out]

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

        # 제거 대상(head 인덱스)
        remove_set = set(int(h) for h in heads_norm.get(layer_idx, []))
        # 남길 Q-head 인덱스(그냥 개별 헤드 단위로)
        keep_heads = [h for h in range(orig_q_heads) if h not in remove_set]

        # 최소 1개는 남겨야 함
        if len(keep_heads) == 0:
            keep_heads = [0]  
            if verbose:
                print(f"      no heads left; keep head {keep_heads}")

        # 새 qh / kvh / groups 계산
        new_q = len(keep_heads)
        new_kv = orig_kv_heads  # KV는 그대로 유지 (여기서는 1)
        assert new_q >= 1 and new_kv >= 1, f"[L{layer_idx}] invalid heads after prune (qh={new_q}, kvh={new_kv})"
        assert new_q % new_kv == 0, f"[L{layer_idx}] GQA mismatch after prune (qh={new_q}, kvh={new_kv})"

        # --- 인덱스 생성 ---
        # Q (행 방향 슬라이스)
        keep_q_idx = []
        for h in keep_heads:
            s = h * head_dim
            e = (h + 1) * head_dim
            keep_q_idx.extend(range(s, e))
        keep_q_idx = torch.tensor(keep_q_idx, device=q_w.device, dtype=torch.long)

        # K/V (KV는 유지) → 인덱스 전체
        keep_k_idx = torch.arange(k_w.shape[0], device=k_w.device, dtype=torch.long)
        # 필요시 KV도 줄이고 싶다면 여기서 선택적으로 슬라이스하되 new_q % new_kv == 0 유지해야 함.

        # --- 가중치/바이어스 슬라이싱 ---
        # Q
        new_q_w = torch.index_select(q_w, 0, keep_q_idx).clone()
        attn.q_proj.weight = torch.nn.Parameter(new_q_w)
        if attn.q_proj.bias is not None:
            new_q_b = torch.index_select(attn.q_proj.bias.data, 0, keep_q_idx).clone()
            attn.q_proj.bias = torch.nn.Parameter(new_q_b)

        # K (그대로)
        attn.k_proj.weight = torch.nn.Parameter(k_w.clone())
        if attn.k_proj.bias is not None:
            attn.k_proj.bias = torch.nn.Parameter(attn.k_proj.bias.data.clone())

        # V (그대로)
        attn.v_proj.weight = torch.nn.Parameter(v_w.clone())
        if attn.v_proj.bias is not None:
            attn.v_proj.bias = torch.nn.Parameter(attn.v_proj.bias.data.clone())

        # O (열 방향 = Q concat 축)
        new_o_w = torch.index_select(o_w, 1, keep_q_idx).clone()
        attn.o_proj.weight = torch.nn.Parameter(new_o_w)

        # --- 모듈 메타데이터 동기화 ---
        attn.q_proj.out_features = attn.q_proj.weight.shape[0]
        attn.k_proj.out_features = attn.k_proj.weight.shape[0]
        attn.v_proj.out_features = attn.v_proj.weight.shape[0]
        attn.q_proj.in_features  = attn.q_proj.weight.shape[1]
        attn.k_proj.in_features  = attn.k_proj.weight.shape[1]
        attn.v_proj.in_features  = attn.v_proj.weight.shape[1]
        attn.o_proj.in_features  = attn.o_proj.weight.shape[1]

        # 새 헤드 수 세팅 (hidden_size/head_dim은 불변)
        new_q_heads = attn.q_proj.weight.shape[0] // head_dim
        new_kv_heads = attn.k_proj.weight.shape[0] // head_dim  
        attn.num_heads = int(new_q_heads)
        attn.num_key_value_heads = int(new_kv_heads)
        attn.num_key_value_groups = int(new_q_heads // new_kv_heads)
        attn.head_dim = head_dim

        # 불변식 확인
        assert new_q_heads >= 1 and new_kv_heads >= 1
        assert new_q_heads % new_kv_heads == 0
        assert attn.o_proj.weight.shape[1] == attn.q_proj.weight.shape[0]

        if verbose:
            print(f"      keep Q-heads: {keep_heads}")
            print(f"      after: Q_out={attn.q_proj.weight.shape[0]} "
                  f"(= {attn.num_heads}×{head_dim}), "
                  f"KV_out={attn.k_proj.weight.shape[0]} "
                  f"(= {attn.num_key_value_heads}×{head_dim}), "
                  f"groups={attn.num_key_value_groups}")

        # 리포트(남은 Q-head 인덱스)
        heads_by_layer[layer_idx] = keep_heads

    # 레이어별 남은 헤드 수 리포트
    heads_remaining = {}
    for layer_idx in range(num_layers):
        attn = model.model.layers[layer_idx].self_attn
        qh = attn.q_proj.weight.shape[0] // head_dim
        kvh = attn.k_proj.weight.shape[0] // head_dim
        heads_remaining[layer_idx] = {"num_q_heads": int(qh), "num_kv_heads": int(kvh)}

    # 최종 무결성
    for i, layer in enumerate(model.model.layers):
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


def prune_ffn_neurons(
    model,
    neurons_to_prune,
    *,
    min_intermediate: int = 400,
    edge_layers_protect: int = 2,    # 앞/뒤 N개 레이어 보호
    edge_ratio_scale: float = 0.5,   # 보호 레이어는 제거비율 절반
    max_layer_prune_ratio: float = 0.45,  # 레이어별 최대 제거율 3%
    verbose: bool = True
):
    if verbose:
        print("\n Pruning FFN neurons...")

    raw_by_layer = {}
    for item in neurons_to_prune:
        li = int(item["layer"]); ni = int(item["neuron"])
        raw_by_layer.setdefault(li, set()).add(ni)

    num_layers = len(model.model.layers)
    neurons_by_layer = {}

    if verbose:
        print(f"  Pruning neurons in {len(raw_by_layer)} layers")

    for layer_idx in sorted(raw_by_layer.keys()):
        mlp = model.model.layers[layer_idx].mlp
        gate_w = mlp.gate_proj.weight.data
        up_w   = mlp.up_proj.weight.data
        down_w = mlp.down_proj.weight.data

        inter = gate_w.shape[0]
        assert up_w.shape[0] == inter and down_w.shape[1] == inter, f"[L{layer_idx}] mismatched FFN shapes"

        # 원래 선택된 제거 집합
        remove_sorted = sorted(raw_by_layer[layer_idx])

        # 엣지 레이어 보호
        if (layer_idx < edge_layers_protect) or (layer_idx >= num_layers - edge_layers_protect):
            target_remove = int(round(len(remove_sorted) * edge_ratio_scale))
            remove_sorted = remove_sorted[:target_remove]
            if verbose and len(raw_by_layer[layer_idx]) != len(remove_sorted):
                print(f"    [edge-protect] L{layer_idx}: remove {len(raw_by_layer[layer_idx])} → {len(remove_sorted)}")

        # 레이어별 최대 제거율 제한
        cap = int(inter * max_layer_prune_ratio)
        if len(remove_sorted) > cap:
            if verbose:
                print(f"    [cap] L{layer_idx}: remove {len(remove_sorted)} → {cap} (cap {max_layer_prune_ratio*100:.1f}%)")
            remove_sorted = remove_sorted[:cap]

        remove_set = set(remove_sorted)
        keep = [i for i in range(inter) if i not in remove_set]

        if len(keep) < min_intermediate:
            if verbose:
                print(f"    Layer {layer_idx}: keep {len(keep)} < {min_intermediate}, adjusting…")
            keep = list(range(min(inter, max(min_intermediate, len(keep)))))

        if len(keep) == inter:
            neurons_by_layer[layer_idx] = []
            if verbose:
                print(f"    Layer {layer_idx}: {inter} -> {len(keep)} (no-op)")
            continue

        keep_t = torch.tensor(sorted(set(keep)), device=gate_w.device, dtype=torch.long)

        # 슬라이싱
        new_gate_w = torch.index_select(gate_w, 0, keep_t).clone()
        new_up_w   = torch.index_select(up_w,   0, keep_t).clone()
        new_down_w = torch.index_select(down_w, 1, keep_t).clone()

        # 스케일 보정 없음 (과상승 방지)
        mlp.gate_proj.weight = torch.nn.Parameter(new_gate_w)
        mlp.up_proj.weight   = torch.nn.Parameter(new_up_w)
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

        # 메타
        mlp.gate_proj.out_features = mlp.gate_proj.weight.shape[0]
        mlp.up_proj.out_features   = mlp.up_proj.weight.shape[0]
        mlp.down_proj.in_features  = mlp.down_proj.weight.shape[1]
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
    """
    프루닝된 모델의 config를 실제 크기로 업데이트
    
    주의: 레이어마다 크기가 다르므로, config의 기본값은 의미가 없어짐
    하지만 model.save_pretrained()가 config를 저장하므로 업데이트 필요
    """
    print("\n Config 업데이트 중...")
    
    # 각 레이어의 실제 크기 계산
    layer_sizes = []
    for layer in model.model.layers:
        layer_sizes.append({
            'q_heads': layer.self_attn.q_proj.weight.shape[0] // (model.config.hidden_size // model.config.num_attention_heads),
            'intermediate_size': layer.mlp.gate_proj.weight.shape[0]
        })
    
    # config는 단일 값만 가질 수 있으므로, 평균값이나 최소값으로 설정
    avg_intermediate = sum(l['intermediate_size'] for l in layer_sizes) // len(layer_sizes)
    
    model.config.intermediate_size = avg_intermediate
    
    print(f"  Config updated: intermediate_size = {avg_intermediate} (평균값)")
    print(f"  ℹ실제 로드 시에는 pruned_structure.json의 레이어별 크기 사용")

def _check_attention_invariants(model, verbose: bool = True):
    """프루닝 후 GQA/차원 무결성 빠른 검사."""
    hd = model.config.head_dim
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        o_w = attn.o_proj.weight

        assert q_w.shape[0] % hd == 0, f"[L{i}] q_out % head_dim != 0"
        assert k_w.shape[0] % hd == 0, f"[L{i}] k_out % head_dim != 0"

        qh = q_w.shape[0] // hd
        kvh = k_w.shape[0] // hd
        assert qh >= 1 and kvh >= 1, f"[L{i}] invalid heads qh={qh}, kvh={kvh}"
        assert qh % kvh == 0, f"[L{i}] GQA mismatch: qh={qh}, kvh={kvh} (qh % kvh != 0)"
        assert o_w.shape[1] == q_w.shape[0], f"[L{i}] o_in({o_w.shape[1]}) != q_out({q_w.shape[0]})"

        if verbose and i == 0:
            print(f"[invariants] pass (showing first layer): qh={qh}, kvh={kvh}, head_dim={hd}")

if __name__ == "__main__":
    print("=" * 60)
    print("Activation-based Pruning Integration")
    print("=" * 60)
    
    # 경로 설정
    model_path = "models/original/gemma-3-1b-it"
    heads_path = "results/activation_heads_to_prune_it.json"
    neurons_path = "results/activation_neurons_to_prune_it.json"
    output_path = "models/pruned_activation_it"
    
    # [1/5] 프루닝 대상 로드
    print("\n[1/5] Loading pruning targets...")
    heads_to_prune, neurons_to_prune = load_pruning_targets(heads_path, neurons_path)
    
    # [2/5] 모델 로드
    print(f"\n[2/5] Loading model from {model_path}...")
    print("  Using float32 for numerical stability")
    print("  Using eager attention implementation")
    
    model, tokenizer = load_model(model_path, dtype=torch.float32)
    
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # [3/5] Attention heads 프루닝 
    print("\n[3/5] Pruning attention heads...")
    model, heads_by_layer, heads_remaining = prune_attention_heads(model, heads_to_prune)
    
    # [4/5] FFN neurons 프루닝
    print("\n[4/5] Pruning FFN neurons...")
    model, neurons_by_layer = prune_ffn_neurons(model, neurons_to_prune)
    
    # [5/5] 모델 저장
    print("\n[5/5] Saving pruned model...")
    
    save_pruned_model(
        model, 
        tokenizer, 
        output_path,
        model_path, 
        pruning_metadata={
            'method': 'activation',
            'heads': heads_by_layer,
            'neurons': neurons_by_layer,
            'heads_remaining': heads_remaining,
        }
    )
    
    print("\n" + "=" * 60)
    print("Activation-based Pruning Integration completed!")
    print("=" * 60)
    print(f"\nOutput: {output_path}/")
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