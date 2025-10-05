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


def prune_attention_heads(model, heads_to_prune):
    """Attention heads 프루닝"""
    print("\nPruning attention heads...")
    
    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)
    
    # config의 head_dim 사용 (고정값)
    head_dim = config.head_dim
    
    print(f"  Config: Q heads={num_q_heads}, KV heads={num_kv_heads}, head_dim={head_dim}")
    
    heads_by_layer = {}
    for item in heads_to_prune:
        layer = item['layer']
        head = item['head']
        if layer not in heads_by_layer:
            heads_by_layer[layer] = []
        heads_by_layer[layer].append(head)
    
    print(f"  Pruning heads in {len(heads_by_layer)} layers")
    
    heads_remaining = {}
    
    for layer_idx, heads in heads_by_layer.items():
        print(f"    Layer {layer_idx}: pruning {len(heads)} heads")
        
        attention = model.model.layers[layer_idx].self_attn
        
        q_weight = attention.q_proj.weight.data
        k_weight = attention.k_proj.weight.data
        v_weight = attention.v_proj.weight.data
        
        # config의 head_dim 사용
        actual_q_head_dim = head_dim
        actual_kv_head_dim = head_dim
        
        print(f"      Before: Q {q_weight.shape[0]}({num_q_heads}×{head_dim}), K/V {k_weight.shape[0]}({num_kv_heads}×{head_dim})")
        
        # 남길 헤드 인덱스
        keep_heads = [h for h in range(num_q_heads) if h not in heads]
        
        if len(keep_heads) == 0:
            print(f"      ⚠️ Warning: Keeping at least 1 head")
            keep_heads = [0]
        
        # GQA 매핑
        if num_kv_heads < num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            needed_kv_heads = set()
            for q_head in keep_heads:
                kv_head = q_head // repeat_factor
                needed_kv_heads.add(kv_head)
            keep_kv_heads = sorted(list(needed_kv_heads))
            
            print(f"      Keep Q heads: {keep_heads} (total: {len(keep_heads)})")
            print(f"      Need KV heads: {keep_kv_heads} (total: {len(keep_kv_heads)})")
        else:
            keep_kv_heads = keep_heads
        
        new_num_q_heads = len(keep_heads)
        new_num_kv_heads = len(keep_kv_heads)
        
        heads_remaining[layer_idx] = {
            'num_q_heads': new_num_q_heads,
            'num_kv_heads': new_num_kv_heads
        }
        
        # Q projection 프루닝
        keep_q_indices = []
        for h in keep_heads:
            start = h * actual_q_head_dim
            end = (h + 1) * actual_q_head_dim
            keep_q_indices.extend(range(start, end))
        
        keep_q_indices = torch.tensor(keep_q_indices, device=q_weight.device, dtype=torch.long)
        q_bias = attention.q_proj.bias.data if attention.q_proj.bias is not None else None
        
        attention.q_proj.weight = torch.nn.Parameter(
            torch.index_select(q_weight, 0, keep_q_indices).clone()
        )
        if q_bias is not None:
            attention.q_proj.bias = torch.nn.Parameter(
                torch.index_select(q_bias, 0, keep_q_indices).clone()
            )
        
        # K projection 프루닝
        keep_k_indices = []
        for h in keep_kv_heads:
            start = h * actual_kv_head_dim
            end = (h + 1) * actual_kv_head_dim
            keep_k_indices.extend(range(start, end))
        
        keep_k_indices = torch.tensor(keep_k_indices, device=k_weight.device, dtype=torch.long)
        k_bias = attention.k_proj.bias.data if attention.k_proj.bias is not None else None
        
        attention.k_proj.weight = torch.nn.Parameter(
            torch.index_select(k_weight, 0, keep_k_indices).clone()
        )
        if k_bias is not None:
            attention.k_proj.bias = torch.nn.Parameter(
                torch.index_select(k_bias, 0, keep_k_indices).clone()
            )
        
        # V projection 프루닝
        v_bias = attention.v_proj.bias.data if attention.v_proj.bias is not None else None
        
        attention.v_proj.weight = torch.nn.Parameter(
            torch.index_select(v_weight, 0, keep_k_indices).clone()
        )
        if v_bias is not None:
            attention.v_proj.bias = torch.nn.Parameter(
                torch.index_select(v_bias, 0, keep_k_indices).clone()
            )
        
        # O projection
        o_weight = attention.o_proj.weight.data
        o_bias = attention.o_proj.bias.data if attention.o_proj.bias is not None else None
        
        attention.o_proj.weight = torch.nn.Parameter(
            torch.index_select(o_weight, 1, keep_q_indices).clone()
        )
        if o_bias is not None:
            attention.o_proj.bias = torch.nn.Parameter(o_bias.clone())
        
        # Attention 모듈 속성 업데이트
        new_q_dim = len(keep_q_indices)
        new_kv_dim = len(keep_k_indices)
        
        # 유일한 변경: head_dim은 항상 256으로 고정
        attention.num_heads = new_num_q_heads
        attention.num_key_value_heads = new_num_kv_heads
        attention.num_key_value_groups = new_num_q_heads // new_num_kv_heads
        attention.head_dim = head_dim  # config 값 사용 (256)
        attention.hidden_size = new_q_dim

        # Linear 레이어의 속성도 명시적으로 업데이트
        attention.q_proj.out_features = new_q_dim
        attention.k_proj.out_features = new_kv_dim
        attention.v_proj.out_features = new_kv_dim
        attention.o_proj.in_features = new_q_dim

        # 디버깅용 출력 추가
        print(f"      속성 업데이트 완료:")
        print(f"         num_heads: {attention.num_heads}")
        print(f"         num_key_value_heads: {attention.num_key_value_heads}")
        print(f"         q_proj.out_features: {attention.q_proj.out_features}")
        
        print(f"      After: Q {new_q_dim}({new_num_q_heads}×{head_dim}), K/V {new_kv_dim}({new_num_kv_heads}×{head_dim})")
        print(f"      New num_key_value_groups: {attention.num_key_value_groups}")
    
    # 프루닝되지 않은 레이어도 원본 헤드 수 저장
    for layer_idx in range(len(model.model.layers)):
        if layer_idx not in heads_remaining:
            heads_remaining[layer_idx] = {
                'num_q_heads': num_q_heads,
                'num_kv_heads': num_kv_heads
            }
    
    print("  Attention heads pruning completed")
    return model, heads_by_layer, heads_remaining

def prune_ffn_neurons(model, neurons_to_prune):
    """FFN neurons 프루닝"""
    print("\nPruning FFN neurons...")
    
    # 레이어별로 제거할 뉴런 그룹화
    neurons_by_layer = {}
    for item in neurons_to_prune:
        layer = item['layer']
        neuron = item['neuron']
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append(neuron)
    
    print(f"  Pruning neurons in {len(neurons_by_layer)} layers")
    
    for layer_idx, neurons in neurons_by_layer.items():
        mlp = model.model.layers[layer_idx].mlp
        
        # 현재 뉴런 수
        intermediate_size = mlp.gate_proj.weight.data.shape[0]
        
        # 남길 뉴런 인덱스
        keep_neurons = [n for n in range(intermediate_size) if n not in neurons]
        
        if len(keep_neurons) == 0:
            print(f"    ⚠️ Warning: Layer {layer_idx} keeping at least 1 neuron")
            keep_neurons = [0]
        
        print(f"    Layer {layer_idx}: {intermediate_size} -> {len(keep_neurons)} neurons")
        
        keep_neurons_tensor = torch.tensor(keep_neurons, device=mlp.gate_proj.weight.device, dtype=torch.long)
        
        # gate_proj
        gate_weight = mlp.gate_proj.weight.data
        gate_bias = mlp.gate_proj.bias.data if mlp.gate_proj.bias is not None else None
        
        mlp.gate_proj.weight = torch.nn.Parameter(
            torch.index_select(gate_weight, 0, keep_neurons_tensor).clone()
        )
        mlp.gate_proj.out_features = len(keep_neurons)

        if gate_bias is not None:
            mlp.gate_proj.bias = torch.nn.Parameter(
                torch.index_select(gate_bias, 0, keep_neurons_tensor).clone()
            )
        
        # up_proj
        up_weight = mlp.up_proj.weight.data
        up_bias = mlp.up_proj.bias.data if mlp.up_proj.bias is not None else None
        
        mlp.up_proj.weight = torch.nn.Parameter(
            torch.index_select(up_weight, 0, keep_neurons_tensor).clone()
        )
        mlp.up_proj.out_features = len(keep_neurons)
        if up_bias is not None:
            mlp.up_proj.bias = torch.nn.Parameter(
                torch.index_select(up_bias, 0, keep_neurons_tensor).clone()
            )
        
        # down_proj
        down_weight = mlp.down_proj.weight.data
        down_bias = mlp.down_proj.bias.data if mlp.down_proj.bias is not None else None
        
        mlp.down_proj.weight = torch.nn.Parameter(
            torch.index_select(down_weight, 1, keep_neurons_tensor).clone()
        )
        mlp.down_proj.in_features = len(keep_neurons)
        if down_bias is not None:
            mlp.down_proj.bias = torch.nn.Parameter(down_bias.clone())

        if hasattr(mlp, 'intermediate_size'):
          mlp.intermediate_size = len(keep_neurons)
    
    print("  FFN neurons pruning completed")
    return model, neurons_by_layer

def update_config_for_pruned_model(model):
    """
    프루닝된 모델의 config를 실제 크기로 업데이트
    
    주의: 레이어마다 크기가 다르므로, config의 기본값은 의미가 없어짐
    하지만 model.save_pretrained()가 config를 저장하므로 업데이트 필요
    """
    print("\nConfig 업데이트 중...")
    
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
    print(f"  실제 로드 시에는 pruned_structure.json의 레이어별 크기 사용")

if __name__ == "__main__":
    print("=" * 60)
    print("Magnitude-based Pruning Integration")
    print("=" * 60)
    
    # 경로 설정
    model_path = "models/original/gemma-3-1b-pt"
    heads_path = "results/magnitude_heads_to_prune.json"
    neurons_path = "results/magnitude_neurons_to_prune.json"
    output_path = "models/pruned_magnitude"
    
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
            'method': 'magnitude',
            'heads': heads_by_layer,
            'neurons': neurons_by_layer,
            'heads_remaining': heads_remaining,
        }
    )
    
    print("\n" + "=" * 60)
    print("Magnitude-based Pruning Integration completed!")
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