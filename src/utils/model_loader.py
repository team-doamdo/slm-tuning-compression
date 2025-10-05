import torch
import os
import json
import safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Dict, Any


def load_model(
    model_path: str,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    원본 모델 로드 (HuggingFace 표준)
    
    Args:
        model_path: 모델 경로
        device: 'cuda' 또는 'cpu'
        dtype: torch.float32, torch.float16 등 (None이면 float32)
        trust_remote_code: 커스텀 코드 신뢰 여부
    
    Returns:
        (model, tokenizer)
    """
    print(f"모델 로드 중: {model_path}")
    
    # dtype 강제 float32 설정 (수치 안정성)
    if dtype is None:
        dtype = torch.float32  
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("⚠️ pad_token이 없어서 eos_token으로 설정")
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == 'cuda' else None,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager"  # Gemma-3 필수
        )
        
        # 강제 float32 변환 (저장된 모델이 float16이어도)
        if model.dtype != torch.float32:
            print(f"  Converting model from {model.dtype} to float32...")
            model = model.float()
        
        if device == 'cpu':
            model = model.to('cpu')
        
        print(f"모델 로드 완료")
        print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   dtype: {model.dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise


def save_pruned_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str,
    original_model_path: str,
    pruning_metadata: Dict[str, Any]
) -> None:
    """
    프루닝된 모델 저장 - tied weights 문제 해결
    
    Args:
        model: 프루닝된 모델
        tokenizer: 토크나이저
        output_path: 저장 경로
        original_model_path: 원본 모델 경로
        pruning_metadata: 프루닝 정보
    """
    print(f"\n 프루닝 모델 저장: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # heads_remaining 정보 추출
    heads_remaining = pruning_metadata.get('heads_remaining', {})
    
    # 1. 각 레이어의 실제 크기 저장
    layer_structure = {}
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        # 실제 크기 계산
        head_dim = model.config.head_dim  # 256
        q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
        kv_heads = layer.self_attn.k_proj.weight.shape[0] // head_dim
        
        layer_structure[layer_idx] = {
            'q_dim': layer.self_attn.q_proj.weight.shape[0],
            'k_dim': layer.self_attn.k_proj.weight.shape[0],
            'v_dim': layer.self_attn.v_proj.weight.shape[0],
            'o_input_dim': layer.self_attn.o_proj.weight.shape[1],
            'intermediate_size': layer.mlp.gate_proj.weight.shape[0],
            'num_q_heads': q_heads,
            'num_kv_heads': kv_heads,
            # 동적 속성이 있으면 사용, 없으면 계산값 사용
            'actual_num_heads': getattr(layer.self_attn, 'num_heads', q_heads),
            'actual_num_kv_heads': getattr(layer.self_attn, 'num_key_value_heads', kv_heads),
            'actual_num_kv_groups': getattr(layer.self_attn, 'num_key_value_groups', q_heads // kv_heads),
            'actual_q_out_features': getattr(layer.self_attn.q_proj, 'out_features', layer.self_attn.q_proj.weight.shape[0]),
            'actual_k_out_features': getattr(layer.self_attn.k_proj, 'out_features', layer.self_attn.k_proj.weight.shape[0]),
        }
    
    # 2. Tied weights 처리를 위한 save_pretrained 사용
    # 하지만 config는 수정하지 않음 (원본 유지)
    print(" 모델 저장 중...")
    
    # 임시로 tie_word_embeddings 설정 확인
    original_tie = model.config.tie_word_embeddings
    
    try:
        # HuggingFace의 save_pretrained 사용 (tied weights 자동 처리)
        model.save_pretrained(
            output_path, 
            safe_serialization=True,
            max_shard_size="10GB"  # 단일 파일로 저장
        )
        print(f"  model.safetensors 저장 완료")
        
    except Exception as e:
        print(f"  ⚠️ save_pretrained 실패, 대체 방법 사용: {e}")
        
        # 대체 방법: tied weights 해제 후 저장
        state_dict = model.state_dict()
        
        # lm_head.weight이 embed_tokens.weight와 같은 경우 복사본 생성
        if 'lm_head.weight' in state_dict and 'model.embed_tokens.weight' in state_dict:
            if state_dict['lm_head.weight'].data_ptr() == state_dict['model.embed_tokens.weight'].data_ptr():
                # 복사본 생성하여 메모리 공유 해제
                state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()
                print("  Tied weights 분리 완료")
        
        # safetensors로 저장
        safetensors_path = os.path.join(output_path, 'model.safetensors')
        safetensors.torch.save_file(state_dict, safetensors_path)
        print(f"  model.safetensors 저장 완료 (대체 방법)")
        
        # config도 저장
        model.config.save_pretrained(output_path)
    
    # 3. 토크나이저 저장
    tokenizer.save_pretrained(output_path)
    print(f"  토크나이저 저장 완료")
    
    # 4. 원본 모델의 필수 파일들 복사 (토크나이저 관련)
    print(f" 원본 모델의 토크나이저 파일 복사: {original_model_path}")
    
    import shutil
    
    files_to_copy = [
        'tokenizer.json',
        'tokenizer_config.json', 
        'special_tokens_map.json',
        'tokenizer.model',
    ]
    
    for filename in files_to_copy:
        src = os.path.join(original_model_path, filename)
        dst = os.path.join(output_path, filename)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"   {filename} 복사 완료")
    
    # 5. 프루닝 구조 정보 저장
    structure_info = {
        'is_pruned': True,
        'original_model_path': original_model_path,
        'layer_structure': layer_structure,
        'pruning_metadata': pruning_metadata,
        'base_config': {
            'num_hidden_layers': model.config.num_hidden_layers,
            'hidden_size': model.config.hidden_size,
            'num_attention_heads': model.config.num_attention_heads,
            'num_key_value_heads': getattr(model.config, 'num_key_value_heads', 
                                          model.config.num_attention_heads),
            'intermediate_size': model.config.intermediate_size,
            'head_dim': model.config.head_dim,
        }
    }
    
    structure_path = os.path.join(output_path, 'pruned_structure.json')
    with open(structure_path, 'w') as f:
        json.dump(structure_info, f, indent=2)
    print(f"  pruned_structure.json 저장 완료")
    
    # 6. 프루닝 메타데이터 요약 저장
    metadata_summary = {
        'pruning_method': pruning_metadata.get('method', 'magnitude'),
        'total_heads_pruned': sum(len(v) for v in pruning_metadata.get('heads', {}).values()) if pruning_metadata.get('heads') else 0,
        'total_neurons_pruned': sum(len(v) for v in pruning_metadata.get('neurons', {}).values()) if pruning_metadata.get('neurons') else 0,
        'layers_affected': len(pruning_metadata.get('heads', {})) + len(pruning_metadata.get('neurons', {})),
        'model_size': sum(p.numel() for p in model.parameters()),
    }
    
    metadata_path = os.path.join(output_path, 'pruning_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_summary, f, indent=2)
    print(f"   pruning_metadata.json 저장 완료")
    
    print(f"\n 저장 완료")
    print(f"   model.safetensors: 프루닝된 가중치")
    print(f"   pruned_structure.json: 레이어별 구조 정보")
    print(f"   pruning_metadata.json: 프루닝 요약")
    print(f"   토크나이저 파일들: 원본에서 복사")
    print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")


def load_pruned_model(
    model_path: str,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    프루닝 모델 로드 - 완전 수정 버전
    
    주요 수정사항:
    1. 가중치 중복 적용 제거
    2. 레이어 재구성과 가중치 로드 분리
    3. shape 검증 강화
    4. 속성 동기화 보장
    """
    if dtype is None:
        dtype = torch.float32
    
    # 1. pruned_structure.json 확인
    structure_path = os.path.join(model_path, 'pruned_structure.json')
    
    if not os.path.exists(structure_path):
        print(" 원본 모델로 감지됨")
        return load_model(model_path, device=device, dtype=dtype)
    
    print(f" 프루닝 모델 로드 중: {model_path}")
    
    try:
        # 2. 구조 정보 로드
        with open(structure_path, 'r') as f:
            structure_info = json.load(f)
        
        layer_structure = structure_info['layer_structure']
        base_config = structure_info['base_config']
        pruning_metadata = structure_info.get('pruning_metadata', {})
        
        print(f" 프루닝 구조 정보 로드 완료")
        print(f"   레이어 수: {base_config['num_hidden_layers']}")
        print(f"   프루닝 방법: {pruning_metadata.get('method', 'unknown')}")
        
        # 3. 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   pad_token을 eos_token으로 설정")
        
        # 4. config 로드
        config = AutoConfig.from_pretrained(model_path)
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        
        print(f"   hidden_size: {hidden_size}, head_dim: {head_dim}")
        
        # 5. 빈 모델 생성 (원본 구조로 시작)
        print("\n 기본 모델 구조 생성 중...")
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype,
            attn_implementation="eager"
        )
        
        # 6. 저장된 가중치 로드
        safetensor_path = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(safetensor_path):
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")
        
        print(" 저장된 가중치 파일 로드 중...")
        state_dict = safetensors.torch.load_file(safetensor_path)
        print(f"   로드된 키 개수: {len(state_dict)}")
        
        # 7. 각 레이어를 프루닝된 구조로 재구성
        print("\n레이어별 구조 재구성 중...")
        
        for layer_idx in range(base_config['num_hidden_layers']):
            layer = model.model.layers[layer_idx]
            layer_info = layer_structure[str(layer_idx)]
            
            # 디버깅용: 첫 몇 개 레이어 정보 출력
            if layer_idx < 3:
                print(f"\n  Layer {layer_idx} 재구성:")
                print(f"    목표 크기 - Q: {layer_info['q_dim']}, K: {layer_info['k_dim']}, "
                      f"V: {layer_info['v_dim']}, FFN: {layer_info['intermediate_size']}")
            
            # ========== Attention 모듈 재구성 ==========
            # Q projection
            q_dim = layer_info['q_dim']
            new_q_proj = torch.nn.Linear(hidden_size, q_dim, bias=False)
            layer.self_attn.q_proj = new_q_proj
            
            # K projection  
            k_dim = layer_info['k_dim']
            new_k_proj = torch.nn.Linear(hidden_size, k_dim, bias=False)
            layer.self_attn.k_proj = new_k_proj
            
            # V projection
            v_dim = layer_info['v_dim']
            new_v_proj = torch.nn.Linear(hidden_size, v_dim, bias=False)
            layer.self_attn.v_proj = new_v_proj
            
            # O projection (input dimension이 q_dim과 같아야 함)
            o_input_dim = layer_info['o_input_dim']
            new_o_proj = torch.nn.Linear(o_input_dim, hidden_size, bias=False)
            layer.self_attn.o_proj = new_o_proj
            
            # Attention 메타데이터 업데이트
            num_q_heads = q_dim // head_dim
            num_kv_heads = k_dim // head_dim
            
            layer.self_attn.num_heads = num_q_heads
            layer.self_attn.num_key_value_heads = num_kv_heads
            layer.self_attn.num_key_value_groups = num_q_heads // num_kv_heads if num_kv_heads > 0 else 1
            layer.self_attn.head_dim = head_dim
            
            # ========== FFN 모듈 재구성 ==========
            intermediate_size = layer_info['intermediate_size']
            
            # Gate projection
            new_gate = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            layer.mlp.gate_proj = new_gate
            
            # Up projection
            new_up = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            layer.mlp.up_proj = new_up
            
            # Down projection
            new_down = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
            layer.mlp.down_proj = new_down
            
            # FFN 메타데이터
            if hasattr(layer.mlp, 'intermediate_size'):
                layer.mlp.intermediate_size = intermediate_size
            
            if layer_idx < 3:
                print(f"    재구성 완료 - Q heads: {num_q_heads}, KV heads: {num_kv_heads}, "
                      f"FFN neurons: {intermediate_size}")
        
        # 8. 가중치 적용 (한 번만!)
        print("\n 가중치 적용 중...")
        
        # 적용 통계
        applied_keys = []
        skipped_keys = []
        shape_mismatch_keys = []
        
        for key in state_dict.keys():
            if 'model.layers' in key:
                # 레이어 관련 가중치
                parts = key.split('.')
                layer_idx = int(parts[2])
                module_name = '.'.join(parts[3:])
                
                # 모델에서 해당 모듈 찾기
                layer = model.model.layers[layer_idx]
                
                try:
                    # 모듈 경로 따라가기
                    target_module = layer
                    for part in parts[3:]:
                        if hasattr(target_module, part):
                            target_module = getattr(target_module, part)
                        else:
                            skipped_keys.append(key)
                            break
                    else:
                        # weight나 bias인 경우
                        if isinstance(target_module, torch.nn.Parameter):
                            parent_path = '.'.join(parts[3:-1])
                            param_name = parts[-1]
                            parent_module = layer
                            
                            for part in parent_path.split('.'):
                                if part:
                                    parent_module = getattr(parent_module, part)
                            
                            # Shape 확인
                            saved_shape = state_dict[key].shape
                            current_param = getattr(parent_module, param_name)
                            
                            if current_param.shape == saved_shape:
                                # Shape이 일치하면 적용
                                current_param.data.copy_(state_dict[key])
                                applied_keys.append(key)
                            else:
                                print(f"    ⚠️ Shape mismatch: {key}")
                                print(f"       Expected: {current_param.shape}, Got: {saved_shape}")
                                shape_mismatch_keys.append(key)
                        elif hasattr(target_module, 'weight'):
                            # Linear 모듈의 weight
                            if target_module.weight.shape == state_dict[key].shape:
                                target_module.weight.data.copy_(state_dict[key])
                                applied_keys.append(key)
                            else:
                                print(f"    ⚠️ Shape mismatch: {key}")
                                print(f"       Expected: {target_module.weight.shape}, Got: {state_dict[key].shape}")
                                shape_mismatch_keys.append(key)
                                
                except Exception as e:
                    print(f"    ❌ Error applying {key}: {e}")
                    skipped_keys.append(key)
                    
            else:
                # 레이어 외부 가중치 (embedding, lm_head, norm 등)
                if key == 'model.embed_tokens.weight':
                    model.model.embed_tokens.weight.data.copy_(state_dict[key])
                    applied_keys.append(key)
                elif key == 'lm_head.weight':
                    model.lm_head.weight.data.copy_(state_dict[key])
                    applied_keys.append(key)
                elif key == 'model.norm.weight':
                    model.model.norm.weight.data.copy_(state_dict[key])
                    applied_keys.append(key)
                else:
                    # 기타 가중치 처리
                    try:
                        # key를 통해 모듈 찾기
                        parts = key.split('.')
                        target = model
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        
                        param_name = parts[-1]
                        if hasattr(target, param_name):
                            param = getattr(target, param_name)
                            if isinstance(param, torch.nn.Parameter):
                                param.data.copy_(state_dict[key])
                                applied_keys.append(key)
                    except:
                        skipped_keys.append(key)
        
        print(f"\n 가중치 적용 결과:")
        print(f"    적용됨: {len(applied_keys)}개")
        print(f"   ⚠️ Shape 불일치: {len(shape_mismatch_keys)}개")
        print(f"   ⏭ 건너뜀: {len(skipped_keys)}개")
        
        # 9. 디바이스 및 dtype 설정
        print(f"\n 모델을 {device}로 이동 중...")
        model = model.to(device)
        
        if model.dtype != dtype:
            print(f"   dtype을 {dtype}으로 변환 중...")
            model = model.to(dtype)
        
        # 10. 최종 검증
        print("\n 최종 검증:")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   총 파라미터: {total_params:,}")
        
        # NaN 체크
        has_nan = False
        nan_locations = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                nan_locations.append(name)
        
        if has_nan:
            print(f"   ❌ NaN 감지됨! 위치: {nan_locations[:5]}")
        else:
            print(f"    NaN 없음")
        
        # 가중치 범위 체크
        weight_stats = []
        for i in [0, len(model.model.layers)//2, len(model.model.layers)-1]:
            if i < len(model.model.layers):
                layer = model.model.layers[i]
                q_weight = layer.self_attn.q_proj.weight
                ffn_weight = layer.mlp.gate_proj.weight
                
                q_mean = q_weight.mean().item()
                q_std = q_weight.std().item()
                ffn_mean = ffn_weight.mean().item()
                ffn_std = ffn_weight.std().item()
                
                weight_stats.append({
                    'layer': i,
                    'q_mean': q_mean,
                    'q_std': q_std,
                    'ffn_mean': ffn_mean,
                    'ffn_std': ffn_std
                })
        
        print(f"\n   가중치 통계 (정상 범위 확인):")
        for stat in weight_stats:
            print(f"   Layer {stat['layer']}:")
            print(f"     Q: mean={stat['q_mean']:.6f}, std={stat['q_std']:.6f}")
            print(f"     FFN: mean={stat['ffn_mean']:.6f}, std={stat['ffn_std']:.6f}")
            
            # 이상치 감지
            if abs(stat['q_mean']) > 1.0 or stat['q_std'] > 10.0:
                print(f"     ⚠️ Q 가중치 이상 감지!")
            if abs(stat['ffn_mean']) > 1.0 or stat['ffn_std'] > 10.0:
                print(f"     ⚠️ FFN 가중치 이상 감지!")
        
        # 구조 정보 출력
        print(f"\n 프루닝된 구조:")
        sample_layers = [0, 5, 10, 15, 20, 25] if base_config['num_hidden_layers'] >= 26 else [0]
        
        for idx in sample_layers:
            if idx < len(model.model.layers):
                layer = model.model.layers[idx]
                attn = layer.self_attn
                mlp = layer.mlp
                
                print(f"   Layer {idx}:")
                print(f"     Attention: {attn.num_heads} heads (Q={attn.q_proj.weight.shape[0]}, "
                      f"K={attn.k_proj.weight.shape[0]}, V={attn.v_proj.weight.shape[0]})")
                print(f"     FFN: {mlp.gate_proj.weight.shape[0]} neurons")
        
        print(f"\n 프루닝 모델 로드 완료!")
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ 프루닝 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

def save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str,
    safe_serialization: bool = True
) -> None:
    """
    일반 모델 저장 (하위 호환성)
    
    Args:
        model: 모델
        tokenizer: 토크나이저
        save_path: 저장 경로
        safe_serialization: safetensors 형식 사용
    """
    print(f" 모델 저장 중: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(save_path)
    
    print(f" 모델 저장 완료")


def get_model_size(model: AutoModelForCausalLM) -> dict:
    """모델 크기 정보"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    memory_mb = memory_bytes / (1024 ** 2)
    
    return {
        'num_parameters': total_params,
        'trainable_parameters': trainable_params,
        'memory_mb': memory_mb,
    }


# 하위 호환성을 위한 alias
get_model_params = lambda model: sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("model_loader.py - 통합 모델 로더")
    print("=" * 60)
    print("\n사용 가능한 함수:")
    print("  1. load_model() - 원본 모델 로드")
    print("  2. save_pruned_model() - 프루닝 모델 저장")
    print("  3. load_pruned_model() - 프루닝/원본 자동 감지 로드")
    print("  4. save_model() - 일반 저장 (하위 호환)")
    print("  5. get_model_size() - 모델 크기 정보")