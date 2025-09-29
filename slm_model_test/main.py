# 기본 라이브러리
import os
import gc
import copy
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 수치 연산 라이브러리
import numpy as np

# PyTorch 관련
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# Transformers 및 PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset as HFDataset

# 유틸리티
import psutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# 경고 무시
warnings.filterwarnings('ignore')

# ============ 설정 정보 ============
MODEL_CONFIG = {
    "original_model": "google/gemma-3-1b-pt",
    "hf_token": "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"
}

PRUNING_CONFIG = {
    "structured_ratios": [0.95, 0.92, 0.90],  # 5%, 8%, 10%만 제거
    "protect_layers": [0, 1, 24, 25],  # 입출력 레이어만 보호
    "min_intermediate_size": 5500  # 최소 크기 상향
}

# LoRA 설정 - 메모리 최적화
LORA_CONFIG = {
    "r": 16,  
    "lora_alpha": 32,  
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], 
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

TRAINING_CONFIG = {
    "learning_rate": 2e-5,  
    "batch_size": 1,
    "epochs": 3,  
    "optimizer": "AdamW",
    "warmup_steps": 100
}

# 농업 관련 중요 토큰 (영어)
IMPORTANT_TOKENS = [
    "tomato", "plant", "leaf", "root", "fruit", "flower", "seed",
    "temperature", "humidity", "light", "water", "soil", "nutrient",
    "disease", "pest", "growth", "harvest", "greenhouse", "farming",
    "cultivation", "irrigation", "fertilizer", "pH", "EC", "PPM",
    "fungus", "bacteria", "virus", "blight", "wilt", "spot", "rot",
    "nitrogen", "phosphorus", "potassium", "calcium", "magnesium"
]

# ============ 1. 데이터셋 클래스 ============
class TomatoQADataset(Dataset):
    """토마토 스마트팜 Q&A 데이터셋"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # JSON 데이터 로드
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} QA pairs from {data_path}")
        except FileNotFoundError:
            print(f"Warning: {data_path} not found. Creating sample dataset...")
            self.data = self._create_sample_data()
            self._save_sample_data(data_path)
    
    def _create_sample_data(self):
        """샘플 데이터 생성"""
        return [
            {
                "instruction": "My tomato leaves have brown spots with yellow halos. What should I do?",
                "input": "",
                "output": "This appears to be early blight. Remove affected leaves, improve air circulation, avoid overhead watering, apply copper fungicide every 7-10 days.",
                "category": "disease_management"
            }
        ]
    
    def _save_sample_data(self, path):
        """샘플 데이터 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"Sample dataset saved to {path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 프롬프트 생성
        prompt = f"Question: {item['instruction']}\n"
        if item.get('input'):
            prompt += f"Context: {item['input']}\n"
        prompt += f"Answer: {item['output']}"
        
        # 토크나이징
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

# ============ 2. 프루닝 모듈 ============
class ModelPruner:
    """모델 경량화를 위한 프루닝 클래스"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # M1 Mac device 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 중요 토큰 ID 추출
        self.important_token_ids = self._get_important_token_ids()
        print(f"Identified {len(self.important_token_ids)} important token IDs for protection")

    
    def _get_important_token_ids(self):
        """농업 관련 중요 토큰 ID 추출"""
        token_ids = []
        for token in IMPORTANT_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            token_ids.extend(ids)
        return list(set(token_ids))
    
    def evaluate_layer_importance(self, sample_texts=None):
        """레이어별 중요도 평가 - 가중치 기반 분석으로 변경"""
        print("Evaluating layer importance using weight analysis...")
        
        layer_importance = {}
        
        # Gemma 모델의 레이어별 역할 기반 중요도 사전 정의
        num_layers = len(self.model.model.layers)
        
        for i in range(num_layers):
            layer = self.model.model.layers[i]
            
            # 1. 위치 기반 중요도 (입출력층은 매우 중요)
            position_score = 0
            if i < 4:  # 입력층
                position_score = 100
            elif i >= num_layers - 4:  # 출력층
                position_score = 100
            else:  # 중간층
                position_score = 50
            
            # 2. 가중치 크기 기반 중요도
            weight_score = 0
            with torch.no_grad():
                # Attention 가중치 확인
                if hasattr(layer, 'self_attn'):
                    for param_name, param in layer.self_attn.named_parameters():
                        if 'weight' in param_name:
                            weight_score += param.abs().mean().item()
                
                # FFN 가중치 확인
                if hasattr(layer, 'mlp'):
                    for param_name, param in layer.mlp.named_parameters():
                        if 'weight' in param_name:
                            weight_score += param.abs().mean().item() * 0.5  # FFN은 가중치 낮게
            
            # 3. 토마토 관련 토큰에 대한 반응성
            if sample_texts is None:
                sample_texts = [
                    "Tomato cultivation requires careful attention",
                    "The optimal growing conditions for tomatoes",
                    "Disease management in tomato plants"
                ]
            
            activation_score = 0
            try:
                self.model.eval()
                with torch.no_grad():
                    for text in sample_texts:
                        inputs = self.tokenizer(text, return_tensors="pt", max_length=50, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # 레이어의 출력 크기 측정
                        hidden_states = self.model.model.embed_tokens(inputs['input_ids'])
                        for j in range(i + 1):
                            hidden_states = self.model.model.layers[j](hidden_states)[0]
                        activation_score += hidden_states.abs().mean().item()
            except:
                # 활성화 측정 실패시 기본값
                activation_score = 10
            
            # 종합 점수
            layer_importance[i] = position_score + weight_score + activation_score
        
        # 정규화
        max_importance = max(layer_importance.values()) if layer_importance else 1
        for idx in layer_importance:
            layer_importance[idx] = (layer_importance[idx] / max_importance) * 100
        
        # 결과 출력
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        print("Layer importance ranking (high to low):")
        for idx, (layer_id, score) in enumerate(sorted_layers[:10]):
            print(f"  Layer {layer_id}: {score:.2f}")
        
        return layer_importance
    
    def validate_pruned_model(self, test_prompts=None):
        """프루닝된 모델의 텍스트 생성 능력 엄격 검증"""
        print("\n[Validating Pruned Model]")
        print("="*60)
        
        import re
        
        if test_prompts is None:
            test_prompts = [
                "What causes brown spots on tomato leaves?",
                "What temperature is best for tomato growth?",
                "When should I prune tomato suckers?",
                "What pH level do tomatoes prefer?",
                "How often should tomatoes be watered?"
            ]
        
        self.model.eval()
        passed_tests = 0
        total_tests = len(test_prompts)
        
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_only = generated[len(prompt):].strip()
            
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            
            # 검증 체크리스트
            checks = {
                "has_content": len(generated_only.split()) >= 3,
                # ASCII + 일반적인 구두점 + degree 기호 허용
                "is_english": not bool(re.search(r'[^\x00-\x7F\s°''""–—]', generated_only)),
                "no_special_tokens": all(token not in generated_only for token in ['<unused', '|', '[', ']']),
                "has_letters": any(c.isalpha() for c in generated_only),
                "not_repetitive": len(set(generated_only.split())) >= len(generated_only.split()) * 0.5,
                # 한글, 중국어, 일본어, 아랍어 등 명백한 다른 언어 감지
                "no_foreign_scripts": not bool(re.search(r'[\u0600-\u06FF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]', generated_only))
            }
            
            # 각 체크 결과 출력
            failed_checks = [check for check, passed in checks.items() if not passed]
            
            if failed_checks:
                print(f"Failed checks: {', '.join(failed_checks)}")
            else:
                print("✓ All checks passed")
                passed_tests += 1
            
            print()
        
        success_rate = passed_tests / total_tests
        print(f"Validation result: {passed_tests}/{total_tests} tests passed ({success_rate*100:.0f}%)")
        
        # 80% 이상 통과시 성공
        if success_rate >= 0.8:
            print("✓ Model validation passed")
            return True
        else:
            print("✗ Model validation failed - text generation compromised")
            return False
    
    def safe_structured_prune_mlp(self, keep_ratio: float = 0.95, protect_layers=None):
        """안전한 구조적 FFN 프루닝 - 더 보수적으로 수정"""
        print(f"\n[Safe Structured Pruning - FFN]")
        print("="*60)
        print(f"Target: Keep {keep_ratio*100:.0f}% of intermediate neurons")
        
        # 1. 레이어 중요도 평가
        layer_importance = self.evaluate_layer_importance()
        
        # 중요도가 모두 0이면 프루닝 중단
        if all(score < 0.01 for score in layer_importance.values()):
            print("ERROR: Layer importance measurement failed. Aborting pruning.")
            return self.model
        
        # 2. 보호 레이어 설정 - 수정: 보호 레이어 줄이기
        num_layers = len(self.model.model.layers)
        if protect_layers is None:
            protect_layers = set()
            
            # 입출력 레이어 보호 (처음 3개, 마지막 3개로 축소)
            protect_layers.update(range(3))  # 0, 1, 2
            protect_layers.update(range(num_layers-3, num_layers))  # 23, 24, 25
            
            # 중요도 상위 30% 레이어만 보호 (기존 50%에서 축소)
            sorted_importance = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
            num_important = len(sorted_importance) // 3  # 30%
            for layer_id, score in sorted_importance[:num_important]:
                if score > 80:  # 중요도 80 이상인 경우만
                    protect_layers.add(layer_id)
        
        print(f"Protected layers ({len(protect_layers)}): {sorted(protect_layers)}")
        
        # 3. 프루닝 적용
        layers = self.model.model.layers
        first_mlp = layers[0].mlp if hasattr(layers[0], 'mlp') else None
        
        if first_mlp is None:
            print("ERROR: No MLP layers found")
            return self.model
        
        original_inter = first_mlp.gate_proj.out_features
        new_inter = int(original_inter * keep_ratio)
        
        # 최소 크기 보장도 조정
        min_size = int(original_inter * 0.7)  
        new_inter = max(new_inter, min_size)
        
        print(f"Intermediate size: {original_inter} → {new_inter} (keeping {new_inter/original_inter*100:.0f}%)")
        
        pruned_count = 0
        failed_count = 0
        
        for li, layer in enumerate(layers):
            if li in protect_layers:
                print(f"  Layer {li}: Protected (importance: {layer_importance.get(li, 0):.2f})")
                continue
            
            if not hasattr(layer, 'mlp'):
                print(f"  Layer {li}: No MLP module")
                continue
            
            mlp = layer.mlp
            
            try:
                # 가중치 중요도 기반 프루닝
                with torch.no_grad():
                    # Gate와 Up 프로젝션의 중요도 계산
                    gate_importance = mlp.gate_proj.weight.data.abs().mean(dim=1)
                    up_importance = mlp.up_proj.weight.data.abs().mean(dim=1)
                    combined_score = (gate_importance + up_importance) / 2
                    
                    # 상위 뉴런 선택
                    keep_idx = torch.topk(combined_score, k=new_inter).indices.sort()[0]
                    
                    # 새로운 레이어 생성 및 가중치 복사
                    new_gate = nn.Linear(mlp.gate_proj.in_features, new_inter, 
                                        bias=mlp.gate_proj.bias is not None)
                    new_gate = new_gate.to(mlp.gate_proj.weight.device).to(mlp.gate_proj.weight.dtype)
                    new_gate.weight.data = mlp.gate_proj.weight.data[keep_idx, :]
                    if mlp.gate_proj.bias is not None:
                        new_gate.bias.data = mlp.gate_proj.bias.data[keep_idx]
                    
                    new_up = nn.Linear(mlp.up_proj.in_features, new_inter,
                                    bias=mlp.up_proj.bias is not None)
                    new_up = new_up.to(mlp.up_proj.weight.device).to(mlp.up_proj.weight.dtype)
                    new_up.weight.data = mlp.up_proj.weight.data[keep_idx, :]
                    if mlp.up_proj.bias is not None:
                        new_up.bias.data = mlp.up_proj.bias.data[keep_idx]
                    
                    new_down = nn.Linear(new_inter, mlp.down_proj.out_features,
                                        bias=mlp.down_proj.bias is not None)
                    new_down = new_down.to(mlp.down_proj.weight.device).to(mlp.down_proj.weight.dtype)
                    new_down.weight.data = mlp.down_proj.weight.data[:, keep_idx]
                    if mlp.down_proj.bias is not None:
                        new_down.bias.data = mlp.down_proj.bias.data
                    
                    # 레이어 교체
                    mlp.gate_proj = new_gate
                    mlp.up_proj = new_up
                    mlp.down_proj = new_down
                    
                    pruned_count += 1
                    print(f"  Layer {li}: Pruned to {new_inter} neurons")
                    
            except Exception as e:
                print(f"  Layer {li}: Pruning failed - {str(e)}")
                failed_count += 1
        
        # Config 업데이트
        if hasattr(self.model.config, 'intermediate_size'):
            self.model.config.intermediate_size = new_inter
        
        print(f"\nPruning summary: {pruned_count} layers pruned, {failed_count} failed, {len(protect_layers)} protected")
        
        return self.model

    def update_model(self, model):
        """프루닝 대상 모델 업데이트"""
        self.model = model
        return self

# ============ 3. LoRA 파인튜닝 모듈 ============
class LoRAFineTuner:
    """LoRA 파인튜닝 클래스"""
    
    def __init__(self, model, tokenizer, dataset_path: str):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = TomatoQADataset(dataset_path, tokenizer)
        
        # M1 Mac device 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 모델을 float32로 변환 (안정성 향상)
        self.model = self.model.float()
        print("Model converted to float32 for training stability")
    
    def prepare_model(self):
        """LoRA 적용"""
        print(f"\n[Step 4] Applying LoRA Configuration")
        print("="*60)
        
        # 모델을 학습 가능하도록 설정
        self.model.train()
        
        # 그래디언트 체크포인팅 활성화 (메모리 절약)
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=LORA_CONFIG["target_modules"],
            bias=LORA_CONFIG["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        # LoRA 적용
        self.model = get_peft_model(self.model, lora_config)
        
        # LoRA 파라미터 정보 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"  LoRA rank: {LORA_CONFIG['r']}")
        print(f"  LoRA alpha: {LORA_CONFIG['lora_alpha']}")
        print(f"  Target modules: {', '.join(LORA_CONFIG['target_modules'])}")
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Total params: {total_params:,}")
        
        return self.model
    
    def train(self):
        """LoRA 파인튜닝 실행 - 안정화 버전"""
        print(f"\n[Step 5] Starting LoRA Fine-tuning")
        print("="*60)
        
        # 데이터로더 생성
        dataloader = DataLoader(
            self.dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
        # 옵티마이저 설정 - 작은 learning rate
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=0.01
        )
        
        # Learning rate scheduler 추가
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=len(dataloader) * TRAINING_CONFIG["epochs"],
            eta_min=1e-7
        )
        
        # 학습 루프
        self.model.train()
        self.model.to(self.device)
        
        total_steps = len(dataloader) * TRAINING_CONFIG["epochs"]
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        valid_loss_count = 0
        total_loss = 0.0
        
        for epoch in range(TRAINING_CONFIG["epochs"]):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # 배치 데이터를 디바이스로 이동
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Gradient 초기화
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Loss 유효성 체크
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        
                        # Gradient norm 체크
                        total_norm = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        # Gradient가 너무 크면 스킵
                        if total_norm < 100:  # threshold
                            optimizer.step()
                            scheduler.step()
                            
                            epoch_loss += loss.item()
                            batch_count += 1
                            valid_loss_count += 1
                            total_loss += loss.item()
                        else:
                            print(f"\nSkipping batch {batch_idx} due to large gradient norm: {total_norm:.2f}")
                    else:
                        # NaN loss 발생 시 모델 복구 시도
                        print(f"\nNaN/Inf loss at batch {batch_idx}, skipping...")
                        # 이전 체크포인트로 복구하거나 계속 진행
                        
                except RuntimeError as e:
                    print(f"\nError at batch {batch_idx}: {e}")
                    continue
                
                # 진행 상황 업데이트
                progress_bar.update(1)
                if valid_loss_count > 0:
                    progress_bar.set_postfix({'loss': total_loss / valid_loss_count})
                
                # 메모리 정리 (매 10 배치마다)
                if batch_idx % 10 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['epochs']}, Average Loss: {avg_loss:.4f}")
                print(f"Valid batches: {batch_count}/{len(dataloader)}")
            else:
                print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['epochs']}, No valid losses recorded")
                print("WARNING: Training may have failed. Consider checking model and data.")
        
        progress_bar.close()
        
        # 학습 실패 체크
        if valid_loss_count == 0:
            print("\n WARNING: No valid training steps completed. Model may be unusable.")
        
        return self.model

# ============ 4. 성능 측정 모듈 ============
class PerformanceEvaluator:
    """모델 성능 측정 클래스"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # M1 Mac device 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  
        else:
            self.device = torch.device("cpu")
        
        self.validation_questions = self._load_validation_questions()

    
    def _load_validation_questions(self):
        """검증용 질문 로드"""
        try:
            with open('validation_questions.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 기본 검증 질문
            return [
                "My tomato leaves have brown spots. What should I do?",
                "Temperature is 35C in greenhouse. Emergency help needed.",
                "When should I start pruning tomato suckers?",
                "pH is 5.2, EC is 4.1, plants are wilting. Diagnose the problem.",
                "White powdery substance on tomato leaves. What is this?"
            ]
    
    def measure_response(self, question: str) -> Dict:
        """단일 질문에 대한 응답 성능 측정"""
        # 메모리 측정 준비
        process = psutil.Process()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 시작 측정
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # 모델 추론
        self.model.eval()
        with torch.no_grad():
            try:
                inputs = self.tokenizer(
                    f"Question: {question}\nAnswer:",
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True 
                ).to(self.device)
                
                # temperature와 top_p 조정으로 안정성 향상
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100, 
                    temperature=0.7,      
                    do_sample=False,      
                    top_p=0.9,          
                    top_k=40,            
                    num_beams=1,         # beam search 비활성화
                    early_stopping=True,  
                    use_cache=True,     
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Answer: 이후의 텍스트만 추출
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                    
            except Exception as e:
                print(f"Error generating response: {e}")
                response = "Error: Unable to generate response due to model instability."
        
        # 종료 측정
        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "question": question,
            "response": response,
            "response_time": end_time - start_time,
            "memory_used": memory_after - memory_before,
            "total_memory": memory_after
        }
    
    def evaluate_all(self) -> List[Dict]:
        """모든 검증 질문에 대한 평가"""
        print(f"\n[Step 6] Performance Evaluation")
        print("="*60)
        
        results = []
        for i, question in enumerate(self.validation_questions, 1):
            print(f"\nEvaluating question {i}/{len(self.validation_questions)}...")
            result = self.measure_response(question)
            results.append(result)
            
            print(f"  Response time: {result['response_time']:.2f}s")
            print(f"  Memory used: {result['memory_used']:.2f}MB")
            print(f"  Response preview: {result['response'][:100]}...")
        
        return results
    
    def generate_report(self, results: List[Dict], model_size: float):
        """성능 리포트 생성"""
        print(f"\n[Step 7] Performance Report")
        print("="*60)
        
        # 통계 계산
        avg_response_time = np.mean([r['response_time'] for r in results])
        max_response_time = np.max([r['response_time'] for r in results])
        avg_memory = np.mean([r['total_memory'] for r in results])
        max_memory = np.max([r['total_memory'] for r in results])
        
        # 목표 달성 여부
        time_goal_met = max_response_time < 10.0
        memory_goal_met = model_size < 6000  # MB
        
        # 리포트 출력
        print("\n PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Model Size: {model_size:.2f} MB {'✓' if memory_goal_met else '✗'} (Target: <6000 MB)")
        print(f"Avg Response Time: {avg_response_time:.2f}s")
        print(f"Max Response Time: {max_response_time:.2f}s {'✓' if time_goal_met else '✗'} (Target: <10s)")
        print(f"Avg Memory Usage: {avg_memory:.2f} MB")
        print(f"Max Memory Usage: {max_memory:.2f} MB")
        
        print("\n SAMPLE RESPONSES")
        print("-" * 40)
        for i, result in enumerate(results[:3], 1):
            print(f"\nQ{i}: {result['question']}")
            print(f"A{i}: {result['response'][:200]}...")
            print(f"Time: {result['response_time']:.2f}s, Memory: {result['total_memory']:.2f}MB")
        
        # 리포트 저장
        report = {
            "model_size_mb": float(model_size),  # numpy float 변환
            "avg_response_time": float(avg_response_time),
            "max_response_time": float(max_response_time),
            "avg_memory_mb": float(avg_memory),
            "max_memory_mb": float(max_memory),
            "time_goal_met": bool(time_goal_met),  # Python bool로 변환
            "memory_goal_met": bool(memory_goal_met),
            "detailed_results": results
        }
        
        with open('performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n Report saved to performance_report.json")
        
        return report


# ============ 5. 메인 실행 함수 ============
def get_model_size(model) -> float:
    """모델 크기 계산 (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

# 프루닝 전후 비교를 위한 검증 코드
def verify_pruning(model):
    """프루닝 검증"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total = param.numel()
            zeros = (param == 0).sum().item()
            total_params += total
            zero_params += zeros
            
            if zeros > 0:
                print(f"{name}: {zeros}/{total} zeros ({zeros/total*100:.1f}%)")
    
    print(f"\nOverall sparsity: {zero_params/total_params*100:.1f}%")
    return zero_params / total_params

def main():
    """메인 실행 함수"""
    print("="*60)
    print("TOMATO SMART FARM CHATBOT LIGHTWEIGHT PROJECT")
    print("="*60)
    print(f"Target: <10s response, <6GB memory")
    print(f"Model: {MODEL_CONFIG['original_model']}")
    print("="*60)
    
    # 메모리 최적화를 위한 환경변수 설정 (수정됨)
    import os
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 메모리 제한 해제
    
    # M1 GPU (MPS) 정보 출력
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon GPU (MPS)")
        print("Metal Performance Shaders enabled")
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Device: CPU")
    
    print("\n" + "="*60)
    
    # 1. 모델 및 토크나이저 로드
    print("\n[Initialization] Loading model and tokenizer...")
    print("="*60)
    
    try:
        # M1 Mac용 device 설정
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["original_model"],
            token=MODEL_CONFIG["hf_token"]
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # M1용 모델 로딩 - eager attention 사용
        print("Loading model for Apple Silicon...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["original_model"],
            token=MODEL_CONFIG["hf_token"],
            torch_dtype=torch.float32,  # float16 대신 float32 사용
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation='eager'
        )
        
        # 모델을 MPS로 이동
        if torch.backends.mps.is_available():
            model = model.to(device)
            print("Model moved to MPS (Apple GPU)")
        else:
            print("Model loaded on CPU")
        
        print("Model and tokenizer loaded successfully")
        original_size = get_model_size(model)
        print(f"Original model size: {original_size:.2f} MB")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your HuggingFace token and model name")
        return
    
    # 2. 프루닝 적용 부분 
    print("\n" + "="*60)
    print("STARTING SAFE PRUNING PROCESS")
    print("="*60)

    pruner = ModelPruner(model, tokenizer)

    # 단계적으로 프루닝 적용
    try:
        best_model = None
        best_size = original_size
        best_ratio = 1.0
        
        # 매우 보수적인 프루닝 비율로 시작
        for keep_ratio in [0.90, 0.85, 0.80, 0.75]:  # 5%, 8%, 10% 프루닝만
            print(f"\n>>> Attempting pruning with {keep_ratio*100:.0f}% neurons kept...")
            
            # 모델 복사 - 깨끗한 복사본 사용
            if best_model is None:
                model_backup = copy.deepcopy(model)
            else:
                model_backup = copy.deepcopy(best_model)
            
            # 새로운 pruner 인스턴스 생성 (구조 변경 대응)
            pruner = ModelPruner(model_backup, tokenizer)
            
            # 프루닝 적용
            pruned_model = pruner.safe_structured_prune_mlp(keep_ratio=keep_ratio)
            
            # 엄격한 검증
            if pruner.validate_pruned_model():
                pruned_size = get_model_size(pruned_model)
                print(f"✓ Model size after pruning: {pruned_size:.2f} MB")
                print(f"✓ Size reduction: {(1 - pruned_size/original_size)*100:.1f}%")
                
                # 성공한 모델 저장
                best_model = pruned_model
                best_size = pruned_size
                best_ratio = keep_ratio
                
                # 목표 크기 달성 확인
                if pruned_size < 3000:  # 목표 크기 (약 3.2GB)
                    print(f"✓ Target size achieved with {keep_ratio*100:.0f}% keep ratio")
                    break
            else:
                print(f"✗ Validation failed at {keep_ratio*100:.0f}%, stopping further pruning")
                # 실패시 이전 최적 모델 유지하고 중단
                break
        
        # 최종 모델 설정
        if best_model is not None:
            model = best_model
            pruned_size = best_size
            print(f"\n✓ Final pruning: {best_ratio*100:.0f}% neurons kept, size: {pruned_size:.2f} MB")
        else:
            print("\n✗ All pruning attempts failed, using original model")
            pruned_size = original_size
            
    except Exception as e:
        print(f"Error during pruning: {e}")
        import traceback
        traceback.print_exc()
        pruned_size = original_size
    
    # 4. LoRA 파인튜닝 (메모리 최적화)
    print("\n" + "="*60)
    print("LORA FINE-TUNING")
    print("="*60)
    
    # 메모리 절약을 위한 설정 조정
    LORA_CONFIG["r"] = 4  # 더 작은 rank
    LORA_CONFIG["target_modules"] = ["q_proj", "v_proj"]  # 모듈 수 줄이기
    
    finetuner = LoRAFineTuner(model, tokenizer, "tomato_qa_dataset.json")
    
    try:
        model = finetuner.prepare_model()
        
        # 메모리 정리
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        model = finetuner.train()
    except Exception as e:
        print(f"Error during training: {e}")
        print("Attempting to continue with evaluation...")
    
    # 최종 모델 크기
    final_size = get_model_size(model)
    print(f"\nFinal model size: {final_size:.2f} MB")
    
    # 5. 성능 평가
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    
    try:
        evaluator = PerformanceEvaluator(model, tokenizer)
        results = evaluator.evaluate_all()
        report = evaluator.generate_report(results, final_size)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        report = {
            "time_goal_met": False,
            "memory_goal_met": final_size < 6000
        }
    
    # 6. 모델 저장
    print(f"\n[Step 8] Saving optimized model...")
    print("="*60)
    
    try:
        output_dir = "./optimized_tomato_model"
        
        # 양자화된 모델은 특별한 저장 방식 필요
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
        else:
            torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
        
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # 7. 최종 요약
    print("\n" + "="*60)
    print("PROJECT COMPLETED")
    print("="*60)
    print(f"Original size: {original_size:.2f} MB")
    print(f"After pruning: {pruned_size:.2f} MB")
    print(f"Final size: {final_size:.2f} MB")
    print(f"Total size reduction: {(1 - final_size/original_size)*100:.1f}%")
    
    print("\nOptimization techniques applied:")
    print("- Unstructured pruning (30% sparsity)")
    print("- LoRA fine-tuning")
    
    if 'report' in locals():
        if report.get('time_goal_met'):
            print(f"\nResponse time goal (<10s): ACHIEVED")
        else:
            print(f"\nResponse time goal (<10s): NOT MET")
        
        if report.get('memory_goal_met'):
            print(f"Memory goal (<6GB): ACHIEVED")
        else:
            print(f"Memory goal (<6GB): NOT MET")
    
    print("="*60)
    print("\nCleanup completed. Project finished.")

if __name__ == "__main__":
    main()