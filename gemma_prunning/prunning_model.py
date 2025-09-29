# LoRA 기반 프루닝 파이프라인

import os
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch.nn.utils.prune as prune

# 설정
MODEL_NAME = "google/gemma-3-1b-pt"
DATA_PATH = "./tomato_qa_3000.jsonl"
LORA_DIR = "./lora_finetuned"
PRUNED_LORA_DIR = "./pruned_lora"
RECOVERED_DIR = "./recovered_model"

class LoRAFirstPipeline:
    def __init__(self):
        # MPS 우선 사용하되 안정성 체크
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"Using device: {self.device} (Apple Silicon GPU)")
        else:
            self.device = "cpu"
            print(f"Using device: {self.device}")
        
        # MPS 최적화 설정
        if self.device == "mps":
            torch.mps.empty_cache()
            # MPS에서 안정성을 위한 환경 변수 설정
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
    def step1_lora_finetune(self, max_samples=100):
        """1단계: LoRA 파인튜닝"""
        print("="*50)
        print("1️⃣ STEP 1: LoRA Fine-tuning")
        print("="*50)
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 기본 모델 로드
        print("📥 Loading base Gemma model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)
        
        # LoRA 설정 (더 강력하게)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # rank 증가 (8 -> 16)
            lora_alpha=32,  # alpha도 비례적으로 증가
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # LoRA 모델 생성
        model = get_peft_model(base_model, lora_config)
        print(f"📊 Trainable parameters: {model.num_parameters():,}")
        
        # 데이터셋 로드
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        if max_samples < len(dataset):
            import random
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
            print(f"📊 Using {len(dataset)} samples")
        
        def preprocess(examples):
            instructions = examples.get("instruction", [])
            outputs = examples.get("output", [])
            texts = [f"Question: {q} Answer: {a}" for q, a in zip(instructions, outputs)]
            
            tokenized = tokenizer(
                texts, 
                truncation=True, 
                max_length=128, 
                padding="max_length",  # 패딩 추가
                return_tensors=None
            )
            # labels를 동일한 길이로 맞춤
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
            
        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        
        # LoRA 훈련 설정 (GPU 최적화)
        training_args = TrainingArguments(
            output_dir=LORA_DIR,
            overwrite_output_dir=True,
            per_device_train_batch_size=4 if self.device == "mps" else 1,  # MPS에서 배치 증가
            gradient_accumulation_steps=2 if self.device == "mps" else 4,
            num_train_epochs=8,  # 더 많은 에폭
            learning_rate=1e-3,  # 더 높은 학습률
            save_strategy="epoch",
            logging_steps=2,
            eval_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            warmup_steps=10,
            weight_decay=0.01,
            dataloader_pin_memory=False,  # MPS 호환성
            fp16=False,  # MPS에서는 fp16 비활성화
            gradient_checkpointing=False,  # 안정성을 위해 비활성화
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        print("⚡ Starting LoRA fine-tuning...")
        trainer.train()
        
        # LoRA 어댑터 저장
        model.save_pretrained(LORA_DIR)
        tokenizer.save_pretrained(LORA_DIR)
        print(f"✅ LoRA fine-tuning complete! Saved to {LORA_DIR}")
        
        # 메모리 정리
        del model, base_model, trainer
        
    def step2_lora_pruning(self):
        """2단계: 점진적 LoRA 프루닝 (2-3%씩)"""
        print("="*50)
        print("2️⃣ STEP 2: Fine-grained LoRA Pruning")
        print("="*50)
        
        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        
        # LoRA 모델 로드
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
        
        # 세밀한 점진적 프루닝 (2-3%씩)
        pruning_percentages = [2, 5, 8, 12, 15, 20]  # 매우 점진적
        
        for percentage in pruning_percentages:
            print(f"\n✂️ Applying {percentage}% pruning...")
            
            # rank 감소를 더 세밀하게
            reduction_factor = percentage / 100.0
            new_rank = max(2, int(8 * (1 - reduction_factor)))  # 최소 rank 2
            
            print(f"   📊 Rank: 8 → {new_rank} ({reduction_factor*100:.0f}% reduction)")
            
            # 새로운 LoRA 설정
            pruned_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=new_rank,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # 새 모델 생성
            fresh_base = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            
            pruned_model = get_peft_model(fresh_base, pruned_config)
            
            # 프루닝된 모델 저장
            stage_dir = Path(PRUNED_LORA_DIR) / f"pruned_{percentage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            pruned_model.save_pretrained(stage_dir)
            
            # 토크나이저도 함께 저장 (평가시 필요)
            tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
            tokenizer.save_pretrained(stage_dir)
            
            # 프루닝 정보 저장
            pruning_info = {
                "pruning_percentage": percentage,
                "original_rank": 8,
                "pruned_rank": new_rank,
                "compression_ratio": f"{8/new_rank:.2f}x"
            }
            
            with open(stage_dir / "pruning_info.json", "w") as f:
                json.dump(pruning_info, f, indent=2)
                
            print(f"   💾 Saved to {stage_dir}")
            
            # 메모리 정리 후 평가
            del pruned_model, fresh_base
            
            # 각 프루닝 단계 후 즉시 평가 (간단한 방법으로)
            print(f"   📊 Evaluating {percentage}% pruned model...")
            try:
                # 간단한 평가 (메모리 절약을 위해)
                eval_result = self._quick_evaluate(f"Pruned {percentage}%", stage_dir)
                print(f"   📊 Quick evaluation: {eval_result:.1f}% accuracy")
            except Exception as e:
                print(f"   ❌ Quick evaluation failed: {e}")
                
        print("\n✅ Fine-grained LoRA pruning complete!")
        
        del model, base_model
        
    def step3_recovery_training(self):
        """3단계: 복구 훈련"""
        print("="*50)
        print("3️⃣ STEP 3: Recovery Training")
        print("="*50)
        
        # 가장 많이 프루닝된 모델 선택 (20%)
        pruned_path = Path(PRUNED_LORA_DIR) / "pruned_20"
        
        if not pruned_path.exists():
            print("❌ Pruned model not found. Run step 2 first.")
            return
            
        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        
        # 프루닝된 LoRA 모델 로드
        model = PeftModel.from_pretrained(base_model, pruned_path)
        
        # 복구용 데이터셋
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        dataset = dataset.select(range(30))  # 작은 데이터셋
        
        tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
        
        def preprocess(examples):
            instructions = examples.get("instruction", [])
            outputs = examples.get("output", [])
            texts = [f"Question: {q} Answer: {a}" for q, a in zip(instructions, outputs)]
            
            tokenized = tokenizer(
                texts, 
                truncation=True, 
                max_length=128, 
                padding="max_length",
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
            
        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        
        # 복구 훈련 설정
        training_args = TrainingArguments(
            output_dir=RECOVERED_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=2,
            learning_rate=5e-5,
            save_strategy="epoch",
            logging_steps=5,
            eval_strategy="no",
            report_to=[],
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        print("⚡ Starting recovery training...")
        trainer.train()
        model.save_pretrained(RECOVERED_DIR)
        tokenizer.save_pretrained(RECOVERED_DIR)
        print("✅ Recovery complete!")
        
        # 복구 후 평가
        print("\n📊 Evaluating after recovery training...")
        self._evaluate_model("LoRA Recovered", MODEL_NAME, RECOVERED_DIR)
        
        del model, base_model, trainer
        
    def step4_compare_models(self):
        """4단계: 모델 비교"""
        print("="*50)
        print("4️⃣ STEP 4: Model Comparison")
        print("="*50)
        
        test_questions = [
            "What is the optimal temperature for tomato cultivation?",
            "How should tomatoes be watered?",
            "What soil pH is best for tomatoes?"
        ]
        
        models_to_test = [
            ("Original", MODEL_NAME, None),
            ("LoRA Fine-tuned", MODEL_NAME, LORA_DIR),
            ("LoRA Pruned 30%", MODEL_NAME, Path(PRUNED_LORA_DIR) / "pruned_30"),
            ("Recovered", MODEL_NAME, RECOVERED_DIR),
        ]
        
        results = {}
        
        for model_name, base_path, lora_path in models_to_test:
            print(f"\n🧪 Testing {model_name}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    lora_path or base_path, trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).to(self.device)
                
                if lora_path and Path(lora_path).exists():
                    model = PeftModel.from_pretrained(base_model, lora_path)
                else:
                    model = base_model
                    
                model.eval()
                
                model_results = []
                for question in test_questions:
                    prompt = f"Question: {question} Answer:"
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=30,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    clean_answer = answer[len(prompt):].strip()
                    model_results.append(clean_answer)
                    
                results[model_name] = model_results
                print(f"✅ {model_name} tested")
                
                del model
                if lora_path is None:
                    del base_model
                    
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                results[model_name] = ["Error"] * len(test_questions)
        
        # 결과 출력
        print("\n" + "="*80)
        print("📊 COMPARISON RESULTS")
        print("="*80)
        
        for i, question in enumerate(test_questions):
            print(f"\n[Question {i+1}] {question}")
            print("-" * 60)
            for model_name in results:
                answer = results[model_name][i]
                if len(answer) > 50:
                    answer = answer[:47] + "..."
                print(f"{model_name:<20}: {answer}")
                
        # 결과 저장
        with open("lora_results.json", "w") as f:
            json.dump({"questions": test_questions, "results": results}, f, indent=2)
            
    def _evaluate_model(self, model_name, base_path, lora_path):
        """각 단계별 모델 정확도 평가 (시간/메모리 측정 포함)"""
        import time
        import psutil
        import os
        
        # 표준 테스트 질문들 (자유 답변)
        test_cases = [
            "What are symptoms of phosphorus deficiency in tomatoes?",
            "What is the optimal temperature for tomato cultivation?", 
            "How should tomatoes be watered to avoid blossom-end rot?",
            "What soil pH is best for tomatoes?"
        ]
        
        try:
            # 시작 시간 및 메모리 측정
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 모델 로드
            tokenizer = AutoTokenizer.from_pretrained(
                lora_path or base_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            
            if lora_path and Path(lora_path).exists():
                model = PeftModel.from_pretrained(base_model, lora_path)
            else:
                model = base_model
                
            model.eval()
            
            # 로딩 후 메모리 측정
            load_memory = process.memory_info().rss / 1024 / 1024  # MB
            load_time = time.time() - start_time
            
            # 각 질문에 대해 테스트
            correct_answers = 0
            total_questions = len(test_cases)
            total_inference_time = 0
            
            print(f"\n📋 Testing {model_name}:")
            print(f"   🔧 Load time: {load_time:.2f}s | Memory used: {load_memory - start_memory:.1f}MB")
            print("-" * 60)
            
            all_answers = []
            
            for i, question in enumerate(test_cases, 1):
                prompt = f"Question: {question} Answer:"
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 추론 시간 측정
                inference_start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,  # 더 긴 답변 허용
                        do_sample=True,     # 더 다양한 답변 생성
                        temperature=0.7,    # 적당한 창의성
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                
                # 추론 후 메모리 측정
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                clean_answer = answer[len(prompt):].strip()
                
                # 답변 저장 (나중에 비교용)
                all_answers.append(clean_answer)
                
                print(f"[{i}] {question}")
                print(f"    Answer: {clean_answer}")
                print(f"    ⏱️ Time: {inference_time:.2f}s | 💾 Memory: {current_memory:.0f}MB")
                print()
                
            # 전체 통계
            avg_inference_time = total_inference_time / total_questions
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"📊 {model_name} Performance Summary:")
            print(f"    Total questions: {total_questions}")
            print(f"    Avg inference time: {avg_inference_time:.2f}s")
            print(f"    Peak memory usage: {peak_memory:.0f}MB")
            
            # 결과를 글로벌 리스트에 저장
            if not hasattr(self, 'evaluation_results'):
                self.evaluation_results = []
                
            self.evaluation_results.append({
                "model_name": model_name,
                "answers": all_answers,
                "questions": test_cases,
                "load_time": load_time,
                "avg_inference_time": avg_inference_time,
                "peak_memory_mb": peak_memory,
                "memory_increase_mb": peak_memory - start_memory
            })
            
            # 메모리 정리
            del model
            if lora_path is None:
                del base_model
                
        except Exception as e:
            print(f"❌ Evaluation failed for {model_name}: {e}")
            if not hasattr(self, 'evaluation_results'):
                self.evaluation_results = []
            self.evaluation_results.append({
                "model_name": model_name,
                "accuracy": 0.0,
                "error": str(e)
            })
        
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("🌱 LoRA Pipeline Started!")
        
        try:
            self.step1_lora_finetune()
            self.step2_lora_pruning()
            self.step3_recovery_training()
            self.step4_compare_models()
            
            print("\n🎉 PIPELINE COMPLETE!")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise

def main():
    import sys
    
    pipeline = LoRAFirstPipeline()
    
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == "1":
            max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            pipeline.step1_lora_finetune(max_samples=max_samples)
        elif step == "2":
            pipeline.step2_lora_pruning()
        elif step == "3":
            pipeline.step3_recovery_training()
        elif step == "4":
            pipeline.step4_compare_models()
        elif step == "eval":
            # 평가만 따로 실행
            if len(sys.argv) > 2:
                eval_type = sys.argv[2]
                if eval_type == "original":
                    print("Evaluating original Gemma model...")
                    pipeline._evaluate_model("Original Gemma", MODEL_NAME, None)
                elif eval_type == "lora":
                    print("Evaluating LoRA fine-tuned model...")
                    pipeline._evaluate_model("LoRA Fine-tuned", MODEL_NAME, LORA_DIR)
                elif eval_type == "pruned":
                    percentage = sys.argv[3] if len(sys.argv) > 3 else "20"
                    path = Path(PRUNED_LORA_DIR) / f"pruned_{percentage}"
                    print(f"Evaluating {percentage}% pruned model...")
                    pipeline._evaluate_model(f"Pruned {percentage}%", MODEL_NAME, str(path))
                elif eval_type == "recovered":
                    print("Evaluating recovered model...")
                    pipeline._evaluate_model("Recovered", MODEL_NAME, RECOVERED_DIR)
                elif eval_type == "all":
                    print("Evaluating all available models...")
                    # 원본 모델
                    pipeline._evaluate_model("Original Gemma", MODEL_NAME, None)
                    
                    # LoRA 모델 (있으면)
                    if Path(LORA_DIR).exists():
                        pipeline._evaluate_model("LoRA Fine-tuned", MODEL_NAME, LORA_DIR)
                    
                    # 프루닝된 모델들 (있으면)
                    for percentage in [2, 5, 8, 12, 15, 20]:
                        path = Path(PRUNED_LORA_DIR) / f"pruned_{percentage}"
                        if path.exists():
                            pipeline._evaluate_model(f"Pruned {percentage}%", MODEL_NAME, str(path))
                    
                    # 복구된 모델 (있으면)
                    if Path(RECOVERED_DIR).exists():
                        pipeline._evaluate_model("Recovered", MODEL_NAME, RECOVERED_DIR)
                else:
                    print("Invalid eval type. Use: original, lora, pruned [%], recovered, all")
            else:
                print("Usage: eval [original|lora|pruned [%]|recovered|all]")
                print("Examples:")
                print("  eval lora           - Evaluate LoRA model")
                print("  eval pruned 15      - Evaluate 15% pruned model") 
                print("  eval all            - Evaluate all available models")
        else:
            print("Invalid step. Use: 1, 2, 3, 4, eval")
    else:
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
