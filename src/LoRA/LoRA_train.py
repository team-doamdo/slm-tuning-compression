import os, sys, torch
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

BASE_DIR = r"G:\.shortcut-targets-by-id\12oVyr8-DOoCV5HEklI8mr3lhS8VoDoiP\smartfarm_pruning"

PRUNED_SRC = os.path.join(BASE_DIR, "1st", "src")
DATA_FILE = os.path.join(BASE_DIR, "1st", "data", "tomato_dataset.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sys.path.append(PRUNED_SRC)
from utils.model_loader import load_pruned_model

# activation 방식 + gradient 방식 + magnitude 방식 비율 별 자동 반복
PRUNING_METHODS = ["activation","gradient", "magnitude"]
PRUNED_RATIOS = [5, 10, 15, 20, 25, 30]

for method in PRUNING_METHODS:
    for ratio in PRUNED_RATIOS:

        print("\n" + "="*120)
        print(f"pruned_{method}_{ratio} LoRA 학습 시작")
        print("="*120)

        MODEL_DIR = os.path.join(
            BASE_DIR, "1st", "models", "pruned", method, f"pruned_{method}_{ratio}"
        )

        MERGED_DIR = os.path.join(
            BASE_DIR, "1st", "models", "LoRA", method, f"pruned_{method}_{ratio}"
        )

        os.makedirs(MERGED_DIR, exist_ok=True)

        print("Tokenizer 불러오는 중")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"프루닝된 모델 로드: pruned_{method}_{ratio}")
        model, _ = load_pruned_model(
            MODEL_DIR,
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        print("데이터셋 로드 중")
        raw = load_dataset("json", data_files={"train": DATA_FILE})["train"]

        def to_text(example):
            instr = example["instruction"].strip()
            out = example["output"].strip()
            return {
                "text": f"### Instruction:\n{instr}\n\n### Response:\n{out}\n"
            }

        train_dataset = raw.map(to_text, remove_columns=raw.column_names)

        print("SFT 설정 중")
        sft_config = SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            num_train_epochs=3,
            bf16=(device == "cuda"),
            report_to="none",
            logging_steps=50,
            save_strategy="no"
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            formatting_func=lambda ex: ex["text"],
            processing_class=tokenizer
        )

        print(f"LoRA 학습 시작: pruned_{method}_{ratio}")
        trainer.train()
        print(f"LoRA 학습 완료: pruned_{method}_{ratio}")

        print(f"LoRA 어댑터 저장 중 → {MERGED_DIR}")
        model.save_pretrained(MERGED_DIR, safe_serialization=True)
        tokenizer.save_pretrained(MERGED_DIR)

        print(f"저장 완료: pruned_{method}_{ratio}")

print("\n 전체 모델 LoRA 학습 완료")