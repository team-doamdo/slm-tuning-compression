import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(
    r"G:\.shortcut-targets-by-id\12oVyr8-DOoCV5HEklI8mr3lhS8VoDoiP\smartfarm_pruning"
)

ORIGINAL_MODEL_DIR = BASE_DIR / "1st" / "models" / "original" / "gemma-3-4b-it"

DATA_FILE = BASE_DIR / "1st" / "data" / "tomato_dataset.json"

SAVE_LORA_DIR = BASE_DIR / "1st" / "models" / "LoRA" / "original"
SAVE_LORA_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Tokenizer 로드")
tokenizer = AutoTokenizer.from_pretrained(
    ORIGINAL_MODEL_DIR,   
    use_fast=False,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("원본 Gemma-3-4B-IT 모델 로드")
model = AutoModelForCausalLM.from_pretrained(
    ORIGINAL_MODEL_DIR,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
).to(device)

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

raw = load_dataset("json", data_files={"train": str(DATA_FILE)})["train"]

def to_text(example):
    return {
        "text": (
            f"### Instruction:\n{example['instruction'].strip()}\n\n"
            f"### Response:\n{example['output'].strip()}\n"
        )
    }

train_dataset = raw.map(to_text, remove_columns=raw.column_names)

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

print("LoRA 학습 시작, Gemma-3-4B-IT (원본 모델)")
trainer.train()
print("LoRA 학습 완료")

print(f"LoRA 어댑터 저장: {SAVE_LORA_DIR}")
model.save_pretrained(SAVE_LORA_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_LORA_DIR)