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

METHOD = "gradient"
PRUNED_RATIOS = [5, 10, 15, 20, 25, 30]

for ratio in PRUNED_RATIOS:
    print(f"\nGradient pruning {ratio}% LoRA training")

    MODEL_DIR = os.path.join(BASE_DIR, "1st", "models", "pruned", METHOD, f"pruned_{METHOD}_{ratio}")
    SAVE_DIR  = os.path.join(BASE_DIR, "1st", "models", "LoRA", METHOD, f"pruned_{METHOD}_{ratio}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = load_pruned_model(
        MODEL_DIR,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files={"train": DATA_FILE})["train"]

    def to_text(ex):
        return {
            "text": f"### Instruction:\n{ex['instruction'].strip()}\n\n### Response:\n{ex['output'].strip()}\n"
        }

    train_dataset = raw.map(to_text, remove_columns=raw.column_names)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            num_train_epochs=3,
            bf16=(device == "cuda"),
            logging_steps=50,
            report_to="none",
            save_strategy="no",
        ),
        train_dataset=train_dataset,
        formatting_func=lambda ex: ex["text"],
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)

print("Gradient pruning LoRA 완료")
