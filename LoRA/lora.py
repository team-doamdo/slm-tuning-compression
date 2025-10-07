import torch
import json
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from model_loader import load_pruned_model

# 0️⃣ 디바이스 설정 (Mac용 MPS fallback 포함)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 1️⃣ 프루닝된 모델 로드
model_path = "./pruning_gem"
model, tokenizer = load_pruned_model(
    model_path, 
    device=device, 
    dtype=torch.float16 if device.type == "cuda" else torch.float32
)

# 2️⃣ LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3️⃣ 데이터셋 로드
with open("/Users/tenedict/Desktop/p_teest/test/tomato_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list([
    {"prompt": f"{item.get('instruction','')}\nAnswer:", "response": item.get("output","")} 
    for item in raw_data
])

# 4️⃣ 토크나이즈 및 입력/라벨 구성
def tokenize_fn(example):
    prompt_enc = tokenizer(example["prompt"], truncation=True, max_length=512)
    response_enc = tokenizer(example["response"], truncation=True, max_length=256)
    input_ids = prompt_enc["input_ids"] + response_enc["input_ids"]
    labels = [-100]*len(prompt_enc["input_ids"]) + response_enc["input_ids"]
    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["prompt", "response"])

# 5️⃣ DataCollator 설정 (자동 padding)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    padding=True, 
    return_tensors="pt"
)

# 6️⃣ 학습 설정
training_args = TrainingArguments(
    output_dir="./lora_tomato",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=False,  # MPS/CPU 환경에서는 False
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# 7️⃣ Trainer 초기화
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator
)

# 8️⃣ 학습 시작
trainer.train()

# 9️⃣ LoRA 모델 저장
model.save_pretrained("./lora_tomato")