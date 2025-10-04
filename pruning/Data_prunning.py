import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune

# --- 1. 데이터셋 클래스 정의 ---
class TomatoDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f) 
        self.tokenizer = tokenizer
        self.chat_template = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        formatted_text = self.chat_template.format(instruction=item["instruction"], output=item["output"])
        return self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=256,
            padding="max_length"
        )

# --- 2. 모델 및 데이터 로더 ---
data_path = "tomato.json"
model_name = "google/gemma-3-1b-it"

print("모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager"
)

# M3 Pro Metal(MPS) 디바이스 사용
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.train()  # gradient 계산 위해

dataset = TomatoDataset(data_path, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=data_collator)

# --- 3. 중요도 계산 ---
print("데이터 기반 중요도 계산 중...")
importance = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters() if "weight" in name}

for batch in dataloader:
    # batch dict를 MPS로 이동 (squeeze 제거)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # DataCollator가 이미 labels를 생성했으므로, batch를 그대로 전달
    outputs = model(**batch)
    
    loss = outputs.loss
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name and param.grad is not None:
            importance[name] += param.grad.abs()

    model.zero_grad()

# --- 4. 프루닝 적용 (L1 Unstructured) ---
print("프루닝 적용 중...")
pruning_ratio = 0.3
pruning_targets = [(module, "weight") for module in model.modules() if isinstance(module, torch.nn.Linear)]

# 중요도 flatten으로 global_unstructured에 전달
flattened_importance = []
for module, name in pruning_targets:
    importance_key = f"{module._get_name()}.{name}"
    if importance_key in importance:
        flattened_importance.append(importance[importance_key].flatten())

if flattened_importance:
    prune.global_unstructured(
        parameters_to_prune=pruning_targets,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio
    )
    print(f"프루닝 완료 (제거율: {pruning_ratio*100:.0f}%)")
else:
    print("프루닝할 파라미터를 찾을 수 없습니다. (importance 딕셔너리에 매칭되는 키가 없습니다.)")


# --- 5. 마스크 제거 ---
for module, name in pruning_targets:
    prune.remove(module, name)

# --- 6. 모델 저장 ---
output_dir = "./pruned_model_m3pro"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"경량화된 모델 저장 완료: {output_dir}")