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

# importance 계산 및 파일 저장
importance = {}

for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight") and module.weight.grad is not None:
            key = (module, "weight")  # 모듈 객체 자체를 키로 사용
            if key not in importance:
                importance[key] = torch.zeros_like(module.weight, device=device)
            importance[key] += module.weight.grad.abs()

    model.zero_grad()

torch.save(importance, "importance.pt")

# 저장한 importance 계산 파일 불러오기
# importance = torch.load("importance.pt", map_location=device)

# --- 4. 프루닝 적용 (L1 Unstructured) ---
print("프루닝 적용 중...")
pruning_ratio = 0.3
pruning_targets = list(importance.keys())  # importance에서 뽑은 모듈-파라미터 쌍

if pruning_targets:
    prune.global_unstructured(
        pruning_targets,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio
    )
    print(f"프루닝 완료 (제거율: {pruning_ratio*100:.0f}%)")
else:
    print("프루닝할 파라미터를 찾을 수 없습니다.")


# --- 5. 마스크 제거 ---
for module, name in pruning_targets:
    prune.remove(module, name)

# --- 6. 모델 저장 ---
output_dir = "./pruned_model_m3pro"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"경량화된 모델 저장 완료: {output_dir}")