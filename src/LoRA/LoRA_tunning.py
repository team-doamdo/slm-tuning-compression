#!/usr/bin/env python3
"""
train_lora_with_logging.py
- 프루닝된 모델을 기반으로 LoRA 어댑터 학습
- M3 Pro (MPS) 및 CUDA 환경 모두 대응
- 실시간 로그: tqdm 콘솔 + optional Weights & Biases (wandb)
- 권장: batch_size=1, grad_accum=4, max_length=256, r=8
"""

import os
import math
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# optional logging
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# -----------------------
# Dataset
# -----------------------
class TomatoDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256, template=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template or "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.template.format(instruction=item["instruction"], output=item["output"])
        tok = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}


def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------
# Evaluation utilities
# -----------------------
@torch.no_grad()
def evaluate_loss(model, dataloader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
        else:
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


# -----------------------
# Training loop
# -----------------------
def train(
    data_path,
    pruned_model_path,
    out_dir,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
    epochs=3,
    batch_size=1,
    grad_accum_steps=4,
    max_length=256,
    lr=2e-4,
    weight_decay=0.0,
    fp16=False,
    use_wandb=False,
    save_every_epoch=True,
    seed=42,
    eval_split_ratio=0.05,
):
    torch.manual_seed(seed)

    # device 선택
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True if fp16 else False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False  # MPS: amp not supported reliably
    else:
        device = torch.device("cpu")
        use_amp = False

    print(f"[info] device: {device}, use_amp: {use_amp}")

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(pruned_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_dataset = TomatoDataset(data_path, tokenizer, max_length=max_length)
    n_total = len(full_dataset)
    n_eval = max(1, int(n_total * eval_split_ratio))
    n_train = n_total - n_eval
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_eval])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"[info] dataset sizes: total={n_total}, train={n_train}, eval={n_eval}")

    # 모델 로드 (프루닝된 모델)
    print("[info] loading pruned model ...")
    model = AutoModelForCausalLM.from_pretrained(pruned_model_path, torch_dtype=torch.float32)
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    # PeFT LoRA 설정
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # trainable param counts
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable params: {trainable_count:,} / total params: {total_count:,}")

    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # optimizer / scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    total_updates = math.ceil(len(train_loader) * epochs / grad_accum_steps)
    warmup_steps = max(1, int(0.03 * total_updates))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)
    print(f"[info] total_updates={total_updates}, warmup_steps={warmup_steps}")

    # wandb init
    run = None
    if use_wandb and WANDB_AVAILABLE:
        run = wandb.init(project="slm-lora", config={
            "pruned_model": pruned_model_path,
            "data": data_path,
            "r": r,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_length": max_length,
            "lr": lr,
            "epochs": epochs,
        })
        wandb.watch(model, log="all", log_freq=100)

    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps_in_epoch = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        optimizer.zero_grad()

        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * grad_accum_steps
            steps_in_epoch += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # log to tqdm & wandb
                avg_loss_so_far = epoch_loss / (steps_in_epoch if steps_in_epoch > 0 else 1)
                elapsed = time.time() - start_time
                pbar.set_postfix({"avg_loss": f"{avg_loss_so_far:.4f}", "step": global_step, "elapsed_s": int(elapsed)})

                if run is not None:
                    wandb.log({"train/avg_loss": avg_loss_so_far, "train/step": global_step, "train/elapsed_s": elapsed}, step=global_step)

        # epoch end
        avg_loss = epoch_loss / max(1, steps_in_epoch)
        eval_loss = evaluate_loss(model, eval_loader, device, use_amp=use_amp)
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} eval_loss={eval_loss:.4f} perplexity={perplexity:.2f}")

        if run is not None:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": avg_loss,
                "eval/loss": eval_loss,
                "eval/perplexity": perplexity,
            }, step=global_step)

        # save adapter
        if save_every_epoch:
            save_dir = os.path.join(out_dir, f"epoch-{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[info] saved LoRA adapter to {save_dir}")
            if run is not None:
                wandb.save(os.path.join(save_dir, "*"))

    # final save
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[done] saved final LoRA adapter to {out_dir}")

    if run is not None:
        run.finish()

    return out_dir


# -----------------------
# CLI / default hardcoded paths
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/tomato.json", help="tomato.json path")
    parser.add_argument("--pruned_model", type=str, default="./models/gemma_pruned", help="path to pruned model directory")
    parser.add_argument("--out_dir", type=str, default="./lora_adapter", help="output dir for adapter")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true", help="use fp16 on CUDA (ignored for MPS)")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging (optional)")
    args = parser.parse_args()

    # ensure paths
    Path(args.data).expanduser().resolve()
    Path(args.pruned_model).expanduser().resolve()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.wandb and not WANDB_AVAILABLE:
        print("[warn] wandb not installed or importable. Install with `pip install wandb` to enable.")
        args.wandb = False

    train(
        data_path=args.data,
        pruned_model_path=args.pruned_model,
        out_dir=args.out_dir,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_length=args.max_length,
        lr=args.lr,
        fp16=args.fp16,
        use_wandb=args.wandb,
    )