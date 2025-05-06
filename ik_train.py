import os
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_train_samples", type=int, default=10000)
parser.add_argument("--val_samples", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gradient_acc", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="ik-lr4")
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(args.output_dir, exist_ok=True)

# === Load and split dataset ===
raw_dataset = load_dataset("json", data_files="ik_dataset.json", split="train")
train_dataset = raw_dataset.select(range(0, args.max_train_samples))
val_dataset = raw_dataset.select(range(args.max_train_samples, args.max_train_samples + args.val_samples))

# === Convert each example to 5-way grouped sample ===
def build_grouped_samples(example):
    samples = []
    for i, answer in enumerate(example["all_model_answers"]):
        text = (
            f"You are a conversation evaluating agent. "
            f"Given a User’s Question, an Answer, and the User’s Feedback: determine if the User’s Feedback is consistent with Answer. "
            f"Respond with Yes or No only.\n\n"
            f"User’s Question: {example['original_question']}\n"
            f"Answer: {answer.strip()}\n"
            f"User’s Feedback: {example['generated_feedback']}\n"
            f"Respond with Yes or No only."
        )
        samples.append((text, i))
    random.shuffle(samples)
    texts = [s[0] for s in samples]
    original_label_idx = int(example["answer_source"])
    new_label = [s[1] for s in samples].index(original_label_idx)
    return {"texts": texts, "text_label": new_label, "original_label": original_label_idx}

train_dataset = train_dataset.map(build_grouped_samples).remove_columns(raw_dataset.column_names)
val_dataset = val_dataset.map(build_grouped_samples).remove_columns(raw_dataset.column_names)

# === Model and tokenizer ===
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.cuda()
model.train()

# === Collate function ===
def collate_fn(batch):
    all_texts = sum([item["texts"] for item in batch], [])
    encodings = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encodings["input_ids"].view(len(batch), 5, -1)
    attention_mask = encodings["attention_mask"].view(len(batch), 5, -1)
    labels = torch.tensor([item["text_label"] for item in batch])
    orig_labels = torch.tensor([item["original_label"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "original_label": orig_labels
    }

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# === Optimizer ===
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# === Training ===
print("Starting training...")
for epoch in range(args.num_epochs):
    model.train()
    total_loss = 0
    correct = 0

    for step, batch in enumerate(tqdm(train_loader)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        logits_list = []
        for i in range(5):
            logits = model(input_ids=input_ids[:, i], attention_mask=attention_mask[:, i]).logits.squeeze(-1)
            logits_list.append(logits)
        logits_stacked = torch.stack(logits_list, dim=1)

        loss = F.cross_entropy(logits_stacked, labels) / args.gradient_acc
        loss.backward()

        if (step + 1) % args.gradient_acc == 0 or (step + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_acc
        preds = torch.argmax(logits_stacked, dim=1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_dataset)
    train_loss = total_loss / len(train_dataset)

    # === Validation ===
    model.eval()
    val_correct_4 = 0
    val_total_4 = 0
    val_correct_others = 0
    val_total_others = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            original = batch["original_label"].cuda()

            logits_list = []
            for i in range(5):
                logits = model(input_ids=input_ids[:, i], attention_mask=attention_mask[:, i]).logits.squeeze(-1)
                logits_list.append(logits)
            logits_stacked = torch.stack(logits_list, dim=1)

            preds = torch.argmax(logits_stacked, dim=1)
            for i in range(len(preds)):
                if original[i].item() == 4:
                    val_total_4 += 1
                    val_correct_4 += int(preds[i].item() == labels[i].item())
                else:
                    val_total_others += 1
                    val_correct_others += int(preds[i].item() == labels[i].item())

    val_acc_4 = val_correct_4 / val_total_4 if val_total_4 else 0
    val_acc_others = val_correct_others / val_total_others if val_total_others else 0
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Pos Acc: {val_acc_4:.4f}, Neg Acc: {val_acc_others:.4f}")

    # Save checkpoint
    save_dir = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
