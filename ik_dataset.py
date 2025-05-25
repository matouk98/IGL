#!/usr/bin/env python
# coding: utf-8

import json
import os
import random
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === 配置模型（clarification 和 feedback 都用同一个） ===
model_id = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).cuda()

tokenizer = AutoTokenizer.from_pretrained(model_id, add_special_tokens=False)

# === 生成函数 ===
def generate(*, messages, model, tokenizer, num_return_sequences=1, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_len = inputs.shape[1]
    output_ids = model.generate(
        inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

# === Prompt 工具 ===
def simulator_prompt(mood):
    return f"You are a user simulator. A user has been presented a Question and an Answer. Simulate the user's next statement. The user is {mood} with the Answer to the Question."

def encapsulate_qa(question, answer):
    return f"Question:\n{question}\n\nAnswer:\n{answer}\n"

# === 加载输入数据（来自多个模型） ===
json_files = {
    0: "Qwen2.5-0.5B-Instruct_question_answer_1.json",
    1: "Qwen2.5-0.5B-Instruct_question_answer_2.json",
    2: "Qwen2.5-0.5B-Instruct_question_answer_3.json",
    3: "Qwen2.5-0.5B-Instruct_question_answer_4.json",
    4: "Qwen2.5-14B-Instruct_question_answer.json"
}
data = {idx: json.load(open(path, 'r', encoding='utf-8')) for idx, path in json_files.items()}

# === 数据处理参数 ===
num_samples = 12000
final_dataset = []

# === 主循环处理 ===
for idx in tqdm(range(num_samples)):
    clarifications_1, clarifications_2 = [], []
    for i in range(5):
        clars = data[i][idx].get("clarifications", [])
        clarifications_1.append(clars[0] if len(clars) > 0 else "")
        clarifications_2.append(clars[1] if len(clars) > 1 else "")
    answers = [data[i][idx]['answer'] for i in range(5)]

    chosen_c1_idx = random.choice(range(5))
    chosen_c2_idx = random.choice(range(5))
    chosen_answer_idx = random.choice(range(5))

    original_question = data[0][idx].get('original_prompt', "")
    selected_answer = answers[chosen_answer_idx]
    label = 1 if chosen_answer_idx == 4 else 0

    # clarification answer 1
    c1 = clarifications_1[chosen_c1_idx]
    if c1.strip():
        c1_msg = [{ "role": "user", "content": f"Answer the following question concisely: {c1}" }]
        c1_out = generate(messages=c1_msg, model=model, tokenizer=tokenizer)[0].strip()
    else:
        c1_out = ""

    # clarification answer 2
    c2 = clarifications_2[chosen_c2_idx]
    if c2.strip():
        c2_msg = [{ "role": "user", "content": f"Answer the following question concisely: {c2}" }]
        c2_out = generate(messages=c2_msg, model=model, tokenizer=tokenizer)[0].strip()
    else:
        c2_out = ""

    # feedback generation
    mood = "satisfied" if label == 1 else "unsatisfied"
    fb_msg = [
        { "role": "system", "content": simulator_prompt(mood) },
        { "role": "user", "content": encapsulate_qa(original_question, selected_answer) }
    ]
    feedback_out = generate(messages=fb_msg, model=model, tokenizer=tokenizer)[0].strip()

    # === 汇总结果 ===
    final_dataset.append({
        "original_question": original_question,
        "selected_clarification_1": c1,
        "selected_clarification_2": c2,
        "clarification_1_answer": c1_out,
        "clarification_2_answer": c2_out,
        "selected_model_answer": selected_answer,
        "all_model_answers": answers,
        "answer_source": chosen_answer_idx,
        "label": label,
        "generated_feedback": feedback_out
    })

# === 保存输出 ===
output_path = "ik_dataset_hf.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Saved {len(final_dataset)} samples to {output_path}")
