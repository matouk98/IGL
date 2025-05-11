import os
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, PeftConfig
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epoch_num", type=int, default=1)
parser.add_argument("--model_name", type=str, default="lr1e-4")
args = parser.parse_args()

iter_num = args.epoch_num
model_name = args.model_name
adapter_path = "ik-lr1e-4/epoch-1/ik"
llm_gen_path = "Qwen/Qwen2.5-7B-Instruct"
json_files = {
    0: "Qwen2.5-0.5B-Instruct_question_answer_1.json",
    1: "Qwen2.5-0.5B-Instruct_question_answer_2.json",
    2: "Qwen2.5-0.5B-Instruct_question_answer_3.json",
    3: "Qwen2.5-0.5B-Instruct_question_answer_4.json",
    4: "Qwen2.5-14B-Instruct_question_answer.json"
}
num_samples = 1000

# === load all answers ===
raw_data = {}
for idx, path in json_files.items():
    with open(path, "r", encoding="utf-8") as f:
        raw_data[idx] = json.load(f)[10000:10000+num_samples]

# === load classification model ===
model_cls_path = "ik-lr1e-4/epoch-1"
tokenizer = AutoTokenizer.from_pretrained(model_cls_path)
tokenizer.pad_token = tokenizer.eos_token

base_ik = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", num_labels=1
    )
model = PeftModel.from_pretrained(base_ik, adapter_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 初始化 LLM（用于生成反馈）===
llm = LLM(model=llm_gen_path, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

# === 构造反馈 prompt（对0-3模型的随机回答）===
feedback_prompts = []
questions = []
answer_sources = []

for i in range(num_samples):
    question = raw_data[0][i]["original_prompt"]
    chosen_model = random.choice([0, 1, 2, 3])
    answer = raw_data[chosen_model][i]["answer"]
    prompt = (
        f"You are a user simulator. A user has been presented a Question and an Answer. "
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Simulate the user’s feedback. The user is unsatisfied with the Answer to the Question.\n\n"
    )
    feedback_prompts.append(prompt)
    questions.append(question)
    answer_sources.append(chosen_model)

# === 生成反馈 ===
print("🔄 Generating feedback...")
feedback_outputs = llm.generate(feedback_prompts, sampling_params)
feedback_texts = [out.outputs[0].text.strip() for out in feedback_outputs]

# === 构造测试样本（每个问题生成5条：对应0-4的answer）===
all_logits = [[] for _ in range(num_samples)]

def tokenize_batch(prompts):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    return input_ids, attention_mask

print("🧠 Running binary predictions for all 5 candidate answers...")

for model_idx in range(5):
    prompts = []
    for i in range(num_samples):
        question = questions[i]
        answer = raw_data[model_idx][i]["answer"]
        feedback = feedback_texts[i]
        prompt = (
            "You are a conversation evaluating agent. "
            "Given a User’s Question, an Answer, and the User’s Feedback: determine if the User’s Feedback is consistent with Answer. "
            "Respond with Yes or No only.\n\n"
            f"User’s Question: {question}\n"
            f"Answer: {answer.strip()}\n"
            f"User’s Feedback: {feedback}\n"
            "Respond with Yes or No only."
        )
        prompts.append(prompt)

    # 批量推理
    batch_logits = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        input_ids, attention_mask = tokenize_batch(batch_prompts)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            batch_logits.extend(logits.cpu().tolist())

    # 保存当前模型的logits
    for i in range(num_samples):
        all_logits[i].append(batch_logits[i])

# === 每个样本对5个logit做softmax，提取被选中模型的概率 ===
probs_selected = [
    torch.softmax(torch.tensor(logits), dim=-1)[answer_sources[i]].item()
    for i, logits in enumerate(all_logits)
]
avg_prob = sum(probs_selected) / len(probs_selected)
print(f"\n🔍 Avg softmax probability for randomly selected 0.5B answers: {avg_prob:.4f}")

# === 可视化 ===
os.makedirs('./neg_figs', exist_ok=True)
plt.hist(probs_selected, bins=50, alpha=0.75)
plt.title("Softmax Probability for Random 0.5B Answers (Binary Model)")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(f"./neg_figs/predicted_prob_softmax_0.5B_{model_name}_{iter_num}.png")
print(f"📊 Saved histogram to predicted_prob_softmax_0.5B_{model_name}_{iter_num}.png")
