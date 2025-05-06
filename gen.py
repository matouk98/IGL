from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re

model_name = "Qwen/Qwen2.5-14B-Instruct"
ds = load_dataset("RLHFlow/prompt-collection-v0.1", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def truncate_prompt(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    else:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

cnt = 0
max_tokens = 512
prompts = []
for sample in ds:
    text = sample['context_messages'][0]['content']
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        cnt += 1
        prompts.append(text)
    if cnt == 20000:
        break

output_file = "{}_question_answer.json".format(model_name.split('/')[-1])
llm = LLM(model=model_name, max_model_len=2048)

sampling_params_clarification = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    seed=1
)

sampling_params_answer = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    seed=1
)

def make_clarification_prompt(original_prompt):
    return (
        f"You are about to answer the following question:\n\n"
        f"\"{original_prompt}\"\n\n"
        f"Before answering, list 2 clarifying questions that would help you answer better."
        f" Output them clearly numbered (1, 2)."
    )

def make_answer_prompt(original_prompt):
    return (
        f"{original_prompt}\n"
        f"Please provide a helpful and informative answer."
    )

def extract_clarifications(text):
    matches = re.findall(r'\d+\.\s*(.+)', text.strip())
    return [m.strip() for m in matches[:2] if m.strip() != ""]

# First generate clarifications
clarification_prompts = [make_clarification_prompt(p) for p in prompts]
clarification_outputs = llm.generate(clarification_prompts, sampling_params_clarification)
clarifications_list = []
for output in clarification_outputs:
    generated_text = output.outputs[0].text.strip()
    clarifications = extract_clarifications(generated_text)
    clarifications_list.append(clarifications)

# Then generate answers
answer_prompts = [make_answer_prompt(p) for p in prompts]
answer_outputs = llm.generate(answer_prompts, sampling_params_answer)
answers = [output.outputs[0].text.strip() for output in answer_outputs]

# Combine results and track stats
results = []
missing_clarification_count = 0
missing_answer_count = 0
for i in range(len(prompts)):
    clars = clarifications_list[i]
    ans = answers[i]
    if len(clars) < 2:
        missing_clarification_count += 1
    if ans.strip() == "":
        missing_answer_count += 1

    results.append({
        "original_prompt": prompts[i],
        "clarifications": clars,
        "answer": ans
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Saved question-answer pairs to", output_file)
print(f"❗ Missing clarifications (less than 2 or contain empty): {missing_clarification_count}")
print(f"❗ Missing answers (empty): {missing_answer_count}")
