import json
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm

# 1. 配置模型路径和文件路径
model_name_7b = "Qwen/Qwen2.5-7B-Instruct"
json_files = {
    0: "Qwen2.5-0.5B-Instruct_question_answer_1.json",
    1: "Qwen2.5-0.5B-Instruct_question_answer_2.json",
    2: "Qwen2.5-0.5B-Instruct_question_answer_3.json",
    3: "Qwen2.5-0.5B-Instruct_question_answer_4.json",
    4: "Qwen2.5-14B-Instruct_question_answer.json"
}

# 2. 加载所有模型的clarification/answer数据
data = {idx: json.load(open(path, 'r', encoding='utf-8')) for idx, path in json_files.items()}

# 3. 初始化7B模型用于推理clarification answers和feedback
llm = LLM(model=model_name_7b, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

# 4. 构造样本（处理前12000条问题）
num_samples = 12000
final_dataset = []

# 5. 批量处理 Clarifications (Stage 1 & 2)
clarification_prompts = []
clarification_meta = []

for idx in range(num_samples):
    clarifications_1, clarifications_2 = [], []
    for i in range(5):
        clars = data[i][idx].get("clarifications", [])
        clarifications_1.append(clars[0] if len(clars) > 0 else "")
        clarifications_2.append(clars[1] if len(clars) > 1 else "")
    answers = [data[i][idx]['answer'] if 'answer' in data[i][idx] else "" for i in range(5)]
    assert isinstance(answers, list) and len(answers) == 5 and all(isinstance(a, str) for a in answers)

    chosen_c1_idx = random.choice([0, 1, 2, 3, 4])
    chosen_c2_idx = random.choice([0, 1, 2, 3, 4])
    chosen_answer_idx = random.choice([0, 1, 2, 3, 4])

    clarification_prompts.append(f"Answer the following question concisely: {clarifications_1[chosen_c1_idx]}")
    clarification_prompts.append(f"Answer the following question concisely: {clarifications_2[chosen_c2_idx]}")

    clarification_meta.append({
        "original_question": data[0][idx].get('original_prompt', ""),
        "selected_clarification_1": clarifications_1[chosen_c1_idx],
        "selected_clarification_2": clarifications_2[chosen_c2_idx],
        "selected_model_answer": answers[chosen_answer_idx],
        "all_model_answers": answers,
        "answer_source": chosen_answer_idx,
        "label": 1 if chosen_answer_idx == 4 else 0
    })

# 6. 批量生成clarification answers
print("\nGenerating clarification answers...")
clarification_outputs = llm.generate(clarification_prompts, sampling_params)
num_empty_c1 = sum(1 for m in clarification_meta if not m["selected_clarification_1"].strip())
num_empty_c2 = sum(1 for m in clarification_meta if not m["selected_clarification_2"].strip())
print(f"Clarification 1 is empty in {num_empty_c1} samples.")
print(f"Clarification 2 is empty in {num_empty_c2} samples.")

clarification_1_answers = clarification_outputs[0::2]
clarification_2_answers = clarification_outputs[1::2]

# 7. 批量生成positive/negative feedback prompts
feedback_prompts = []
# for meta, c1_out, c2_out in zip(clarification_meta, clarification_1_answers, clarification_2_answers):
#     feedback_type = "positive" if meta["label"] == 1 else "negative"
#     prompt = (
#         f"Given:\n"
#         f"- Original Question: {meta['original_question']}\n"
#         f"- Clarification Question 1: {meta['selected_clarification_1']}\n"
#         f"- Clarification Answer 1: {c1_out.outputs[0].text.strip() if meta['selected_clarification_1'].strip() else ''}\n"
#         f"- Clarification Question 2: {meta['selected_clarification_2']}\n"
#         f"- Clarification Answer 2: {c2_out.outputs[0].text.strip() if meta['selected_clarification_2'].strip() else ''}\n"
#         f"- Final Answer: {meta['selected_model_answer']}\n\n"
#         f"Please write a short {feedback_type} feedback evaluating the quality of the final answer, based on the clarifications."
#     )
#     feedback_prompts.append(prompt)
for meta in clarification_meta:
    mood = "satisfied" if meta["label"] == 1 else "unsatisfied"
    prompt = (
    f"You are a user simulator. A user has been presented a Question and an Answer. "
    f"Question: {meta['original_question']}\n"
    f"Answer: {meta['selected_model_answer']}"
    f"Simulate the user’s feedback. The user is {mood} with the Answer to the Question.\n\n"
)
    feedback_prompts.append(prompt)

# 8. 批量生成feedback
print("\nGenerating feedback...")
feedback_outputs = llm.generate(feedback_prompts, sampling_params)

# 9. 整理最终数据
for idx, (meta, c1_out, c2_out, feedback_out) in enumerate(zip(clarification_meta, clarification_1_answers, clarification_2_answers, feedback_outputs)):
    final_dataset.append({
        "original_question": meta['original_question'],
        "selected_clarification_1": meta['selected_clarification_1'],
        "selected_clarification_2": meta['selected_clarification_2'],
        "clarification_1_answer": c1_out.outputs[0].text.strip() if meta['selected_clarification_1'].strip() else '',
        "clarification_2_answer": c2_out.outputs[0].text.strip() if meta['selected_clarification_2'].strip() else '',
        "selected_model_answer": meta['selected_model_answer'],
        "all_model_answers": meta['all_model_answers'],
        "answer_source": meta['answer_source'],
        "label": meta['label'],
        "generated_feedback": feedback_out.outputs[0].text.strip()
    })

# 10. 保存成新文件
output_file = "ik_dataset.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

print(f"\n✅ All done! Saved {len(final_dataset)} samples to {output_file}")
