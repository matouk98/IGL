import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import matplotlib.pyplot as plt
import argparse

# === CLI and Config ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--ik_model_dir", type=str, default="ik-lr1e-4/epoch-1/ik")
    parser.add_argument("--policy_lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="th98")
    parser.add_argument("--reward_thresh", type=float, default=0.98)
    parser.add_argument("--json_patterns", type=str, nargs=5, default=[
        "Qwen2.5-0.5B-Instruct_question_answer_1.json",
        "Qwen2.5-0.5B-Instruct_question_answer_2.json",
        "Qwen2.5-0.5B-Instruct_question_answer_3.json",
        "Qwen2.5-0.5B-Instruct_question_answer_4.json",
        "Qwen2.5-14B-Instruct_question_answer.json"
    ])
    return parser.parse_args()

# === Main ===
def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.output_dir, exist_ok=True)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.batch_size

    raw_data = {}
    for idx, path in enumerate(args.json_patterns):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data[idx] = json.load(f)

    # Initialize DM model
    base_dm = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1
    )
    dm_lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj']
    )
    dm_model = get_peft_model(base_dm, dm_lora_config)

    # Load IK model separately (only for inference, no accelerator wrapping)
    base_ik = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", num_labels=1
    )
    ik_model = PeftModel.from_pretrained(base_ik, args.ik_model_dir)
    ik_model.eval()
    ik_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    fb_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct").to(device)
    fb_model.eval()
    def generate_feedback(prompt):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = fb_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated = outputs[0][inputs.input_ids.shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        return decoded.strip()

    dm_model.train()
    optimizer = torch.optim.AdamW(dm_model.parameters(), lr=args.policy_lr)
    dm_model, optimizer = accelerator.prepare(dm_model, optimizer)

    def batch_infer(prompts, model_obj, requires_grad=True):
        all_logits = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i+args.batch_size]
            enc = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True).to(device)
            if requires_grad:
                out = model_obj(**enc).logits.squeeze(-1)
            else:
                with torch.no_grad():
                    out = model_obj(**enc).logits.squeeze(-1)
            all_logits.append(out)
        return torch.cat(all_logits, dim=0)

    start_idx = 10000
    num_samples = len(raw_data[4]) - start_idx
    true_rewards, est_rewards = [], []
    est_1_correct, est_1_total = 0, 0
    est_0_correct, est_0_total = 0, 0
    print("Starting DM training (softmax policy, no clarifications)...")
    for i in tqdm(range(min(args.num_steps, num_samples))):
        question = raw_data[4][i + start_idx]['original_prompt']
        answers = [raw_data[j][i + start_idx]['answer'] for j in range(5)]
        label = 4

        dm_prompts = [
            f"You are a decision making agent. Given a User’s Question and an Answer, determine if the decision should be this answer. Respond with Yes or No only.\n\n"
            f"Question: {question}\nAnswer: {a}\nRespond with Yes or No only."
            for a in answers
        ]
        logits = batch_infer(dm_prompts, dm_model, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        action_tensor = torch.multinomial(probs, 1)  # shape: [1]
        action = action_tensor.item()  # for later label matching

        fb_prompt = (
            f"You are a user simulator. A user has been presented a Question and an Answer.\n"
            f"Question: {question}\n"
            f"Answer: {answers[action]}\n"
            f"Simulate the user’s feedback. The user is {'satisfied' if action == label else 'unsatisfied'} with the Answer to the Question.\n\n"
        )
        feedback = generate_feedback(fb_prompt)

        ik_prompts = [
            "You are a conversation evaluating agent. "
            "Given a User’s Question, an Answer, and the User’s Feedback: determine if the User’s Feedback is consistent with Answer. "
            "Respond with Yes or No only.\n\n"
            f"User’s Question: {question}\n"
            f"Answer: {a.strip()}\n"
            f"User’s Feedback: {feedback}\n"
            "Respond with Yes or No only."
            for a in answers
        ]
        ik_logits = batch_infer(ik_prompts, ik_model, requires_grad=False)
        ik_probs = torch.softmax(ik_logits, dim=-1)
        r_est = int(ik_probs[action].item() >= args.reward_thresh)
        # r_est = int(action == label)
        r_true = int(action == label)
        
        logp = torch.log(probs[action_tensor] + 1e-8).squeeze()
        r_est_tensor = torch.tensor(r_est, dtype=logp.dtype, device=logp.device)
        loss = - r_est_tensor * logp
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        true_rewards.append(r_true)
        est_rewards.append(r_est)

        if r_true == 1:
            est_1_total += 1
            est_1_correct += ik_probs[action].item()
        else:
            est_0_total += 1
            est_0_correct += ik_probs[action].item()

        if (i + 1) % 100 == 0:
            avg_true = np.mean(true_rewards)
            avg_est = np.mean(est_rewards)
            acc_1 = est_1_correct / est_1_total if est_1_total > 0 else 0.0
            acc_0 = est_0_correct / est_0_total if est_0_total > 0 else 0.0
            accelerator.print(f"Iteration {i + 1}: Avg True Reward = {avg_true:.4f}, Avg Estimated Reward = {avg_est:.4f}")
            accelerator.print(f"\tPositive Prob: {acc_1:.4f} ({est_1_correct}/{est_1_total})")
            accelerator.print(f"\tNegative Prob: {acc_0:.4f} ({est_0_correct}/{est_0_total})")

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, 'trained_dm_adapter')
        os.makedirs(save_path, exist_ok=True)
        dm_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        accelerator.print(f"Saved trained DM adapter to {save_path}")

    steps = np.arange(len(true_rewards))
    plt.plot(steps, np.cumsum(true_rewards)/(steps+1), label='True Reward')
    plt.plot(steps, np.cumsum(est_rewards)/(steps+1), label='Estimated Reward')
    plt.xlabel('Step'); plt.ylabel('Avg Reward'); plt.title('Softmax Policy Learning Curve')
    plt.legend(); plt.savefig(os.path.join(args.output_dir, 'dm_softmax_curve.png'))

if __name__ == '__main__':
    main()
