from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig

# === Config ===
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
save_dir = "./saved_adapter"
adapter_name = "ik"

# === Step 1: Create base model with adapter and save it ===
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

ik_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, ik_config, adapter_name=adapter_name)
print(f"✅ Created PEFT model with adapter '{adapter_name}'.")

# Save adapter to disk (it will go under saved_adapter/ik/)
model.save_pretrained(save_dir, adapter_name=adapter_name)

# === Step 2: Reload base model and load saved adapter ===
new_base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

# Use dummy adapter to wrap base model so we can call load_adapter()
dm_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(new_base_model, dm_config, adapter_name="dm")
print("✅ DM PEFT model created.")

# Load the saved "ik" adapter
adapter_path = f"{save_dir}/{adapter_name}"
model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
print(f"✅ Adapter '{adapter_name}' loaded from: {adapter_path}")

# Activate the "ik" adapter
model.active_adapter = adapter_name
print(f"✅ Active adapter set to: {model.active_adapter}")

# === Step 3: Verify adapter state dict ===
try:
    state_dict = model.get_adapter_state_dict(adapter_name)
    print(f"✅ Adapter '{adapter_name}' state dict keys (preview): {list(state_dict.keys())[:5]}")
except Exception as e:
    print(f"❌ Failed to get adapter state dict: {e}")
