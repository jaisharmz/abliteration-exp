import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM, EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTTrainer

# Note: This script is configured for Llama 3.2 1B.

# 1. Model and Tokenizer Configuration
# Define the base model and the new model ID for saving
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NEW_MODEL_ID = "llama-3.2-1b-finetuned-humaneval"

# Set up quantization config for 4-bit training to save memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Set pad token for consistency
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 2. Dataset Preparation and Formatting
print("Loading and formatting the dataset...")
# Load the dataset from Hugging Face Hub
ds = load_dataset("code-rag-bench/humaneval", split="train")

# We need to format the dataset into a conversational format for Llama 3
def formatting_prompts_func(examples):
    formatted_texts = []
    # Loop through each example in the batch
    for i in range(len(examples['prompt'])):
        # Construct the conversation history in the Llama 3 chat format
        conversation = [
            {"role": "user", "content": examples['prompt'][i]},
            {"role": "assistant", "content": examples['canonical_solution'][i]}
        ]
        # Apply the chat template to format it correctly for the model's tokenizer
        formatted_texts.append(tokenizer.apply_chat_template(conversation, tokenize=False))
    return {"text": formatted_texts}

# Map the formatting function over the dataset
formatted_ds = ds.map(formatting_prompts_func, batched=True, remove_columns=list(ds.features.keys()))

# Split the dataset into training and validation sets
split_ds = formatted_ds.train_test_split(test_size=0.1, seed=42)
train_dataset = split_ds['train']
eval_dataset = split_ds['test']


# 3. PEFT (Parameter-Efficient Fine-Tuning) Configuration
# Set up LoRA configuration for Llama 3
lora_config = LoraConfig(
    r=8, # LoRA rank
    lora_alpha=16, # Alpha parameter
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # All attention and feed-forward layers
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 4. Training Arguments
# Using arguments compatible with older library versions. Early stopping is removed to prevent errors.
training_args = TrainingArguments(
    output_dir="./results-llama-3.2-1b", # Directory to save the fine-tuned model
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5, # Set a higher max number of epochs
    logging_steps=10,
    save_steps=10,
    eval_steps=10,
    do_eval=True, # Use this argument for older library versions
    save_total_limit=2,
    # Conflicting arguments for early stopping have been removed for compatibility.
    fp16=False,
    bf16=True,
    push_to_hub=False,
)

# 5. Fine-tuning with SFTTrainer
print("Starting the fine-tuning process...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Provide the validation dataset
    peft_config=lora_config,
    # Callbacks removed to prevent compatibility errors
)

# Start training!
trainer.train()

# Save the model locally after training is complete
print(f"Training complete. Saving model and tokenizer to {training_args.output_dir}...")
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print("Finetuning script finished. The model is available in the local directory.")

