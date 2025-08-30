import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
import torch.nn.functional as F
import einops
from jaxtyping import Float
from torch import Tensor

# Note: This script should be run AFTER finetune_model_code_generation.py

# 1. Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./results-llama-3.2-1b"

# Set up quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Load Base Model and Tokenizer
print("Loading base model and tokenizer...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 3. Dataset Preparation
print("Loading and formatting datasets for analysis...")
coding_ds = load_dataset("code-rag-bench/humaneval", split="train")
anchor_ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")

def formatting_prompts_func(examples, text_field='prompt'):
    formatted_texts = []
    for i in range(len(examples[text_field])):
        conversation = [{"role": "user", "content": examples[text_field][i]}]
        formatted_texts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
    return {"text": formatted_texts}

formatted_coding_ds = coding_ds.map(lambda x: formatting_prompts_func(x, text_field='prompt'), batched=True, remove_columns=list(coding_ds.features.keys()))
formatted_anchor_ds = anchor_ds.map(lambda x: formatting_prompts_func(x, text_field='prompt'), batched=True, remove_columns=list(anchor_ds.features.keys()))

# 4. Activation and Weight Analysis
def get_activations(model, dataset, max_samples=100):
    """
    Gets the mean input and output activations for specified layers.
    """
    input_activations = {}
    output_activations = {}
    
    temp_inputs = {}
    temp_outputs = {}

    def hook_fn(module, input, output, name):
        # Hook for output activations
        if name not in temp_outputs:
            temp_outputs[name] = []
        # Average over the sequence length dimension (dim=1)
        temp_outputs[name].append(output[0].mean(dim=1).squeeze(0).detach().cpu())
        
        # Hook for input activations
        if name not in temp_inputs:
            temp_inputs[name] = []
        # Input is a tuple, we take the first element and average over sequence length
        temp_inputs[name].append(input[0].mean(dim=1).squeeze(0).detach().cpu())

    hooks = []
    # Target LoRA-tuned layers for activation hooking
    for name, module in model.named_modules():
        if "o_proj" in name or "down_proj" in name:
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

    subset = dataset.shuffle(seed=42).select(range(min(len(dataset), max_samples)))
    for i in range(len(subset)):
        inputs = tokenizer(subset[i]['text'], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

    for hook in hooks:
        hook.remove()

    # Calculate the mean activation across all collected samples for each layer
    for name, act_list in temp_outputs.items():
        if act_list:
            output_activations[name] = torch.stack(act_list).mean(dim=0)
    
    for name, act_list in temp_inputs.items():
        if act_list:
            input_activations[name] = torch.stack(act_list).mean(dim=0)
            
    return input_activations, output_activations

# Calculate delta_W from the LoRA adapters
print("Calculating delta_W from LoRA matrices...")
delta_W = {}
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

for name, module in ft_model.named_modules():
    if isinstance(module, LoraLayer) and hasattr(module, 'active_adapter'):
        # Handle case where active_adapter can be a list
        adapter_name = module.active_adapter
        if isinstance(adapter_name, list):
            adapter_name = adapter_name[0]  # Take the first adapter name

        if adapter_name in module.lora_A:
            lora_A = module.lora_A[adapter_name].weight
            lora_B = module.lora_B[adapter_name].weight
            delta_W[name] = (lora_B @ lora_A).detach().cpu()


print("Calculating activations on the base model...")
base_coding_inputs, base_coding_outputs = get_activations(base_model, formatted_coding_ds)
_, base_anchor_outputs = get_activations(base_model, formatted_anchor_ds)

# Calculate the coding vector from the base model's output activations
coding_vector = {layer: base_coding_outputs[layer] - base_anchor_outputs[layer] for layer in base_coding_outputs}
print("Coding vector calculated.")

# 5. Compare the effect of delta_W to the coding vector
similarities = {}
for layer_name, d_W in delta_W.items():
    # Reconcile layer names
    clean_layer_name = layer_name.replace("base_model.model.", "")
    
    if clean_layer_name in coding_vector and clean_layer_name in base_coding_inputs:
        # Get the average input activation for this layer
        avg_input_activation = base_coding_inputs[clean_layer_name]
        
        # Calculate the change in activation induced by delta_W
        # Shape: (out_features, in_features) @ (in_features) -> (out_features)
        induced_delta_activation = d_W @ avg_input_activation
        
        # Get the target direction (the coding vector)
        target_delta_activation = coding_vector[clean_layer_name]
        
        # Compare the induced change with the target change
        similarity = F.cosine_similarity(induced_delta_activation, target_delta_activation, dim=0)
        similarities[clean_layer_name] = similarity.item()

print("\n--- Analysis Results ---")
print("Cosine Similarity between Induced Activation Change (from delta_W) and the Coding Vector:")
for layer, sim in similarities.items():
    print(f"{layer}: {sim:.4f}")

# Included from the article for reference/future use
def get_orthogonalized_matrix(matrix: Float[Tensor, "... d_model"], vec: Float[Tensor, "d_model"]) -> Float[Tensor, "... d_model"]:
    """Projects a matrix to be orthogonal to a vector."""
    proj = (
        einops.einsum(
            matrix, vec.view(-1, 1), "... d_model, d_model single -> ... single"
        )
        * vec
    )
    return matrix - proj

print("\nAnalysis script finished.")

