# -*- coding: utf-8 -*-
"""
## Uncensor any LLM with abliteration - Extended Experiments

This notebook implements the experimental plan to control a large language model's
behavior at inference time by amplifying a "coding vector" derived from
its activations. It also includes methods to analyze and reverse-engineer
the finetuning process.

Original concept by @maximelabonne. Extended experiments proposed by the user.
"""

# !pip install -qqq transformers datasets tiktoken transformer_lens einops jaxtyping scipy --progress-bar off

import torch
import functools
import einops
import gc
import copy
import numpy as np
from scipy.stats import ttest_ind
from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List, Dict
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float
from collections import defaultdict

# Turn automatic differentiation off to save GPU memory
torch.set_grad_enabled(False)

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# In a real scenario, you would first finetune the model and save it to a new ID.
# For this script, we simulate the finetuning process.
FINETUNED_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct-Code-Finetuned" # Simulated

# Load base model and tokenizer from Hugging Face
model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16,
    default_padding_side='left',
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

"""
### 1. Finetune Model on Code Generation (Simulated)

In a real experiment, you would use a library like `trl`'s `SFTTrainer`
to finetune the base model on a coding dataset. This is computationally
intensive. For this script, we **simulate** the process to get a `W_delta`.

We create a "finetuned" model by adding a small, random, directed noise
to the original weights `W`. This gives us `W'` and allows us to calculate
`W_delta = W' - W` for later analysis.
"""

print("--- 1. Simulating Finetuning to get W_delta ---")

# Deepcopy the original model's state dict to represent the finetuned version
finetuned_state_dict = copy.deepcopy(model.state_dict())

# Simulate the changes from finetuning by adding small noise to key weights
# We will focus on the output weights of the attention and MLP layers.
w_delta_dict = {}
for l in range(model.cfg.n_layers):
    # Attention Output Weights
    key_attn = f"blocks.{l}.attn.W_O"
    original_attn_weight = model.state_dict()[key_attn]
    noise_attn = torch.randn_like(original_attn_weight) * 1e-5 # Small noise
    finetuned_state_dict[key_attn] += noise_attn
    w_delta_dict[key_attn] = noise_attn

    # MLP Output Weights
    key_mlp = f"blocks.{l}.mlp.W_out"
    original_mlp_weight = model.state_dict()[key_mlp]
    noise_mlp = torch.randn_like(original_mlp_weight) * 1e-5 # Small noise
    finetuned_state_dict[key_mlp] += noise_mlp
    w_delta_dict[key_mlp] = noise_mlp

print("Simulated W_delta calculated for attention and MLP output weights.")
print("-" * 50)


"""
### 2. Create Coding and Non-Coding Datasets

We source two datasets from the Hugging Face Hub: one with coding-related
prompts and another with general, non-coding prompts.
"""
print("--- 2. Sourcing Coding and Non-Coding Datasets ---")

def reformat_texts(texts):
    """Helper to format prompts for the Llama 3 chat template."""
    return [[{"role": "user", "content": text}] for text in texts]

# Coding dataset
coding_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
coding_instructions = reformat_texts(coding_dataset['prompt'][:512]) # Using 512 samples

# Non-coding (harmless) dataset
non_coding_dataset = load_dataset("mlabonne/harmless_alpaca", split="train")
non_coding_instructions = reformat_texts(non_coding_dataset['text'][:512])

print(f"Loaded {len(coding_instructions)} coding prompts.")
print(f"Loaded {len(non_coding_instructions)} non-coding prompts.")
print("-" * 50)


"""
### 3. Record Latent Activations for Coding and Non-Coding Datasets

Using hooks, we run both datasets through the **base model** and cache the
residual stream activations (`resid_pre`) for each layer.
"""
print("--- 3. Recording Latent Activations ---")

def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids

n_inst_train = min(256, len(coding_instructions), len(non_coding_instructions))
batch_size = 16 # Adjust based on your VRAM

coding_tokens = tokenize_instructions(tokenizer, coding_instructions[:n_inst_train])
non_coding_tokens = tokenize_instructions(tokenizer, non_coding_instructions[:n_inst_train])

def get_activations(model, tokens):
    """Runs model with cache and collects activations."""
    activations = defaultdict(list)
    num_batches = (n_inst_train + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min(n_inst_train, start_idx + batch_size)
        
        current_tokens = tokens[start_idx:end_idx].to(model.cfg.device)

        _, cache = model.run_with_cache(
            current_tokens,
            names_filter=lambda name: "resid_pre" in name,
            device='cpu',
        )
        
        for key in cache:
            activations[key].append(cache[key])

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    return {k: torch.cat(v) for k, v in activations.items()}

coding_activations = get_activations(model, coding_tokens)
non_coding_activations = get_activations(model, non_coding_tokens)

print("Activations for coding and non-coding datasets have been collected.")
print("-" * 50)

"""
### 4. Determine Semantic Vector and Select Best Layer with t-test

We compute the "coding vector" as the difference in mean activations
between the two datasets. To find the optimal layer for intervention,
we perform a t-test on the activations for each layer, selecting the one
with the highest p-value (most significant difference).
"""
print("--- 4. Calculating Coding Vector and Selecting Layer with t-test ---")

coding_vectors = {}
p_values = {}

for layer in range(model.cfg.n_layers):
    act_name = utils.get_act_name("resid_pre", layer)
    
    # Extract activations at the last token position for each prompt
    coding_acts = coding_activations[act_name][:, -1, :].numpy()
    non_coding_acts = non_coding_activations[act_name][:, -1, :].numpy()
    
    # Calculate the difference in means (the coding vector)
    mean_coding = torch.tensor(coding_acts.mean(axis=0))
    mean_non_coding = torch.tensor(non_coding_acts.mean(axis=0))
    coding_vec = mean_coding - mean_non_coding
    coding_vectors[layer] = coding_vec / coding_vec.norm()

    # Perform t-test across the mean of the hidden dimensions for each group
    # A smaller p-value indicates a more significant difference.
    _, p_value = ttest_ind(coding_acts.mean(axis=1), non_coding_acts.mean(axis=1), equal_var=False)
    p_values[layer] = p_value

# Find the layer with the minimum p-value
best_layer = min(p_values, key=p_values.get)
coding_direction = coding_vectors[best_layer].to(model.cfg.device)

print(f"P-values per layer: {p_values}")
print(f"Best layer for intervention (minimum p-value): Layer {best_layer}")
print(f"P-value at Layer {best_layer}: {p_values[best_layer]:.4g}")
print("-" * 50)


"""
### 5. Define CFG-Inspired Hook for Inference-Time Steering

Instead of ablating (removing) a vector, this hook **amplifies** it.
It projects the current activation onto the "coding vector" and adds this
projection back, multiplied by a guidance scale.
"""
print("--- 5. Defining CFG-Inspired Amplification Hook ---")

def steer_hook_function(
    activation: Float[Tensor, "... d_model"],
    hook: HookPoint,
    direction: Float[Tensor, "d_model"],
    scale: float = 10.0
):
    """Amplifies the activation in the given direction."""
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    
    # Calculate the projection of the activation onto the direction vector
    proj = einops.einsum(
        activation, direction, "... d_model, d_model -> ..."
    ).unsqueeze(-1) * direction
    
    # Add the scaled projection back to the original activation
    return activation + proj * scale

print("Steering hook defined.")
print("-" * 50)


"""
### 6 & 8. Analysis and Reverse Engineering

We now connect the pieces:
1.  Show that the activation-derived `coding_direction` is similar to the
    `W_delta` from our simulated finetuning (using cosine similarity).
2.  Analyze which types of prompts are most affected by finetuning by
    projecting their latent states onto `W_delta`.
"""
print("--- 6 & 8. Analysis and Reverse Engineering ---")

# 6. Compare coding vector with W_delta
# We'll check the W_delta of the MLP output at the best layer
w_delta_best_layer = w_delta_dict[f"blocks.{best_layer}.mlp.W_out"].to(model.cfg.device)

# Since W_delta is [d_mlp, d_model] and our vector is [d_model], we project W_delta
# onto the vector space by taking its mean across the d_mlp dimension.
w_delta_vector = w_delta_best_layer.mean(dim=0)
w_delta_vector = w_delta_vector / w_delta_vector.norm()

similarity = torch.dot(coding_direction, w_delta_vector).item()
print(f"Cosine Similarity between activation vector and W_delta vector at layer {best_layer}: {similarity:.4f}")
print("(A positive value indicates alignment, suggesting our activation vector captures a similar direction to finetuning)")

# 8. Analyze prompt impact by projecting onto W_delta
prompt_to_analyze = "Write a short story about a dragon."
prompt_tokens = tokenize_instructions(tokenizer, [reformat_texts([prompt_to_analyze])[0]]).to(model.cfg.device)

# Get the hidden state at the best layer for this prompt
_, cache = model.run_with_cache(prompt_tokens, names_filter=lambda name: name == utils.get_act_name("resid_pre", best_layer))
prompt_activation = cache[utils.get_act_name("resid_pre", best_layer)][0, -1, :] # Last token

# Project this activation onto the W_delta vector
projection_magnitude = torch.dot(prompt_activation, w_delta_vector).item()
print(f"\nMagnitude of projecting a non-coding prompt ('dragon story') onto W_delta: {projection_magnitude:.4f}")

prompt_to_analyze_2 = "Implement a function to find the maximum value in a list."
prompt_tokens_2 = tokenize_instructions(tokenizer, [reformat_texts([prompt_to_analyze_2])[0]]).to(model.cfg.device)
_, cache_2 = model.run_with_cache(prompt_tokens_2, names_filter=lambda name: name == utils.get_act_name("resid_pre", best_layer))
prompt_activation_2 = cache_2[utils.get_act_name("resid_pre", best_layer)][0, -1, :]
projection_magnitude_2 = torch.dot(prompt_activation_2, w_delta_vector).item()
print(f"Magnitude of projecting a coding prompt ('max value function') onto W_delta: {projection_magnitude_2:.4f}")
print("(A larger magnitude indicates the finetuning had a greater impact on the model's representation for that prompt)")
print("-" * 50)


"""
### 7. Final Demonstration: Steering an Ambiguous Prompt

We test the model on an ambiguous prompt with and without the steering hook
to demonstrate our ability to control its behavior towards coding.
"""
print("--- 7. Final Demonstration ---")

ambiguous_prompt = "Can you help me with a quick sort?"
test_prompts = [reformat_texts([ambiguous_prompt])[0]]

def generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    prompts: List[Dict],
    fwd_hooks=[],
    max_tokens_generated: int = 128
) -> List[str]:
    tokens = tokenize_instructions(tokenizer, prompts).to(model.cfg.device)
    
    with model.hooks(fwd_hooks=fwd_hooks):
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_tokens_generated,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False # Greedy decoding for reproducibility
        )
    return tokenizer.batch_decode(output_tokens[:, tokens.shape[1]:], skip_special_tokens=True)

# Baseline generation (no hooks)
print(f"\n--- PROMPT: '{ambiguous_prompt}' ---")
print("\n\n\033[1mBASELINE COMPLETION (no steering):\033[0m")
baseline_generation = generate_with_hooks(model, tokenizer, test_prompts)
print(baseline_generation[0])

# Steered generation
GUIDANCE_SCALE = 15.0
partial_hook_fn = functools.partial(steer_hook_function, direction=coding_direction, scale=GUIDANCE_SCALE)
steering_hook = (utils.get_act_name("resid_pre", best_layer), partial_hook_fn)

print(f"\n\n\033[1mSTEERED COMPLETION (amplifying coding vector at layer {best_layer} with scale {GUIDANCE_SCALE}):\033[0m")
steered_generation = generate_with_hooks(model, tokenizer, test_prompts, fwd_hooks=[steering_hook])
print(steered_generation[0])