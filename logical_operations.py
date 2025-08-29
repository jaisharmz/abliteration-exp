# Full, final script for steering and similarity analysis

import torch
import functools
import einops
import gc
import os
import psutil

from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List, Dict, Any
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int
from collections import defaultdict

# DEBUG: Function to print memory usage
def print_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    print(f"--- MEMORY CHECK ({stage}) ---")
    print(f"Current RAM usage: {ram_gb:.2f} GB")
    print("--------------------------\n")

print("--- SCRIPT START ---")
print_memory_usage("Initial state")

# Turn automatic differentiation off to save GPU memory (credit: Undi95)
torch.set_grad_enabled(False)

# --- MODEL AND TOKENIZER SETUP ---
MODEL_TYPE = "meta-llama/Meta-Llama-3-8B-Instruct" # Or your actual model path

print(f"Attempting to load model: {MODEL_TYPE}")
# Ensure you have the model downloaded locally as per original script's instructions
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_TYPE,
    local_files_only=True, # Make sure this is True if you're loading from a local path
    dtype=torch.bfloat16,
    default_padding_side='left'
).to('cuda') # Move model to GPU immediately to save RAM, if possible
print("Model loaded successfully.")
print_memory_usage("After loading model")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")
# --- END MODEL AND TOKENIZER SETUP ---


def tokenize_instructions(tokenizer, instructions: List[List[Dict[str, str]]]):
    """
    Tokenizes a list of instructions formatted for chat templates.
    """
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False, # Set to True if context window is an issue for very long prompts
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids

# Helper function to get activation index
def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer)
    return cache_dict[utils.get_act_name(*key)]


# --- DEFINE POSITIVE AND NEGATIVE INSTRUCTIONS ---
positive_inst_train = [
    [{"role": "user", "content": "Describe a truly beautiful and uplifting scene."}],
    [{"role": "user", "content": "Tell me about a moment of pure joy and happiness."}],
    [{"role": "user", "content": "Write a short story with a very optimistic ending."}],
    [{"role": "user", "content": "What are some of the most wonderful things in life?"}],
    [{"role": "user", "content": "Recount a tale of triumph and success."}],
    [{"role": "user", "content": "Imagine a world filled with peace and prosperity."}],
    [{"role": "user", "content": "Share your happiest memory."}],
    [{"role": "user", "content": "Describe the feeling of accomplishment."}],
]

negative_inst_train = [
    [{"role": "user", "content": "Describe a somewhat frustrating and difficult situation (without being harmful)."}],
    [{"role": "user", "content": "Tell me about a moment of mild disappointment or sadness (without being harmful)."}],
    [{"role": "user", "content": "Write a short story with a slightly melancholic tone."}],
    [{"role": "user", "content": "What are some minor challenges one might face in a day?"}],
    [{"role": "user", "content": "Recount a tale of minor setbacks and difficulties (harmlessly)."}],
    [{"role": "user", "content": "Imagine a dull and uneventful day."}],
    [{"role": "user", "content": "Share a memory of a minor annoyance."}],
    [{"role": "user", "content": "Describe the feeling of a small failure."}],
]

n_inst_steer = min(128, len(positive_inst_train), len(negative_inst_train))
print(f"\nUsing {n_inst_steer} steering examples for positive vs. negative.")

# --- TOKENIZE DATASETS ---
print("Tokenizing positive and negative datasets...")
positive_tokens = tokenize_instructions(
    tokenizer,
    instructions=positive_inst_train[:n_inst_steer],
).to("cuda")
negative_tokens = tokenize_instructions(
    tokenizer,
    instructions=negative_inst_train[:n_inst_steer],
).to("cuda")
print("Tokenization complete.")
print_memory_usage("After tokenization of steering datasets")

# --- CACHE ACTIVATIONS ---
batch_size = 8 # Reduce batch size for caching to avoid OOM
print(f"Using batch size: {batch_size} for activation caching.")

positive_acts = defaultdict(list)
negative_acts = defaultdict(list)

num_batches_steer = (n_inst_steer + batch_size - 1) // batch_size
print(f"Starting activation caching loop for steering ({num_batches_steer} batches)...")
for i in tqdm(range(num_batches_steer)):
    start_idx = i * batch_size
    end_idx = min(n_inst_steer, start_idx + batch_size)

    pos_logits, pos_cache = model.run_with_cache(
        positive_tokens[start_idx:end_idx],
        names_filter=lambda hook_name: 'resid' in hook_name,
        device='cpu',
        reset_hooks_end=True
    )
    neg_logits, neg_cache = model.run_with_cache(
        negative_tokens[start_idx:end_idx],
        names_filter=lambda hook_name: 'resid' in hook_name,
        device='cpu',
        reset_hooks_end=True
    )

    for key in pos_cache:
        positive_acts[key].append(pos_cache[key])
        negative_acts[key].append(neg_cache[key])

    del pos_logits, neg_logits, pos_cache, neg_cache
    gc.collect()
    torch.cuda.empty_cache()

print("\nActivation caching loop finished.")

positive_acts = {k: torch.cat(v) for k, v in positive_acts.items()}
negative_acts = {k: torch.cat(v) for k, v in negative_acts.items()}
print("Activations concatenated successfully.")
print_memory_usage("After concatenating activations")


###################################################################################
# --- START OF NEW/MODIFIED SECTION ---
# This is where you place the first new block of code.
# It replaces the old "COMPUTE STEERING DIRECTIONS" section.
###################################################################################

# --- COMPUTE STEERING DIRECTIONS AND ANALYZE VECTOR SIMILARITY ---
print("Computing steering directions and analyzing vector similarity...")
activation_layers_to_use = ["resid_pre", "resid_post"]
steering_directions = defaultdict(list)

# NEW: Dictionary to store similarity results
similarity_results = defaultdict(dict)

for layer_num in range(model.cfg.n_layers):
    pos_idx = -1  # Position index for the last token

    for layer in activation_layers_to_use:
        try:
            positive_mean_act = get_act_idx(positive_acts, layer, layer_num)[:, pos_idx, :].mean(dim=0)
            negative_mean_act = get_act_idx(negative_acts, layer, layer_num)[:, pos_idx, :].mean(dim=0)

            # --- INSERT THIS NEW ANALYSIS BLOCK ---
            # 1. Calculate Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(positive_mean_act, negative_mean_act, dim=0)

            # 2. Calculate Euclidean Distance
            euclidean_dist = torch.linalg.norm(positive_mean_act - negative_mean_act)
            
            # 3. Store the results
            similarity_results[layer][layer_num] = {
                "cosine_similarity": cos_sim.item(),
                "euclidean_distance": euclidean_dist.item()
            }
            # --- END OF NEW ANALYSIS BLOCK ---

            # The original steering direction calculation remains the same
            steering_dir = positive_mean_act - negative_mean_act
            steering_dir = steering_dir / steering_dir.norm() # Normalize the direction
            steering_directions[layer].append(steering_dir)
            
        except KeyError:
            # This part can be simplified since you will have the keys
            continue

print("Steering directions computed.")

###################################################################################
# This is where you place the second new block of code.
# It comes immediately after the section above.
###################################################################################

# --- PRINT SIMILARITY ANALYSIS RESULTS ---
print("\n--- VECTOR SIMILARITY ANALYSIS ---")
for layer in activation_layers_to_use:
    print(f"\nAnalysis for '{layer}' activations:")
    print("Layer | Cosine Similarity | Euclidean Distance")
    print("-------------------------------------------------")
    for layer_num, metrics in sorted(similarity_results[layer].items()):
        cos_sim = metrics['cosine_similarity']
        euc_dist = metrics['euclidean_distance']
        # Highlight interesting values
        color_start = "\033[93m" if cos_sim < -0.1 else "" # Yellow for negative similarity
        color_end = "\033[0m" if cos_sim < -0.1 else ""
        print(f"{layer_num: <5} | {color_start}{cos_sim: <17.4f}{color_end} | {euc_dist: <18.2f}")
# --- END OF NEW PRINTING BLOCK ---

###################################################################################
# --- END OF NEW/MODIFIED SECTION ---
# The rest of the script continues as before.
###################################################################################


# --- INFERENCE-TIME INTERVENTION HOOK (FOR STEERING) ---
def direction_steering_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    coeff: float = 1.0,
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    return activation + coeff * direction


# --- HELPER FUNCTIONS FOR GENERATION ---
def _generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    tokens: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks=[],
) -> List[str]:
    all_tokens = torch.zeros(
        (tokens.shape[0], tokens.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    )
    all_tokens[:, : tokens.shape[1]] = tokens
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_tokens[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_tokens[:, -max_tokens_generated + i] = next_tokens
    return tokenizer.batch_decode(
        all_tokens[:, tokens.shape[1] :], skip_special_tokens=True
    )

def get_generations(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[List[Dict[str, str]]],
    fwd_hooks=[],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    generations = []
    gen_batch_size = min(batch_size, 4)
    for i in tqdm(range(0, len(instructions), gen_batch_size)):
        tokens = tokenize_instructions(
            tokenizer, instructions=instructions[i : i + gen_batch_size]
        ).to("cuda")
        generation = _generate_with_hooks(
            model,
            tokenizer,
            tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations


# --- TESTING AND EVALUATION ---

STEERING_LAYER_CANDIDATE = 20 # Example: Layer 20
STEERING_LAYER_CANDIDATE = min(max(0, STEERING_LAYER_CANDIDATE), model.cfg.n_layers - 1)

# Get the specific steering direction (from 'resid_post' of the chosen layer)
selected_steering_dir = steering_directions["resid_post"][STEERING_LAYER_CANDIDATE]
print(f"\nSelected steering direction from resid_post at layer {STEERING_LAYER_CANDIDATE}.")

test_prompts = [
    [{"role": "user", "content": "Describe a typical day at the office."}],
    [{"role": "user", "content": "Write a short sentence about the weather today."}],
    [{"role": "user", "content": "What is a common household item you use daily?"}],
    [{"role": "user", "content": "Explain how a simple machine like a lever works."}],
    [{"role": "user", "content": "Tell me about the process of making coffee."}],
    [{"role": "user", "content": "Describe the view from your window."}],
]
N_INST_TEST_STEER = len(test_prompts)


# 1. Generate baseline completions (no steering)
print("\nGenerating baseline completions...")
baseline_generations = get_generations(
    model, tokenizer, test_prompts, fwd_hooks=[], max_tokens_generated=64, batch_size=N_INST_TEST_STEER
)
print("Baseline completions generated.")


# 2. Generate completions with positive steering
print("\nGenerating completions with POSITIVE steering...")
positive_steering_coeff = 5.0
positive_steering_hook_fn = functools.partial(direction_steering_hook, direction=selected_steering_dir, coeff=positive_steering_coeff)
positive_steering_fwd_hooks = [
    (utils.get_act_name("resid_post", layer_num), positive_steering_hook_fn)
    for layer_num in range(model.cfg.n_layers)
]
positive_steered_generations = get_generations(
    model, tokenizer, test_prompts, fwd_hooks=positive_steering_fwd_hooks, max_tokens_generated=64, batch_size=N_INST_TEST_STEER
)
print("Positive steered completions generated.")


# 3. Generate completions with negative steering
print("\nGenerating completions with NEGATIVE steering...")
negative_steering_coeff = -5.0
negative_steering_hook_fn = functools.partial(direction_steering_hook, direction=selected_steering_dir, coeff=negative_steering_coeff)
negative_steering_fwd_hooks = [
    (utils.get_act_name("resid_post", layer_num), negative_steering_hook_fn)
    for layer_num in range(model.cfg.n_layers)
]
negative_steered_generations = get_generations(
    model, tokenizer, test_prompts, fwd_hooks=negative_steering_fwd_hooks, max_tokens_generated=64, batch_size=N_INST_TEST_STEER
)
print("Negative steered completions generated.")


# Print generations for comparison
print("\n--- COMPARISON OF GENERATIONS ---")
for i in range(N_INST_TEST_STEER):
    print(f"\n\033[1mINSTRUCTION {i}: {test_prompts[i][0]['content']}\033[0m")
    print(f"\033[92mBASELINE COMPLETION:\n{baseline_generations[i]}\033[0m")
    print(f"\033[94mPOSITIVE STEERING (coeff={positive_steering_coeff}):\n{positive_steered_generations[i]}\033[0m")
    print(f"\03i[91mNEGATIVE STEERING (coeff={negative_steering_coeff}):\n{negative_steered_generations[i]}\033[0m")

print("\n--- SCRIPT FINISHED ---")