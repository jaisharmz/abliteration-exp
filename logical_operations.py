# Full, final script for steering and P-VALUE similarity analysis

import torch
import functools
import einops
import gc
import os
import psutil
import numpy as np  # <-- NEW IMPORT
from scipy import stats  # <-- NEW IMPORT

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
# THIS IS THE REPLACED SECTION
# It starts here and includes both the analysis and the printing.
###################################################################################

# --- COMPUTE STEERING DIRECTIONS AND ANALYZE STATISTICAL SIGNIFICANCE ---
print("Computing steering directions and analyzing statistical significance...")
activation_layers_to_use = ["resid_pre", "resid_post"]
steering_directions = defaultdict(list)

# NEW: Dictionary to store statistical results
statistical_results = defaultdict(dict)

for layer_num in tqdm(range(model.cfg.n_layers), desc="Analyzing Layers"):
    pos_idx = -1  # Position index for the last token

    for layer in activation_layers_to_use:
        key = utils.get_act_name(layer, layer_num)
        
        if key in positive_acts and key in negative_acts:
            # Get the FULL set of activations at the last token position
            all_positive_acts = positive_acts[key][:, pos_idx, :]
            all_negative_acts = negative_acts[key][:, pos_idx, :]

            # --- 1. Calculate Steering Direction (for generation later) ---
            positive_mean_act = all_positive_acts.mean(dim=0)
            negative_mean_act = all_negative_acts.mean(dim=0)
            steering_dir = positive_mean_act - negative_mean_act
            steering_directions[layer].append(steering_dir / steering_dir.norm())

            # --- 2. Perform Statistical Analysis ---
            p_values = []
            t_stats = []
            
            # Convert to float32 NumPy arrays for scipy
            positive_acts_np = all_positive_acts.float().cpu().numpy()
            negative_acts_np = all_negative_acts.float().cpu().numpy()

            # Perform an independent t-test for each dimension
            for dim in range(positive_acts_np.shape[1]):
                # Use Welch's t-test (equal_var=False), robust to unequal variances
                t_stat, p_val = stats.ttest_ind(
                    positive_acts_np[:, dim],
                    negative_acts_np[:, dim],
                    equal_var=False,
                    nan_policy='omit'
                )
                
                # We care about the magnitude of difference, so use abs(t_stat)
                t_stats.append(abs(t_stat) if not np.isnan(t_stat) else 0)
                p_values.append(p_val if not np.isnan(p_val) else 1.0)
            
            # --- 3. Store the summarized results for the layer ---
            statistical_results[layer][layer_num] = {
                "mean_p_value": np.mean(p_values),
                "mean_t_stat": np.mean(t_stats),
                "significant_dims": np.sum(np.array(p_values) < 0.05)
            }

print("Statistical analysis and steering directions computed.")

# --- PRINT STATISTICAL ANALYSIS RESULTS ---
print("\n--- STATISTICAL SIGNIFICANCE ANALYSIS (Positive vs. Negative) ---")
for layer in activation_layers_to_use:
    print(f"\nAnalysis for '{layer}' activations:")
    print("Layer | Mean p-value    | Mean t-statistic | Significant Dims")
    print("-----------------------------------------------------------------")
    for layer_num, metrics in sorted(statistical_results[layer].items()):
        p_val = metrics['mean_p_value']
        t_stat = metrics['mean_t_stat']
        sig_dims = metrics['significant_dims']
        
        # Highlight layers with low p-values (high significance)
        color_start = "\033[92m" if p_val < 0.45 else "" # Green for more significant
        color_end = "\033[0m" if p_val < 0.45 else ""
        
        print(f"{layer_num: <5} | {color_start}{p_val: <15.4f}{color_end} | {t_stat: <16.3f} | {sig_dims}")

###################################################################################
# END OF REPLACED SECTION
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

# Let's dynamically find the best layer based on our new stats
# A good heuristic is to find the layer with the highest number of significant dimensions
# or the highest mean t-statistic.
best_layer_info = max(
    [(layer, num, stats) for layer, layer_stats in statistical_results.items() for num, stats in layer_stats.items()],
    key=lambda x: x[2]['mean_t_stat']
)
best_layer_name, best_layer_num, best_layer_stats = best_layer_info
print(f"\nDynamically selected best layer for intervention: {best_layer_name} at layer {best_layer_num}")
print(f"Stats: p-value={best_layer_stats['mean_p_value']:.4f}, t-stat={best_layer_stats['mean_t_stat']:.3f}, sig_dims={best_layer_stats['significant_dims']}")

# Get the specific steering direction from our dynamically found best layer
selected_steering_dir = steering_directions[best_layer_name][best_layer_num]

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
# NOTE: Based on previous results, a large coefficient causes collapse.
# Let's use a much smaller, more reasonable value.
positive_steering_coeff = 0.8
positive_steering_hook_fn = functools.partial(direction_steering_hook, direction=selected_steering_dir, coeff=positive_steering_coeff)
# Let's also apply the hook to a more targeted range of layers
start_hook_layer = max(0, best_layer_num - 5)
end_hook_layer = min(model.cfg.n_layers, best_layer_num + 5)
print(f"Applying hooks from layer {start_hook_layer} to {end_hook_layer}")
positive_steering_fwd_hooks = [
    (utils.get_act_name(best_layer_name.split('.')[-1], l), positive_steering_hook_fn)
    for l in range(start_hook_layer, end_hook_layer)
]
positive_steered_generations = get_generations(
    model, tokenizer, test_prompts, fwd_hooks=positive_steering_fwd_hooks, max_tokens_generated=64, batch_size=N_INST_TEST_STEER
)
print("Positive steered completions generated.")


# 3. Generate completions with negative steering
print("\nGenerating completions with NEGATIVE steering...")
negative_steering_coeff = -0.8
negative_steering_hook_fn = functools.partial(direction_steering_hook, direction=selected_steering_dir, coeff=negative_steering_coeff)
negative_steering_fwd_hooks = [
    (utils.get_act_name(best_layer_name.split('.')[-1], l), negative_steering_hook_fn)
    for l in range(start_hook_layer, end_hook_layer)
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
    # Corrected typo in the next line
    print(f"\033[91mNEGATIVE STEERING (coeff={negative_steering_coeff}):\n{negative_steered_generations[i]}\033[0m")

print("\n--- SCRIPT FINISHED ---")