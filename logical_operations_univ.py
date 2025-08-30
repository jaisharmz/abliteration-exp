# Logical Operations Vector Analysis - Testing Universal Logical Operators

import torch
import functools
import einops
import gc
import os
import psutil
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from torch import Tensor
from typing import List, Dict, Any, Tuple
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

print("--- LOGICAL OPERATIONS ANALYSIS START ---")
print_memory_usage("Initial state")

# Turn automatic differentiation off to save GPU memory
torch.set_grad_enabled(False)

# --- MODEL AND TOKENIZER SETUP ---
MODEL_TYPE = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Attempting to load model: {MODEL_TYPE}")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_TYPE,
    local_files_only=True,
    dtype=torch.bfloat16,
    default_padding_side='left'
).to('cuda')
print("Model loaded successfully.")
print_memory_usage("After loading model")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

def get_activation_for_text(text: str, layer_name: str, layer_num: int) -> torch.Tensor:
    """Get activation vector for a single text at specified layer."""
    tokens = tokenizer.apply_chat_template(
        [[{"role": "user", "content": text}]],
        padding=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).input_ids.to("cuda")
    
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda hook_name: utils.get_act_name(layer_name, layer_num) in hook_name,
        device='cpu',
        reset_hooks_end=True
    )
    
    key = utils.get_act_name(layer_name, layer_num)
    # Return activation at last token position
    activation = cache[key][0, -1, :]
    
    del cache, tokens
    gc.collect()
    torch.cuda.empty_cache()
    
    return activation

def compute_negation_operator(pairs: List[Tuple[str, str]], layer_name: str, layer_num: int) -> torch.Tensor:
    """
    Compute negation operator from positive/negative pairs.
    Returns: negation_vector = mean(negative_activations) - mean(positive_activations)
    """
    positive_acts = []
    negative_acts = []
    
    print(f"Computing negation operator from {len(pairs)} pairs...")
    for pos_text, neg_text in tqdm(pairs):
        pos_act = get_activation_for_text(pos_text, layer_name, layer_num)
        neg_act = get_activation_for_text(neg_text, layer_name, layer_num)
        
        positive_acts.append(pos_act)
        negative_acts.append(neg_act)
    
    pos_mean = torch.stack(positive_acts).mean(dim=0)
    neg_mean = torch.stack(negative_acts).mean(dim=0)
    
    negation_operator = neg_mean - pos_mean
    return negation_operator

def test_cross_domain_generalization(
    negation_op: torch.Tensor, 
    test_pairs: List[Tuple[str, str]], 
    layer_name: str, 
    layer_num: int
) -> List[float]:
    """
    Test if negation operator generalizes across domains.
    Returns list of similarities between predicted and actual negations.
    """
    similarities = []
    
    print(f"Testing generalization on {len(test_pairs)} pairs...")
    for pos_text, neg_text in tqdm(test_pairs):
        # Get actual positive activation
        pos_act = get_activation_for_text(pos_text, layer_name, layer_num)
        
        # Predict negative using learned operator
        predicted_neg = pos_act + negation_op
        
        # Get actual negative activation
        actual_neg = get_activation_for_text(neg_text, layer_name, layer_num)
        
        # Compute similarity
        similarity = cosine_similarity(
            predicted_neg.cpu().numpy().reshape(1, -1),
            actual_neg.cpu().numpy().reshape(1, -1)
        )[0, 0]
        
        similarities.append(similarity)
        
    return similarities

# --- DEFINE DOMAIN-SPECIFIC NEGATION PAIRS ---
emotion_pairs = [
    ("I am happy", "I am not happy"),
    ("I am sad", "I am not sad"),
    ("I am excited", "I am not excited"),
    ("I feel joyful", "I do not feel joyful"),
    ("I am angry", "I am not angry")
]

weather_pairs = [
    ("It is sunny", "It is not sunny"),
    ("It is raining", "It is not raining"),
    ("It is cold", "It is not cold"),
    ("The weather is warm", "The weather is not warm"),
    ("It is cloudy", "It is not cloudy")
]

object_state_pairs = [
    ("The door is open", "The door is not open"),
    ("The light is on", "The light is not on"),
    ("The car is moving", "The car is not moving"),
    ("The book is closed", "The book is not closed"),
    ("The phone is ringing", "The phone is not ringing")
]

# --- ANALYSIS PARAMETERS ---
# Using layer 15 based on your previous results showing it had good statistical properties
ANALYSIS_LAYER_NAME = "resid_pre"
ANALYSIS_LAYER_NUM = 15

print(f"\nAnalyzing logical operations at layer: {ANALYSIS_LAYER_NAME}_{ANALYSIS_LAYER_NUM}")

# --- STEP 1: COMPUTE DOMAIN-SPECIFIC NEGATION OPERATORS ---
print("\n=== STEP 1: Computing Domain-Specific Negation Operators ===")

emotion_negation_op = compute_negation_operator(emotion_pairs, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)
weather_negation_op = compute_negation_operator(weather_pairs, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)
object_negation_op = compute_negation_operator(object_state_pairs, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)

print("All negation operators computed successfully.")

# --- STEP 2: TEST SIMILARITY BETWEEN NEGATION OPERATORS ---
print("\n=== STEP 2: Testing Similarity Between Negation Operators ===")

# Convert to numpy for cosine similarity
emotion_op_np = emotion_negation_op.cpu().numpy().reshape(1, -1)
weather_op_np = weather_negation_op.cpu().numpy().reshape(1, -1)
object_op_np = object_negation_op.cpu().numpy().reshape(1, -1)

# Compute pairwise similarities
emo_weather_sim = cosine_similarity(emotion_op_np, weather_op_np)[0, 0]
emo_object_sim = cosine_similarity(emotion_op_np, object_op_np)[0, 0]
weather_object_sim = cosine_similarity(weather_op_np, object_op_np)[0, 0]

print(f"Emotion-Weather Negation Similarity: {emo_weather_sim:.4f}")
print(f"Emotion-Object Negation Similarity:  {emo_object_sim:.4f}")
print(f"Weather-Object Negation Similarity:  {weather_object_sim:.4f}")
print(f"Average Cross-Domain Similarity:     {(emo_weather_sim + emo_object_sim + weather_object_sim)/3:.4f}")

# --- STEP 3: TEST CROSS-DOMAIN GENERALIZATION ---
print("\n=== STEP 3: Testing Cross-Domain Generalization ===")

# Test 1: Train on emotions, test on weather
print("\nTest 1: Emotion operator → Weather domain")
weather_sims = test_cross_domain_generalization(
    emotion_negation_op, weather_pairs, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM
)
print(f"Weather generalization similarities: {[f'{s:.3f}' for s in weather_sims]}")
print(f"Average weather generalization: {np.mean(weather_sims):.4f}")

# Test 2: Train on emotions, test on objects
print("\nTest 2: Emotion operator → Object domain")
object_sims = test_cross_domain_generalization(
    emotion_negation_op, object_state_pairs, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM
)
print(f"Object generalization similarities: {[f'{s:.3f}' for s in object_sims]}")
print(f"Average object generalization: {np.mean(object_sims):.4f}")

# Test 3: Train on weather, test on emotions
print("\nTest 3: Weather operator → Emotion domain")
emotion_sims = test_cross_domain_generalization(
    weather_negation_op, emotion_pairs, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM
)
print(f"Emotion generalization similarities: {[f'{s:.3f}' for s in emotion_sims]}")
print(f"Average emotion generalization: {np.mean(emotion_sims):.4f}")

# --- STEP 4: ANALYSIS AND CONCLUSIONS ---
print("\n=== STEP 4: Analysis Summary ===")

overall_generalization = np.mean(weather_sims + object_sims + emotion_sims)
operator_similarity = (emo_weather_sim + emo_object_sim + weather_object_sim) / 3

print(f"\nKEY FINDINGS:")
print(f"1. Cross-domain operator similarity: {operator_similarity:.4f}")
print(f"2. Cross-domain generalization accuracy: {overall_generalization:.4f}")

if operator_similarity > 0.7:
    print("✅ HIGH operator similarity - Evidence for universal negation operator!")
elif operator_similarity > 0.5:
    print("⚠️  MODERATE operator similarity - Partial evidence for universal operator")
else:
    print("❌ LOW operator similarity - Limited evidence for universal operator")

if overall_generalization > 0.7:
    print("✅ HIGH generalization - Negation operators transfer well across domains!")
elif overall_generalization > 0.5:
    print("⚠️  MODERATE generalization - Some cross-domain transfer")
else:
    print("❌ LOW generalization - Limited cross-domain transfer")

# --- STEP 5: TEST CONJUNCTION (AND) OPERATOR ---
print("\n=== STEP 5: Testing AND Operator ===")

# Define conjunction examples
conjunction_examples = [
    (("I like apples", "I like oranges"), "I like apples and oranges"),
    (("It is sunny", "It is warm"), "It is sunny and warm"),
    (("The door is open", "The light is on"), "The door is open and the light is on"),
    (("I am happy", "I am excited"), "I am happy and excited"),
    (("The car is red", "The car is fast"), "The car is red and fast")
]

def compute_and_operator(examples: List[Tuple[Tuple[str, str], str]], layer_name: str, layer_num: int) -> torch.Tensor:
    """
    Compute AND operator from conjunction examples.
    AND_op = mean(combined) - mean(A) - mean(B)
    """
    and_ops = []
    
    print(f"Computing AND operator from {len(examples)} examples...")
    for (a_text, b_text), combined_text in tqdm(examples):
        vec_a = get_activation_for_text(a_text, layer_name, layer_num)
        vec_b = get_activation_for_text(b_text, layer_name, layer_num)
        vec_combined = get_activation_for_text(combined_text, layer_name, layer_num)
        
        # AND operator should be what's left after subtracting individual components
        and_op = vec_combined - vec_a - vec_b
        and_ops.append(and_op)
    
    return torch.stack(and_ops).mean(dim=0)

and_operator = compute_and_operator(conjunction_examples, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)

# Test AND operator generalization
test_conjunctions = [
    (("I feel tired", "I feel stressed"), "I feel tired and stressed"),
    (("The weather is cold", "The weather is windy"), "The weather is cold and windy")
]

print("\nTesting AND operator generalization...")
and_similarities = []
for (a_text, b_text), expected_combined in tqdm(test_conjunctions):
    vec_a = get_activation_for_text(a_text, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)
    vec_b = get_activation_for_text(b_text, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)
    
    # Predict combined using learned AND operator
    predicted_combined = vec_a + vec_b + and_operator
    
    # Get actual combined
    actual_combined = get_activation_for_text(expected_combined, ANALYSIS_LAYER_NAME, ANALYSIS_LAYER_NUM)
    
    similarity = cosine_similarity(
        predicted_combined.cpu().numpy().reshape(1, -1),
        actual_combined.cpu().numpy().reshape(1, -1)
    )[0, 0]
    
    and_similarities.append(similarity)
    print(f"AND prediction similarity: {similarity:.4f}")

print(f"\nAverage AND operator generalization: {np.mean(and_similarities):.4f}")

# --- FINAL RESULTS ---
print("\n" + "="*60)
print("FINAL RESULTS - LOGICAL OPERATORS IN NEURAL ACTIVATION SPACE")
print("="*60)
print(f"Universal Negation Operator Evidence:")
print(f"  - Cross-domain similarity: {operator_similarity:.4f}")
print(f"  - Generalization accuracy: {overall_generalization:.4f}")
print(f"\nConjunction (AND) Operator Evidence:")
print(f"  - Generalization accuracy: {np.mean(and_similarities):.4f}")

if operator_similarity > 0.6 and overall_generalization > 0.6:
    print(f"\n🎉 STRONG EVIDENCE for universal logical operators in neural space!")
    print(f"   The model appears to have consistent geometric representations for logical operations.")
elif operator_similarity > 0.4 and overall_generalization > 0.4:
    print(f"\n🤔 MODERATE EVIDENCE for logical operators.")
    print(f"   Some structure exists but may not be fully universal.")
else:
    print(f"\n📊 LIMITED EVIDENCE for universal logical operators.")
    print(f"   Logical operations may be more context-dependent than expected.")

print("\n--- LOGICAL OPERATIONS ANALYSIS COMPLETE ---")