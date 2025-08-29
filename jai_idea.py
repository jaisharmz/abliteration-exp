# -*- coding: utf-8 -*-
"""
Semantic Direction Experiments for Code Generation Models
Exploring how fine-tuning creates discoverable semantic directions
"""

import torch
import functools
import einops
import gc
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List, Dict, Tuple
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int
from collections import defaultdict

# Turn automatic differentiation off to save GPU memory
torch.set_grad_enabled(False)

class SemanticDirectionExplorer:
    def __init__(self, base_model_id: str, finetuned_model_id: str = None):
        """
        Initialize the semantic direction explorer
        
        Args:
            base_model_id: HuggingFace model ID for base model
            finetuned_model_id: Optional fine-tuned model ID for comparison
        """
        self.base_model_id = base_model_id
        self.finetuned_model_id = finetuned_model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models
        print(f"Loading base model: {base_model_id}")
        try:
            self.base_model = HookedTransformer.from_pretrained(
                base_model_id,
                dtype=torch.bfloat16,
                default_padding_side='left',
                device=self.device
            )
        except Exception as e:
            print(f"Error loading {base_model_id}: {e}")
            print("Trying with CPU and float32...")
            self.base_model = HookedTransformer.from_pretrained(
                base_model_id,
                dtype=torch.float32,
                default_padding_side='left',
                device='cpu'
            )
            self.device = 'cpu'
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load fine-tuned model if provided
        self.finetuned_model = None
        self.w_delta = None
        if finetuned_model_id:
            print(f"Loading fine-tuned model: {finetuned_model_id}")
            try:
                self.finetuned_model = HookedTransformer.from_pretrained(
                    finetuned_model_id,
                    dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                    default_padding_side='left',
                    device=self.device
                )
                self._compute_weight_delta()
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")
                print("Continuing without fine-tuned model comparison...")
    
        print(f"Models loaded on device: {self.device}")
        print(f"Model has {self.base_model.cfg.n_layers} layers")
    
    def _compute_weight_delta(self):
        """Compute W_delta = W' - W between fine-tuned and base models"""
        if not self.finetuned_model:
            return
        
        print("Computing weight differences (W_delta)...")
        self.w_delta = {}
        
        base_state = self.base_model.state_dict()
        ft_state = self.finetuned_model.state_dict()
        
        for key in base_state:
            if key in ft_state:
                self.w_delta[key] = ft_state[key] - base_state[key]
        
        print(f"Computed W_delta for {len(self.w_delta)} parameters")
    
    def load_datasets(self, n_samples: int = 500):
        """Load coding and non-coding datasets from HuggingFace"""
        print("Loading datasets...")
        
        # Coding dataset - using CodeAlpaca
        try:
            code_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            self.coding_prompts = []
            for item in code_dataset.select(range(min(n_samples, len(code_dataset)))):
                # Handle different instruction formats
                content = item.get("instruction", "") or item.get("prompt", "") or str(item)
                self.coding_prompts.append([{"role": "user", "content": content}])
            print(f"Loaded {len(self.coding_prompts)} coding prompts")
        except Exception as e:
            print(f"Could not load CodeAlpaca: {e}")
            print("Using backup coding prompts...")
            self.coding_prompts = self._get_backup_coding_prompts()[:n_samples]
        
        # Non-coding dataset - using general Alpaca
        try:
            general_dataset = load_dataset("tatsu-lab/alpaca", split="train")
            # Filter out coding-related prompts
            non_coding = []
            coding_keywords = ['code', 'program', 'function', 'algorithm', 'python', 'javascript', 'java', 'c++', 'programming', 'script', 'debug']
            
            for item in general_dataset:
                instruction = item.get("instruction", "").lower()
                if not any(keyword in instruction for keyword in coding_keywords) and len(instruction) > 10:
                    non_coding.append([{"role": "user", "content": item["instruction"]}])
                if len(non_coding) >= n_samples:
                    break
            
            self.noncoding_prompts = non_coding
            print(f"Loaded {len(self.noncoding_prompts)} non-coding prompts")
        except Exception as e:
            print(f"Could not load Alpaca: {e}")
            print("Using backup non-coding prompts...")
            self.noncoding_prompts = self._get_backup_noncoding_prompts()[:n_samples]
        
        # Ensure we have equal numbers of both types
        min_samples = min(len(self.coding_prompts), len(self.noncoding_prompts), n_samples)
        self.coding_prompts = self.coding_prompts[:min_samples]
        self.noncoding_prompts = self.noncoding_prompts[:min_samples]
        
        print(f"Final dataset sizes: {len(self.coding_prompts)} coding, {len(self.noncoding_prompts)} non-coding")
    
    def _get_backup_coding_prompts(self):
        """Backup coding prompts if dataset loading fails"""
        prompts = [
            "Write a Python function to calculate fibonacci numbers",
            "Create a sorting algorithm in JavaScript", 
            "Implement a binary search tree in C++",
            "Write a REST API endpoint using Flask",
            "Create a React component for a todo list",
            "Implement depth-first search algorithm",
            "Write a function to reverse a linked list",
            "Create a database query to find top customers",
            "Implement a hash table in Python",
            "Write a regular expression to validate emails"
        ]
        return [[{"role": "user", "content": p}] for p in prompts]
    
    def _get_backup_noncoding_prompts(self):
        """Backup non-coding prompts if dataset loading fails"""
        prompts = [
            "Explain the causes of World War II",
            "What are the benefits of meditation?",
            "How do you bake chocolate chip cookies?",
            "Describe the water cycle",
            "What is the history of jazz music?",
            "How do plants perform photosynthesis?",
            "Explain the theory of relativity",
            "What are the health benefits of exercise?",
            "Describe different types of clouds",
            "How do you write a good essay?"
        ]
        return [[{"role": "user", "content": p}] for p in prompts]
    
    def tokenize_instructions(self, instructions: List[List[Dict]]):
        """Tokenize instruction prompts"""
        # Handle models without chat templates (like GPT-2)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                instructions,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).input_ids.to(self.device)
        else:
            # For models without chat templates, extract the content directly
            texts = [inst[0]["content"] for inst in instructions]
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,  # Reasonable max length
                return_tensors="pt",
                return_attention_mask=False
            ).input_ids.to(self.device)
    
    def extract_activations(self, batch_size: int = 32):
        """Extract activations for coding and non-coding prompts"""
        print("Extracting activations...")
        
        n_samples = min(len(self.coding_prompts), len(self.noncoding_prompts))
        
        # Tokenize datasets
        coding_tokens = self.tokenize_instructions(self.coding_prompts[:n_samples])
        noncoding_tokens = self.tokenize_instructions(self.noncoding_prompts[:n_samples])
        
        # Initialize storage
        coding_acts = defaultdict(list)
        noncoding_acts = defaultdict(list)
        
        # Process in batches
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min(n_samples, start_idx + batch_size)
            
            # Get activations for coding prompts
            _, coding_cache = self.base_model.run_with_cache(
                coding_tokens[start_idx:end_idx],
                names_filter=lambda name: 'resid' in name,
                device='cpu',
                reset_hooks_end=True
            )
            
            # Get activations for non-coding prompts  
            _, noncoding_cache = self.base_model.run_with_cache(
                noncoding_tokens[start_idx:end_idx],
                names_filter=lambda name: 'resid' in name,
                device='cpu',
                reset_hooks_end=True
            )
            
            # Store activations (keep in original dtype but on CPU)
            for key in coding_cache:
                coding_acts[key].append(coding_cache[key])
                noncoding_acts[key].append(noncoding_cache[key])
            
            # Cleanup
            del coding_cache, noncoding_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate activations
        self.coding_activations = {k: torch.cat(v) for k, v in coding_acts.items()}
        self.noncoding_activations = {k: torch.cat(v) for k, v in noncoding_acts.items()}
        
        print(f"Extracted activations for {len(self.coding_activations)} layers")
    
    def compute_semantic_directions(self):
        """Compute semantic directions (coding vectors) for each layer"""
        print("Computing semantic directions...")
        
        self.semantic_directions = {}
        self.layer_statistics = {}
        
        activation_layers = ["resid_pre", "resid_mid", "resid_post"]
        
        # Debug: Check data types
        sample_key = list(self.coding_activations.keys())[0]
        print(f"Sample activation dtype: {self.coding_activations[sample_key].dtype}")
        print(f"Sample activation shape: {self.coding_activations[sample_key].shape}")
        
        for layer_num in range(self.base_model.cfg.n_layers):
            for layer_type in activation_layers:
                key = utils.get_act_name(layer_type, layer_num)
                
                if key in self.coding_activations and key in self.noncoding_activations:
                    # Get activations at last position (where response would start)
                    pos = -1
                    coding_acts = self.coding_activations[key][:, pos, :]
                    noncoding_acts = self.noncoding_activations[key][:, pos, :]
                    
                    # Compute means
                    coding_mean = coding_acts.mean(dim=0)
                    noncoding_mean = noncoding_acts.mean(dim=0)
                    
                    # Compute semantic direction (coding vector)
                    semantic_dir = coding_mean - noncoding_mean
                    semantic_dir = semantic_dir / semantic_dir.norm()
                    
                    self.semantic_directions[key] = semantic_dir
                    
                    # Compute statistical significance using t-test (convert to float for stats)
                    t_stats = []
                    p_values = []
                    
                    # Convert to float32 for scipy compatibility
                    coding_acts_float = coding_acts.float().cpu().numpy()
                    noncoding_acts_float = noncoding_acts.float().cpu().numpy()
                    
                    for dim in range(coding_acts.shape[1]):
                        try:
                            t_stat, p_val = stats.ttest_ind(
                                coding_acts_float[:, dim],
                                noncoding_acts_float[:, dim]
                            )
                            if not (np.isnan(t_stat) or np.isnan(p_val)):
                                t_stats.append(abs(t_stat))
                                p_values.append(p_val)
                            else:
                                t_stats.append(0.0)
                                p_values.append(1.0)
                        except:
                            t_stats.append(0.0)
                            p_values.append(1.0)
                    
                    mean_p_value = np.mean(p_values)
                    mean_t_stat = np.mean(t_stats)
                    
                    self.layer_statistics[key] = {
                        'mean_p_value': mean_p_value,
                        'mean_t_stat': mean_t_stat,
                        'significant_dims': sum(p < 0.05 for p in p_values),
                        'total_dims': len(p_values)
                    }
        
        print(f"Computed semantic directions for {len(self.semantic_directions)} layer activations")
    
    def find_best_intervention_layer(self):
        """Find the layer with the most significant differences for intervention"""
        if not self.layer_statistics:
            self.compute_semantic_directions()
        
        # Sort by mean t-statistic (higher = more significant difference)
        sorted_layers = sorted(
            self.layer_statistics.items(),
            key=lambda x: x[1]['mean_t_stat'],
            reverse=True
        )
        
        print("\nTop 5 layers by statistical significance:")
        for i, (layer, stats) in enumerate(sorted_layers[:5]):
            print(f"{i+1}. {layer}: t-stat={stats['mean_t_stat']:.3f}, "
                  f"p-value={stats['mean_p_value']:.3f}, "
                  f"significant_dims={stats['significant_dims']}/{stats['total_dims']}")
        
        self.best_layer = sorted_layers[0][0]
        print(f"\nSelected layer for intervention: {self.best_layer}")
        return self.best_layer
    
    def amplify_semantic_direction_hook(self, 
                                       activation: Float[Tensor, "... d_act"],
                                       hook: HookPoint,
                                       direction: Float[Tensor, "d_act"],
                                       amplification_factor: float = 10.0):
        """Hook function to amplify semantic direction (inspired by CFG)"""
        # Ensure both tensors are on the same device and have the same dtype
        if activation.device != direction.device:
            direction = direction.to(activation.device)
        if activation.dtype != direction.dtype:
            direction = direction.to(activation.dtype)
        
        # Compute projection onto semantic direction
        proj = einops.einsum(
            activation, direction.view(-1, 1), 
            "... d_act, d_act single -> ... single"
        ) * direction
        
        # Amplify the projection and add it back
        amplified_proj = proj * (amplification_factor - 1)  # -1 because original is already there
        
        return activation + amplified_proj
    
    def generate_with_intervention(self,
                                 prompts: List[str],
                                 amplification_factor: float = 10.0,
                                 max_tokens: int = 100,
                                 batch_size: int = 4):
        """Generate text with semantic direction amplification"""
        if not hasattr(self, 'best_layer'):
            self.find_best_intervention_layer()
        
        semantic_dir = self.semantic_directions[self.best_layer]
        
        # Create hook
        hook_fn = functools.partial(
            self.amplify_semantic_direction_hook,
            direction=semantic_dir,
            amplification_factor=amplification_factor
        )
        
        fwd_hooks = [(self.best_layer, hook_fn)]
        
        generations = []
        
        # Handle different tokenization methods
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            formatted_prompts = [[{"role": "user", "content": p}] for p in prompts]
        else:
            formatted_prompts = [[{"role": "user", "content": p}] for p in prompts]  # Keep same format for consistency
        
        for i in tqdm(range(0, len(formatted_prompts), batch_size)):
            batch_prompts = formatted_prompts[i:i+batch_size]
            tokens = self.tokenize_instructions(batch_prompts)
            
            # Generate with hooks
            batch_generations = self._generate_with_hooks(
                tokens, max_tokens, fwd_hooks
            )
            generations.extend(batch_generations)
        
        return generations
    
    def _generate_with_hooks(self, tokens, max_tokens, fwd_hooks):
        """Generate text with forward hooks"""
        all_tokens = torch.zeros(
            (tokens.shape[0], tokens.shape[1] + max_tokens),
            dtype=torch.long,
            device=tokens.device,
        )
        all_tokens[:, :tokens.shape[1]] = tokens
        
        for i in range(max_tokens):
            with self.base_model.hooks(fwd_hooks=fwd_hooks):
                logits = self.base_model(all_tokens[:, :-max_tokens + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_tokens[:, -max_tokens + i] = next_tokens
        
        return self.tokenizer.batch_decode(
            all_tokens[:, tokens.shape[1]:], skip_special_tokens=True
        )
    
    def compare_with_weight_delta(self):
        """Compare semantic directions with W_delta if available"""
        if not self.w_delta:
            print("W_delta not available. Need fine-tuned model for comparison.")
            return
        
        print("Comparing semantic directions with W_delta...")
        
        similarities = {}
        
        # For each semantic direction, find most similar W_delta component
        for layer_key, semantic_dir in self.semantic_directions.items():
            max_similarity = -1
            best_match = None
            
            for weight_key, weight_delta in self.w_delta.items():
                if weight_delta.dim() >= 2:  # Only compare with 2D+ tensors
                    # Flatten weight delta and compute similarity
                    flattened_delta = weight_delta.flatten()
                    if len(flattened_delta) == len(semantic_dir):
                        similarity = torch.cosine_similarity(
                            semantic_dir.flatten(), 
                            flattened_delta.flatten(),
                            dim=0
                        ).item()
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = weight_key
            
            if best_match:
                similarities[layer_key] = {
                    'best_match': best_match,
                    'similarity': max_similarity
                }
        
        # Print top similarities
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1]['similarity'], reverse=True)
        
        print("\nTop 5 semantic direction - W_delta similarities:")
        for i, (layer, sim_info) in enumerate(sorted_sims[:5]):
            print(f"{i+1}. {layer} <-> {sim_info['best_match']}: {sim_info['similarity']:.4f}")
        
        return similarities
    
    def analyze_prompt_projections(self, test_prompts: List[str]):
        """Project test prompts onto W_delta to understand fine-tuning effects"""
        if not self.w_delta:
            print("W_delta not available. Need fine-tuned model for analysis.")
            return
        
        print("Analyzing prompt projections onto W_delta...")
        
        # Get embeddings for test prompts
        formatted_prompts = [[{"role": "user", "content": p}] for p in test_prompts]
        tokens = self.tokenize_instructions(formatted_prompts)
        
        # Get activations at the best layer
        if not hasattr(self, 'best_layer'):
            self.find_best_intervention_layer()
        
        _, cache = self.base_model.run_with_cache(
            tokens,
            names_filter=lambda name: name == self.best_layer,
            device='cpu'
        )
        
        activations = cache[self.best_layer][:, -1, :]  # Last position
        
        # Project onto semantic direction
        semantic_dir = self.semantic_directions[self.best_layer]
        projections = torch.matmul(activations, semantic_dir.cpu())
        
        # Analyze projections
        results = []
        for i, (prompt, proj) in enumerate(zip(test_prompts, projections)):
            results.append({
                'prompt': prompt,
                'projection_magnitude': proj.item(),
                'abs_projection': abs(proj.item())
            })
        
        # Sort by projection magnitude
        results.sort(key=lambda x: x['abs_projection'], reverse=True)
        
        print(f"\nPrompt projections onto coding semantic direction:")
        print("(Higher magnitude = more affected by coding fine-tuning)")
        for i, result in enumerate(results[:10]):  # Top 10
            print(f"{i+1}. [{result['projection_magnitude']:+.4f}] {result['prompt'][:60]}...")
        
        return results
    
    def run_full_experiment(self, test_prompts: List[str] = None):
        """Run the complete semantic direction experiment"""
        print("=" * 60)
        print("SEMANTIC DIRECTION EXPERIMENT")
        print("=" * 60)
        
        # Step 1: Load datasets
        self.load_datasets(n_samples=100)  # Use smaller sample for testing
        
        # Test tokenization
        print("\nTesting tokenization...")
        test_batch = self.coding_prompts[:2]  # Test with 2 examples
        try:
            test_tokens = self.tokenize_instructions(test_batch)
            print(f"Tokenization successful. Sample shape: {test_tokens.shape}")
            print(f"Sample tokens: {test_tokens[0][:10].tolist()}")
        except Exception as e:
            print(f"Tokenization failed: {e}")
            return
        
        # Step 2: Extract activations
        self.extract_activations(batch_size=8)  # Smaller batch size
        
        # Step 3: Compute semantic directions
        self.compute_semantic_directions()
        
        # Step 4: Find best intervention layer
        self.find_best_intervention_layer()
        
        # Step 5: Compare with W_delta if available
        if self.w_delta:
            self.compare_with_weight_delta()
        
        # Step 6: Test interventions on ambiguous prompts
        if test_prompts is None:
            test_prompts = [
                "How do I sort a list?",
                "What's the best way to organize data?",
                "Can you help me with loops?",
                "I need to process some information",
                "How do I make something more efficient?"
            ]
        
        print(f"\n{'-'*60}")
        print("TESTING SEMANTIC DIRECTION AMPLIFICATION")
        print(f"{'-'*60}")
        
        # Generate baseline responses
        print("\nBaseline generations:")
        baseline_gens = self.generate_with_intervention(test_prompts, amplification_factor=1.0)
        
        # Generate with coding amplification
        print("\nCoding-amplified generations:")
        amplified_gens = self.generate_with_intervention(test_prompts, amplification_factor=10.0)
        
        # Display results
        for i, prompt in enumerate(test_prompts):
            print(f"\n{'='*40}")
            print(f"PROMPT: {prompt}")
            print(f"{'='*40}")
            print(f"BASELINE:\n{baseline_gens[i]}")
            print(f"\nCODING-AMPLIFIED:\n{amplified_gens[i]}")
        
        # Step 7: Analyze prompt projections if W_delta available
        if self.w_delta:
            print(f"\n{'-'*60}")
            print("ANALYZING PROMPT PROJECTIONS")
            print(f"{'-'*60}")
            self.analyze_prompt_projections(test_prompts)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")

# Example usage
def get_recommended_models():
    """Get recommended models for different use cases"""
    return {
        "small_fast": ["gpt2", "distilgpt2", "EleutherAI/pythia-70m"],
        "medium": ["gpt2-medium", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b"],
        "code_focused": ["codellama/CodeLlama-7b-hf", "bigcode/santacoder"],
        "instruction_following": ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.1"]
    }

if __name__ == "__main__":
    # Print recommended models
    print("Recommended models by category:")
    for category, models in get_recommended_models().items():
        print(f"{category}: {models}")
    
    print(f"\n{'='*60}")
    print("STARTING EXPERIMENT")
    print(f"{'='*60}")
    
    # Initialize with a base model (you can add a fine-tuned model for comparison)
    explorer = SemanticDirectionExplorer(
        base_model_id="gpt2",  # Use gpt2 - supported by transformer_lens
        # finetuned_model_id="your-finetuned-model-id"  # Optional
    )
    
    # Run the full experiment
    test_prompts = [
        "How do I sort a list?",
        "What's the best way to organize data?", 
        "Can you help me with loops?",
        "I need to process some information",
        "How do I make something more efficient?",
        "What are the health benefits of exercise?",  # Non-coding
        "How do you bake a cake?",  # Non-coding
    ]
    
    explorer.run_full_experiment(test_prompts)