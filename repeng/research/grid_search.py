from pprint import pprint
import re
import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid

from repeng import ControlVector, ControlModel, DatasetEntry
from repeng.research import datasets

from sklearnex import patch_sklearn

patch_sklearn()

# Model configuration
model_name = "qwen/qwen3-4b"

# Quantization config
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    device_map="cuda",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Define parameter grid for comprehensive search
param_grid = {
    'method': ["mean", "median", "pca_diff", "pca_center", "umap"],
    'layer_zones': [
        [[0.2, 0.51]], 
        [[0.1, 0.4]], 
        [[0.3, 0.7]], 
        [[0.4, 0.8]], 
        [[0.5, 0.9]],
        [[0.2, 0.51], [0.7, 0.9]],  # multiple zones
        [[0.1, 0.3], [0.6, 0.8]],
        [[0.0, 0.3]],  # early layers
        [[0.7, 1.0]],  # late layers
    ]
}

# Strengths to test (same as first.py)
strengths = [
    -20,
    -15,
    -10,
    -9,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    -0.8,
    -0.6,
    -0.4,
    -0.2,
    -0.1,
    0,
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    15,
    20,
]

# Scenario for generation
scenario: str = tokenizer.apply_chat_template(
    conversation=[
        {
            "role": "user",
            "content": "My official IQ score is ",
        },
    ],
    continue_final_message=True,
    tokenize=False,
)


def extract_first_number(text: str) -> float | None:
    """Extract the first number from text using regex."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None


def format_layer_zones_for_filename(layer_zones: list) -> str:
    """Format layer zones for use in filenames."""
    zones_str = str(layer_zones).replace(" ", "").replace("[", "").replace("]", "").replace(",", "_").replace(".", "")
    return zones_str


def test_configuration(method: str, layer_zones: list, combo_idx: int, total_combos: int) -> dict:
    """Test a single configuration and return results."""
    print(f"\n=== Combination {combo_idx+1}/{total_combos} ===")
    print(f"Method: {method}")
    print(f"Layer zones: {layer_zones}")
    
    try:
        # Create fresh control model for this configuration
        # Note: ControlModel mutates the base model, so we work with the same instance
        control_model = ControlModel(
            base_model,
            layer_zones=layer_zones,
        )
        
        # Train control vector
        print("Training control vector...")
        trained_vector = ControlVector.train(
            control_model,
            tokenizer,
            datasets.dumb_genius_paragraph,
            batch_size=1,
            method=method,
            cache_path="./model_cache",
        )
        
        # Test all strengths
        scores = {}
        outputs = {}
        
        for strength in strengths:
            print(f"Testing strength: {strength}")
            control_model.set_control(trained_vector, strength)
            
            out = control_model.generate(
                **tokenizer(scenario, return_tensors="pt").to(control_model.device),
                do_sample=False,
                max_new_tokens=30,
                repetition_penalty=1.1,
            )
            
            output = tokenizer.decode(out.squeeze(), skip_special_tokens=True).strip()
            outputs[strength] = output
            
            # Extract score
            score = extract_first_number(output)
            if score is not None:
                scores[strength] = score
        
        # Reset model control after testing
        control_model.reset()
        
        return {
            'method': method,
            'layer_zones': layer_zones,
            'scores': scores,
            'outputs': outputs,
            'success': True
        }
        
    except Exception as e:
        print(f"Error in combination {combo_idx+1}: {e}")
        return {
            'method': method,
            'layer_zones': layer_zones,
            'scores': {},
            'outputs': {},
            'success': False,
            'error': str(e)
        }


# Create directories
os.makedirs("./plots/grid_search", exist_ok=True)
os.makedirs("./tensorboard_logs", exist_ok=True)

# Initialize tensorboard writer
writer = SummaryWriter("./tensorboard_logs/grid_search")

# Grid search
grid = ParameterGrid(param_grid)
total_combinations = len(grid)
all_results = []

print(f"Starting grid search with {total_combinations} combinations...")

for i, params in enumerate(grid):
    method = params['method']
    layer_zones = params['layer_zones']
    
    # Test this configuration
    result = test_configuration(method, layer_zones, i, total_combinations)
    all_results.append(result)
    
    if result['success'] and result['scores']:
        scores = result['scores']
        
        # Log individual points to tensorboard
        zones_tag = format_layer_zones_for_filename(layer_zones)
        for strength, score in scores.items():
            writer.add_scalar(
                f"{method}/zones_{zones_tag}/iq_score",
                score,
                strength
            )
        
        # Create plot for this combination
        plt.figure(figsize=(12, 8))
        strengths_list = sorted(scores.keys())
        scores_list = [scores[s] for s in strengths_list]
        
        plt.plot(strengths_list, scores_list, "bo-", linewidth=2, markersize=6)
        plt.xlabel("Control Strength", fontsize=12)
        plt.ylabel("Extracted IQ Score", fontsize=12)
        plt.title(
            f"IQ Score vs Control Strength\n"
            f"Method: {method}, Layer zones: {layer_zones}\n"
            f"Model: {model_name}",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        
        plt.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="Average IQ (100)")
        plt.axvline(x=0, color="g", linestyle="--", alpha=0.5, label="No Control (0)")
        
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"./plots/grid_search/iq_score_{method}_{zones_tag}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close to save memory
        
        print(f"Plot saved: {plot_filename}")
        
        # Log summary statistics to tensorboard
        if scores_list:
            mean_score = sum(scores_list) / len(scores_list)
            max_score = max(scores_list)
            min_score = min(scores_list)
            score_range = max_score - min_score
            
            # Use combination index as the x-axis for summary stats
            writer.add_scalar(f"summary/{method}_zones_{zones_tag}/mean_score", mean_score, i)
            writer.add_scalar(f"summary/{method}_zones_{zones_tag}/max_score", max_score, i)
            writer.add_scalar(f"summary/{method}_zones_{zones_tag}/min_score", min_score, i)
            writer.add_scalar(f"summary/{method}_zones_{zones_tag}/score_range", score_range, i)
            
            # Also log by method for comparison across layer zones
            writer.add_scalar(f"by_method/{method}/mean_score", mean_score, i)
            writer.add_scalar(f"by_method/{method}/max_score", max_score, i)
            writer.add_scalar(f"by_method/{method}/score_range", score_range, i)
    
    print(f"Completed combination {i+1}/{total_combinations}")

# Log final summary
successful_runs = [r for r in all_results if r['success']]
print(f"\nGrid search completed!")
print(f"Successful runs: {len(successful_runs)}/{total_combinations}")

# Create summary report
summary_file = "./plots/grid_search/summary_report.txt"
with open(summary_file, 'w') as f:
    f.write(f"Grid Search Summary Report\n")
    f.write(f"==========================\n\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Total combinations tested: {total_combinations}\n")
    f.write(f"Successful runs: {len(successful_runs)}\n\n")
    
    f.write("Results by combination:\n")
    for i, result in enumerate(all_results):
        f.write(f"\nCombination {i+1}:\n")
        f.write(f"  Method: {result['method']}\n")
        f.write(f"  Layer zones: {result['layer_zones']}\n")
        f.write(f"  Success: {result['success']}\n")
        if result['success']:
            scores = result['scores']
            if scores:
                scores_list = list(scores.values())
                f.write(f"  Scores found: {len(scores)}/{len(strengths)}\n")
                f.write(f"  Mean score: {sum(scores_list)/len(scores_list):.2f}\n")
                f.write(f"  Score range: {min(scores_list):.2f} - {max(scores_list):.2f}\n")
            else:
                f.write(f"  No valid scores extracted\n")
        else:
            f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

writer.close()
print(f"Summary report saved to: {summary_file}")
print("Results logged to tensorboard. Run: tensorboard --logdir=./tensorboard_logs")
