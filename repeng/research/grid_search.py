# Grid search script version for tracking experiments
grid_search_script_version = "1.0.0"

from pprint import pprint
import re
import os
import math
import torch

# Set matplotlib backend before importing pyplot to ensure non-interactive plotting
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for file output
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
from TaguchiGridSearchConverter import TaguchiGridSearchConverter
from scipy.stats import pearsonr

from repeng import (
    ControlVector,
    ControlModel,
    DatasetEntry,
    __VERSION__ as repeng_version,
)
from repeng.research import datasets

from sklearnex import patch_sklearn
from tqdm import tqdm

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""


patch_sklearn()

USE_TAGUCHI_REDUCTION = True

# Quantization config
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Define parameter grid for comprehensive search
param_grid = {
    "model_name": [
        "qwen/qwen3-4b",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-3-4b-it",
    ],
    # "method": ["mean", "median"],
    "method": ["median", "mean", "pca_diff", "pca_center", "ica_diff", "ica_center", "umap", "umap_densmap"],
    "dataset": ["age", "iq"],
    "layer_zones": [
        # by increments of 0.1
        [[0.0, 0.1]],
        [[0.1, 0.2]],
        [[0.2, 0.3]],
        [[0.3, 0.4]],
        [[0.4, 0.5]],
        [[0.5, 0.6]],
        [[0.6, 0.7]],
        [[0.7, 0.8]],
        [[0.8, 0.9]],
        [[0.9, 1.0]],
        # by increments of 0.2
        [[0.0, 0.2]],
        [[0.1, 0.3]],
        [[0.2, 0.4]],
        [[0.3, 0.5]],
        [[0.4, 0.6]],
        [[0.5, 0.7]],
        [[0.6, 0.8]],
        [[0.7, 0.9]],
        [[0.8, 1.0]],
        # by increments of 0.3
        [[0.0, 0.3]],
        [[0.1, 0.4]],
        [[0.2, 0.5]],
        [[0.3, 0.6]],
        [[0.4, 0.7]],
        [[0.5, 0.8]],
        [[0.6, 0.9]],
        [[0.7, 1.0]],
        # by increments of 0.4
        [[0.0, 0.4]],
        [[0.1, 0.5]],
        [[0.2, 0.6]],
        [[0.3, 0.7]],
        [[0.4, 0.8]],
        [[0.5, 0.9]],
        [[0.6, 1.0]],
        [[0.2, 0.8]],  # most layers
        [[0.1, 0.9]],  # most layers
        [[0.0, 1.0]],  # all layers
        [[0.1, 0.3], [0.6, 0.8]],
        [[0.2, 0.51], [0.7, 0.9]],  # multiple zones
    ],
}

# Strengths to test (same as first.py)
strengths = [
    -5,
    -4,
    -3,
    -2,
    -1.5,
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
    1.5,
    2,
    3,
    4,
    5,
]


def get_data(dataset: str, tokenizer) -> tuple[str, list]:
    """
    Get scenario and dataset based on dataset name.

    Parameters
    ----------
    dataset : str
        Dataset name ("age", etc.)

    Returns
    -------
    tuple[str, list]
        Tuple containing (scenario, dataset_entries)
    """
    if dataset == "age":
        scenario = tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "user",
                    "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, how old is this human?",
                },
                {
                    "role": "assistant",
                    "content": "Alright. I got it. The age of the human I pick is exactly ",
                },
            ],
            continue_final_message=True,
            tokenize=False,
        )
        return scenario, datasets.young_old_paragraph
    elif dataset == "iq":
        scenario = tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "user",
                    "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, what is the IQ score of this human?",
                },
                {
                    "role": "assistant",
                    "content": "Alright. I got it. The IQ of the human I pick is exactly ",
                },
            ],
            continue_final_message=True,
            tokenize=False,
        )
        return scenario, datasets.dumb_genius_paragraph
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def extract_first_number(text: str) -> float | None:
    """Extract the first number from text using regex."""
    lines = text.splitlines()
    lines = [
        li
        for li in lines
        if not (
            # gpt oss
            li.startswith("Knowledge cutoff: ")
            or li.startswith("Current date: ")
            # llama
            or li.startswith("Cutting Knowledge Date: ")
            or li.startswith("Today Date: ")
        )
    ]
    text = "\n".join(lines)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None


def format_layer_zones_for_filename(layer_zones: list) -> str:
    """Format layer zones for use in filenames."""
    zones_str = (
        str(layer_zones)
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "_")
        .replace(".", "")
    )
    return zones_str


def test_configuration(
    model_name: str,
    method: str,
    layer_zones: list,
    dataset: str,
    combo_idx: int,
    total_combos: int,
) -> dict:
    """Test a single configuration and return results."""
    print(f"\n=== Combination {combo_idx+1}/{total_combos} ===")
    print(f"Model: {model_name}")
    print(f"Method: {method}")
    print(f"Dataset: {dataset}")
    print(f"Layer zones: {layer_zones}")

    # Load model and tokenizer for this configuration
    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if not "gemma" in model_name.lower() else None,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Get scenario and dataset for this configuration
    scenario, train_dataset = get_data(dataset, tokenizer)

    # Create unique writer for this combination
    zones_tag = format_layer_zones_for_filename(layer_zones)
    model_tag = model_name.replace("/", "_").replace("-", "_")
    run_name = f"{model_tag}_{dataset}_{method}_zones_{zones_tag}"
    writer = SummaryWriter(f"./tensorboard_logs/grid_search/{run_name}")

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
            train_dataset,
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

            # Print the actual LLM output to screen
            print(f"  Output: {output}")

            # Log the output text to tensorboard
            zones_tag = format_layer_zones_for_filename(layer_zones)
            model_tag = model_name.replace("/", "_").replace("-", "_")
            writer.add_text(
                f"{model_tag}_{dataset}_{method}/zones_{zones_tag}/outputs",
                f"Strength {strength}: {output}",
                global_step=strength,
            )

            # Extract score
            score = extract_first_number(output)
            if score is not None:
                scores[strength] = score
                print(f"  Extracted score: {score}")
            else:
                scores[strength] = float("nan")
                print(f"  No score found in output, treating as NA")

        # Create plot for this combination if we have valid scores
        # Filter out NaN values for plotting
        valid_data = [
            (s, scores[s]) for s in sorted(scores.keys()) if not math.isnan(scores[s])
        ]

        if valid_data:
            print(f"  Creating plot with {len(valid_data)} valid data points")
            strengths_list, scores_list = zip(*valid_data)

            # Debug: Print the data being plotted
            print(f"  Plotting strengths: {strengths_list}")
            print(f"  Plotting scores: {scores_list}")

            # Create the figure
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(strengths_list, scores_list, "bo-", linewidth=2, markersize=6)
            ax.set_xlabel("Control Strength", fontsize=12)
            ax.set_ylabel("Extracted Value", fontsize=12)
            ax.set_title(
                f"Extracted value vs Control Strength ({dataset} dataset)\n"
                f"Method: {method}, Layer zones: {layer_zones}\n"
                f"Model: {model_name}",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3)

            # Add dataset-specific reference lines and y-axis limits
            if dataset == "iq":
                ax.axhline(
                    y=100,
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label="Average IQ (100)",
                )
                ax.set_ylim(0, 200)  # IQ range from 0 to 200
            elif dataset == "age":
                ax.axhline(y=25, color="r", linestyle="--", alpha=0.5, label="Ref(25)")
                ax.set_ylim(0, 150)  # Age range from 0 to 150 years

            ax.legend()
            plt.tight_layout()

            # Explicitly draw the figure to ensure it's rendered
            fig.canvas.draw()

            # Log plot to tensorboard
            try:
                writer.add_figure(
                    "extracted_value_score_plot",
                    fig,
                    global_step=0,
                )
                print(f"  Plot successfully logged to TensorBoard")
            except Exception as e:
                print(f"  Error logging plot to TensorBoard: {e}")

            # Save plot
            zones_tag = format_layer_zones_for_filename(layer_zones)
            model_tag = model_name.replace("/", "_").replace("-", "_")
            plot_filename = f"./plots/grid_search/extracted_value_{model_tag}_{dataset}_{method}_{zones_tag}.png"
            try:
                fig.savefig(
                    plot_filename, dpi=300, bbox_inches="tight", facecolor="white"
                )
                print(f"  Plot saved: {plot_filename}")

                # Check if file was actually created and has content
                if os.path.exists(plot_filename):
                    file_size = os.path.getsize(plot_filename)
                    print(f"  Plot file size: {file_size} bytes")
                else:
                    print(f"  Warning: Plot file was not created!")
            except Exception as e:
                print(f"  Error saving plot: {e}")

            plt.close(fig)  # Close the specific figure to save memory
        else:
            print(
                f"  No valid scores to plot for this combination - all values are NaN"
            )

        # Reset model control and unwrap to restore original state
        control_model.reset()
        control_model.unwrap()

        # Close the writer for this combination
        writer.close()

        return {
            "model_name": model_name,
            "method": method,
            "dataset": dataset,
            "layer_zones": layer_zones,
            "scores": scores,
            "outputs": outputs,
            "success": True,
        }

    except Exception as e:
        print(f"Error in combination {combo_idx+1}: {e}")
        # Close the writer even on error
        writer.close()
        return {
            "model_name": model_name,
            "method": method,
            "dataset": dataset,
            "layer_zones": layer_zones,
            "scores": {},
            "outputs": {},
            "success": False,
            "error": str(e),
        }


# Create directories
os.makedirs("./plots/grid_search", exist_ok=True)
os.makedirs("./tensorboard_logs", exist_ok=True)

# Create main writer for overall grid search logging
main_writer = SummaryWriter(f"./tensorboard_logs/grid_search/main")

# Log version information as metadata
main_writer.add_text(
    "metadata/grid_search_script_version", grid_search_script_version, 0
)
main_writer.add_text("metadata/repeng_version", repeng_version, 0)
# Note: model_name is now part of the grid and logged per combination

# Grid search
grid = ParameterGrid(param_grid)

if USE_TAGUCHI_REDUCTION:
    # Use taguchi arrays to reduce the size of the grid
    converter = TaguchiGridSearchConverter()
    old_grid = grid
    grid = converter.fit_transform(old_grid)
    assert len(grid) <= len(old_grid)
    total_combinations = len(grid)

    print(
        f"Starting grid search with {total_combinations} combinations (before taguchi: {len(old_grid)}..."
    )
else:
    total_combinations = len(grid)
    print(
        f"Starting grid search with {total_combinations} combinations (no taguchi reduction)"
    )

all_results = []

for i, params in enumerate(tqdm(grid, desc="Grid Search Progress", colour="green")):
    model_name = params["model_name"]
    method = params["method"]
    dataset = params["dataset"]
    layer_zones = params["layer_zones"]

    # Test this configuration
    result = test_configuration(
        model_name, method, layer_zones, dataset, i, total_combinations
    )
    all_results.append(result)

    if result["success"] and result["scores"]:
        scores = result["scores"]

        # Log individual points to tensorboard (only valid scores, no NaN values)
        zones_tag = format_layer_zones_for_filename(layer_zones)
        model_tag = model_name.replace("/", "_").replace("-", "_")
        for strength, score in scores.items():
            if not math.isnan(score):
                main_writer.add_scalar(
                    f"{model_tag}_{dataset}_{method}/zones_{zones_tag}/extracted_value",
                    score,
                    strength,
                )

        # Log summary statistics to tensorboard
        if result["success"] and result["scores"]:
            scores = result["scores"]
            # Filter out NaN values for statistics
            valid_data = [
                (s, scores[s])
                for s in sorted(scores.keys())
                if not math.isnan(scores[s])
            ]

            if valid_data:
                strengths_list, scores_list = zip(*valid_data)
            mean_score = sum(scores_list) / len(scores_list)
            max_score = max(scores_list)
            min_score = min(scores_list)
            score_range = max_score - min_score

            # Calculate correlation between control strength and extracted value
            correlation_coeff = 0.0
            correlation_p_value = 1.0
            try:
                if len(strengths_list) > 1 and len(scores_list) > 1:
                    correlation_coeff, correlation_p_value = pearsonr(
                        strengths_list, scores_list
                    )
                    print(
                        f"  Correlation coefficient: {correlation_coeff:.4f} (p-value: {correlation_p_value:.4f})"
                    )
            except Exception as e:
                print(f"  Error calculating correlation: {e}")
                correlation_coeff = 0.0
                correlation_p_value = 1.0

            # Log hyperparameters and metrics for easy filtering
            hparam_dict = {
                "model_name": model_name,
                "method": method,
                "dataset": dataset,
                "layer_zones_str": str(
                    layer_zones
                ),  # String representation for filtering
                "num_layer_zones": len(layer_zones),  # Number of zone pairs
                "combo_idx": i,  # Unique identifier for this combination
                "grid_search_script_version": grid_search_script_version,
                "repeng_version": repeng_version,
            }

            # Add individual zone boundaries as separate hyperparameters for easier filtering
            for zone_idx, zone in enumerate(layer_zones):
                hparam_dict[f"zone_{zone_idx}_start"] = zone[0]
                hparam_dict[f"zone_{zone_idx}_end"] = zone[1]
                hparam_dict[f"zone_{zone_idx}_width"] = zone[1] - zone[0]

            metric_dict = {
                "hparam/mean_score": mean_score,
                "hparam/max_score": max_score,
                "hparam/min_score": min_score,
                "hparam/score_range": score_range,
                "hparam/correlation_coeff": correlation_coeff,
                "hparam/correlation_p_value": correlation_p_value,
                "hparam/num_valid_scores": len(scores_list),
            }

            # Log hyperparameters with metrics - this allows filtering in TensorBoard
            main_writer.add_hparams(hparam_dict, metric_dict)

            # Also log individual parameters as scalars for time-series analysis
            main_writer.add_scalar("params/combo_idx", i, i)
            main_writer.add_scalar("params/num_layer_zones", len(layer_zones), i)
            for zone_idx, zone in enumerate(layer_zones):
                main_writer.add_scalar(f"params/zone_{zone_idx}_start", zone[0], i)
                main_writer.add_scalar(f"params/zone_{zone_idx}_end", zone[1], i)
                main_writer.add_scalar(
                    f"params/zone_{zone_idx}_width", zone[1] - zone[0], i
                )

            # Use combination index as the x-axis for summary stats
            model_tag = model_name.replace("/", "_").replace("-", "_")
            main_writer.add_scalar(
                f"summary/{model_tag}_{dataset}_{method}_zones_{zones_tag}/mean_score",
                mean_score,
                i,
            )
            main_writer.add_scalar(
                f"summary/{model_tag}_{dataset}_{method}_zones_{zones_tag}/max_score",
                max_score,
                i,
            )
            main_writer.add_scalar(
                f"summary/{model_tag}_{dataset}_{method}_zones_{zones_tag}/min_score",
                min_score,
                i,
            )
            main_writer.add_scalar(
                f"summary/{model_tag}_{dataset}_{method}_zones_{zones_tag}/score_range",
                score_range,
                i,
            )
            main_writer.add_scalar(
                f"summary/{model_tag}_{dataset}_{method}_zones_{zones_tag}/correlation_coeff",
                correlation_coeff,
                i,
            )
            main_writer.add_scalar(
                f"summary/{model_tag}_{dataset}_{method}_zones_{zones_tag}/correlation_p_value",
                correlation_p_value,
                i,
            )

            # Also log by method for comparison across layer zones
            main_writer.add_scalar(
                f"by_method/{model_tag}_{dataset}_{method}/mean_score", mean_score, i
            )
            main_writer.add_scalar(
                f"by_method/{model_tag}_{dataset}_{method}/max_score", max_score, i
            )
            main_writer.add_scalar(
                f"by_method/{model_tag}_{dataset}_{method}/score_range", score_range, i
            )
            main_writer.add_scalar(
                f"by_method/{model_tag}_{dataset}_{method}/correlation_coeff",
                correlation_coeff,
                i,
            )

    print(f"Completed combination {i+1}/{total_combinations}")

# Log final summary
successful_runs = [r for r in all_results if r["success"]]
print(f"\nGrid search completed!")
print(f"Successful runs: {len(successful_runs)}/{total_combinations}")

# Create summary report
summary_file = "./plots/grid_search/summary_report.txt"
with open(summary_file, "w") as f:
    f.write(f"Grid Search Summary Report\n")
    f.write(f"==========================\n\n")
    f.write(f"Total combinations tested: {total_combinations}\n")
    f.write(f"Successful runs: {len(successful_runs)}\n\n")

    # Group results by model for easier comparison
    models_tested = set(r.get("model_name", "unknown") for r in all_results)
    f.write(f"Models tested: {', '.join(sorted(models_tested))}\n\n")

    f.write("Results by combination:\n")
    for i, result in enumerate(all_results):
        f.write(f"\nCombination {i+1}:\n")
        f.write(f"  Model: {result.get('model_name', 'unknown')}\n")
        f.write(f"  Method: {result['method']}\n")
        f.write(f"  Dataset: {result['dataset']}\n")
        f.write(f"  Layer zones: {result['layer_zones']}\n")
        f.write(f"  Success: {result['success']}\n")
        if result["success"]:
            scores = result["scores"]
            if scores:
                scores_list = list(scores.values())
                strengths_list = list(scores.keys())
                f.write(f"  Scores found: {len(scores)}/{len(strengths)}\n")
                f.write(f"  Mean score: {sum(scores_list)/len(scores_list):.2f}\n")
                f.write(
                    f"  Score range: {min(scores_list):.2f} - {max(scores_list):.2f}\n"
                )

                # Calculate and report correlation
                try:
                    if len(strengths_list) > 1 and len(scores_list) > 1:
                        correlation_coeff, correlation_p_value = pearsonr(
                            strengths_list, scores_list
                        )
                        f.write(
                            f"  Correlation coefficient: {correlation_coeff:.4f} (p-value: {correlation_p_value:.4f})\n"
                        )
                    else:
                        f.write(f"  Correlation coefficient: N/A (insufficient data)\n")
                except Exception as e:
                    f.write(f"  Correlation coefficient: Error - {e}\n")
            else:
                f.write(f"  No valid scores extracted\n")
        else:
            f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

# Close the main writer
main_writer.close()

print(f"Summary report saved to: {summary_file}")
print("Results logged to tensorboard. Run: tensorboard --logdir=./tensorboard_logs")
