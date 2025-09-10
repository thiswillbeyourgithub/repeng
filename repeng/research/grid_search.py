from typing import List, Dict
from fire import Fire
import re
import os
import math
import torch
import gc
from loguru import logger

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
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


patch_sklearn()

# Configure loguru to write to file
os.makedirs("./logs", exist_ok=True)
logger.add(
    "./logs/grid_search.logs", rotation="10 MB", retention="10 days", level="INFO"
)


from transformers import BitsAndBytesConfig

# Configure quantization for models that support it
# source: https://huggingface.co/docs/transformers/quantization/bitsandbytes
quant_config = BitsAndBytesConfig(
    device_map="auto",
    load_in_4bit=True,
    # load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # allow offloading between gpu and cpu, only for 8bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # faster computation
)

# Define parameter grid for comprehensive search
param_grid = {
    "model_name": [
        "qwen/qwen3-4b",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-7b-it",
    ],
    # "method": ["mean", "median"],
    "method": [
        "median",
        "mean",
        "pca_diff",
        "pca_center",
        "ica_diff",
        "ica_center",
        "umap",
        "umap_densmap",
    ],
    # "dataset": ["age", "iq"],
    "dataset": ["age"],
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
    debug: bool = False,
    batch_size: int = 1,
) -> dict:
    """Test a single configuration and return results."""
    logger.info(f"\n=== Combination {combo_idx+1}/{total_combos} ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Layer zones: {layer_zones}")

    # Load model and tokenizer for this configuration
    logger.info("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
        logger.info("Training control vector...")
        trained_vector = ControlVector.train(
            control_model,
            tokenizer,
            train_dataset,
            batch_size=batch_size,
            method=method,
            cache_path="./model_cache",
        )

        # Test all strengths
        scores = {}
        outputs = {}

        for strength in strengths:
            logger.info(f"Testing strength: {strength}")
            control_model.set_control(trained_vector, strength)

            out = control_model.generate(
                **tokenizer(scenario, return_tensors="pt").to(control_model.device),
                do_sample=False,
                max_new_tokens=50,
                repetition_penalty=1.1,
            )

            output = tokenizer.decode(out.squeeze(), skip_special_tokens=True).strip()
            outputs[strength] = output

            # logger.info the actual LLM output to screen
            logger.info(f"  Output: {output}")

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
                logger.info(f"  Extracted score: {score}")

                # Log individual data point to TensorBoard for native plotting
                # This creates an interactive plot for this specific configuration
                writer.add_scalar(
                    "extracted_value_vs_strength",
                    score,
                    global_step=int(
                        strength * 100
                    ),  # Convert to int, scale by 100 for precision
                )
            else:
                scores[strength] = float("nan")
                logger.info(f"  No score found in output, treating as NA")

        # Create plot for this combination if we have valid scores
        # Filter out NaN values for plotting
        valid_data = [
            (s, scores[s]) for s in sorted(scores.keys()) if not math.isnan(scores[s])
        ]

        if valid_data:
            logger.info(f"  Creating plot with {len(valid_data)} valid data points")
            strengths_list, scores_list = zip(*valid_data)

            # Debug: logger.info the data being plotted
            logger.info(f"  Plotting strengths: {strengths_list}")
            logger.info(f"  Plotting scores: {scores_list}")

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
                ax.set_ylim(0, 300)  # Age range from 0 to 300 years

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
                logger.info(f"  Plot successfully logged to TensorBoard")
            except Exception as e:
                logger.info(f"  Error logging plot to TensorBoard: {e}")
                if debug:
                    raise

            # Save plot
            zones_tag = format_layer_zones_for_filename(layer_zones)
            model_tag = model_name.replace("/", "_").replace("-", "_")
            plot_filename = f"./plots/grid_search/extracted_value_{model_tag}_{dataset}_{method}_{zones_tag}.png"
            try:
                fig.savefig(
                    plot_filename, dpi=300, bbox_inches="tight", facecolor="white"
                )
                logger.info(f"  Plot saved: {plot_filename}")

                # Check if file was actually created and has content
                if os.path.exists(plot_filename):
                    file_size = os.path.getsize(plot_filename)
                    logger.info(f"  Plot file size: {file_size} bytes")
                else:
                    logger.info(f"  Warning: Plot file was not created!")
            except Exception as e:
                logger.info(f"  Error saving plot: {e}")
                if debug:
                    raise

            plt.close(fig)  # Close the specific figure to save memory
        else:
            logger.info(
                f"  No valid scores to plot for this combination - all values are NaN"
            )

        # Reset model control and unwrap to restore original state
        control_model.reset()
        unwrapped_model = control_model.unwrap()

        # Close the writer for this combination
        writer.close()

        # Explicitly delete all model references to free GPU memory
        del trained_vector, control_model, unwrapped_model, base_model

        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
        logger.info(f"Error in combination {combo_idx+1}: {e}")
        # Close the writer even on error
        writer.close()
        if debug:
            raise
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
os.makedirs("./logs", exist_ok=True)

# Grid search script version for tracking experiments
grid_search_script_version = "1.0.0"


def main(debug: bool = False, taguchi_reduction: bool = False, batch_size: int = 1):
    """
    Run comprehensive grid search over control vector configurations.

    This function performs an exhaustive evaluation of different control vector
    training methods, layer zones, datasets, and models to find optimal configurations
    for representation engineering. Results are logged to TensorBoard for analysis
    and visualization.

    The grid search tests each combination by:
    1. Loading a model and creating a ControlModel wrapper
    2. Training a control vector using the specified method and layer zones
    3. Testing the vector at various strength coefficients
    4. Extracting numerical values from model outputs
    5. Computing correlation between strength and extracted values
    6. Logging results and creating plots

    Parameters
    ----------
    debug : bool, default=False
        If True, raises exceptions instead of logging them and continuing.
        Useful for debugging specific configuration failures.
    taguchi_reduction : bool, default=False
        If True, uses Taguchi orthogonal arrays to reduce the parameter grid size
        while maintaining good coverage of the parameter space. This significantly
        reduces computational cost but may miss some parameter interactions.

    Notes
    -----
    The function creates several output directories:
    - ./plots/grid_search/ : Individual plots for each configuration
    - ./tensorboard_logs/grid_search/ : TensorBoard logs for analysis
    - ./logs/ : Text logs from loguru

    GPU memory is explicitly managed by deleting model references and calling
    torch.cuda.empty_cache() after each configuration to prevent OOM errors.

    Results include correlation analysis between control strength and extracted
    values, which helps identify effective control directions.
    """
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

    if taguchi_reduction:
        # Use taguchi arrays to reduce the size of the grid
        converter = TaguchiGridSearchConverter()
        old_grid = grid
        grid: List[Dict] = converter.fit_transform(old_grid)
        assert len(grid) <= len(old_grid)
        total_combinations = len(grid)

        logger.info(
            f"Starting grid search with {total_combinations} combinations (before taguchi: {len(old_grid)})"
        )
    else:
        total_combinations = len(grid)
        logger.info(
            f"Starting grid search with {total_combinations} combinations (no taguchi reduction)"
        )

    # sort the grid to make sure that we switch model as little as possible
    grid = list(grid)
    grid = sorted(grid, key=lambda dictparam: dictparam["model_name"])

    all_results = []

    for i, params in enumerate(tqdm(grid, desc="Grid Search Progress", colour="green")):
        model_name = params["model_name"]
        method = params["method"]
        dataset = params["dataset"]
        layer_zones = params["layer_zones"]

        # Test this configuration
        result = test_configuration(
            model_name,
            method,
            layer_zones,
            dataset,
            i,
            total_combinations,
            debug,
            batch_size,
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
                        logger.info(
                            f"  Correlation coefficient: {correlation_coeff:.4f} (p-value: {correlation_p_value:.4f})"
                        )
                except Exception as e:
                    logger.info(f"  Error calculating correlation: {e}")
                    if debug:
                        raise
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
                    f"by_method/{model_tag}_{dataset}_{method}/mean_score",
                    mean_score,
                    i,
                )
                main_writer.add_scalar(
                    f"by_method/{model_tag}_{dataset}_{method}/max_score", max_score, i
                )
                main_writer.add_scalar(
                    f"by_method/{model_tag}_{dataset}_{method}/score_range",
                    score_range,
                    i,
                )
                main_writer.add_scalar(
                    f"by_method/{model_tag}_{dataset}_{method}/correlation_coeff",
                    correlation_coeff,
                    i,
                )

        logger.info(f"Completed combination {i+1}/{total_combinations}")

    # Log final summary
    successful_runs = [r for r in all_results if r["success"]]
    logger.info(f"\nGrid search completed!")
    logger.info(f"Successful runs: {len(successful_runs)}/{total_combinations}")

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
                            f.write(
                                f"  Correlation coefficient: N/A (insufficient data)\n"
                            )
                    except Exception as e:
                        f.write(f"  Correlation coefficient: Error - {e}\n")
                        if debug:
                            raise
                else:
                    f.write(f"  No valid scores extracted\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

    # Close the main writer
    main_writer.close()

    logger.info(f"Summary report saved to: {summary_file}")
    logger.info(
        "Results logged to tensorboard. Run: tensorboard --logdir=./tensorboard_logs"
    )


if __name__ == "__main__":
    Fire(main)
