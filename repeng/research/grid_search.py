from typing import List, Dict
from fire import Fire
import re
import os
import math
import shutil
import torch
import torch.nn.functional as F
import gc
from loguru import logger

# Set matplotlib backend before importing pyplot to ensure non-interactive plotting
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for file output
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
from GridSearchReductor import GridSearchReductor
from scipy.stats import pearsonr

from repeng import (
    ControlVector,
    ControlModel,
    __VERSION__ as repeng_version,
)
from repeng.research.shared import (
    get_data,
    find_matching_token_ids,
)

from sklearnex import patch_sklearn
from tqdm import tqdm

# Import shared strength configuration to ensure consistency across experiments
from repeng.research.shared import (
    DEFAULT_STRENGTHS,
    FINE_GRAINED_STRENGTHS,
    SHORT_TEST_STRENGTHS,
)


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
    device_map="cuda",
    load_in_4bit=True,
    # load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # allow offloading between gpu and cpu, only for 8bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # faster computation
)

# Define parameter grid for comprehensive search
param_grid = {
    "model_name": [
        # "qwen/qwen3-4b",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
        # "google/gemma-7b-it",
    ],
    # "method": ["mean", "median"],
    "method": [
        "median",
        "mean",
        "pca_diff",
        "pca_center",
        # "ica_diff",
        # "ica_center",
        "umap",
        "umap_densmap",
        "pcaw_svd",
        "pcaw_eigen",
    ],
    # "dataset": ["age", "iq"],
    "dataset": ["age"],
    "normalize": [True, False],
    "rescaling": [False, "layer_magnitude"],
    "enable_thinking": [True],
    "layer_zones": [
        # # by increments of 0.1
        # [[0.0, 0.1]],
        # [[0.1, 0.2]],
        # [[0.2, 0.3]],
        # [[0.3, 0.4]],
        # [[0.4, 0.5]],
        # [[0.5, 0.6]],
        # [[0.6, 0.7]],
        # [[0.7, 0.8]],
        # [[0.8, 0.9]],
        # [[0.9, 1.0]],
        # # by increments of 0.2
        # [[0.0, 0.2]],
        # [[0.1, 0.3]],
        # [[0.2, 0.4]],
        # [[0.3, 0.5]],
        # [[0.4, 0.6]],
        # [[0.5, 0.7]],
        # [[0.6, 0.8]],
        # [[0.7, 0.9]],
        # [[0.8, 1.0]],
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
        # # most layers
        # [[0.2, 0.8]],
        # # most layers
        # [[0.1, 0.9]],
        # # all layers
        # [[0.0, 1.0]],
        # [[0.1, 0.3], [0.6, 0.8]],
        # [[0.2, 0.51], [0.7, 0.9]],  # multiple zones
    ],
}

# Strengths to test (same as playground.py but using standard range)
strengths = FINE_GRAINED_STRENGTHS
# strengths = SHORT_TEST_STRENGTHS

# check that with the way we turn strengths into global_step for tensorboard we don't have collisions, adjust the multiplier if needed
assert len(strengths) == len(set(strengths)), "Found duplicate elements of strengths"
strengths_multiplier_tensorboard = 10
while True:
    _stren = [int(s * strengths_multiplier_tensorboard) for s in strengths]
    if len(_stren) == len(set(_stren)):
        break
    else:
        strengths_multiplier_tensorboard *= 10


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
    normalize: bool,
    rescaling: str | None,
    enable_thinking: bool,
    combo_idx: int,
    total_combos: int,
    debug: bool = False,
    batch_size: int = 1,
    logged_training_logprobs: set | None = None,
) -> dict:
    """Test a single configuration and return results."""

    # ============================================================================
    # PHASE 1: INITIAL SETUP AND LOGGING
    # ============================================================================
    # Log the current configuration parameters for debugging and tracking.
    # This helps identify which specific combination is being tested when
    # reviewing logs or debugging failures.
    logger.info(f"\n=== Combination {combo_idx+1}/{total_combos} ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Layer zones: {layer_zones}")
    logger.info(f"Normalize: {normalize}")
    logger.info(f"Rescaling: {rescaling}")
    logger.info(f"Enable thinking: {enable_thinking}")

    # Clear GPU memory before starting to prevent OOM errors from previous runs.
    # This is crucial when testing many configurations sequentially.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # ============================================================================
    # PHASE 2: MODEL AND TOKENIZER LOADING
    # ============================================================================
    # Load the base language model and tokenizer. This includes retry logic
    # to handle CUDA memory issues that may occur during loading.
    logger.info("Loading model and tokenizer...")

    # Retry logic for model loading with CUDA error handling
    max_retries = 10 if debug else 1
    base_model = None

    for attempt in range(max_retries):
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            logger.info(f"Model loaded successfully on attempt {attempt + 1}")
            break  # Success, exit retry loop

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                logger.info(f"CUDA error on attempt {attempt + 1}/{max_retries}: {e}")
                if debug:
                    logger.info("Entering breakpoint to allow freeing GPU memory...")
                    breakpoint()  # Allow user to free GPU memory and continue
                    # After resuming from breakpoint, force cleanup before retry
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                else:
                    # If not in debug mode, just raise the error immediately
                    raise
            else:
                # Non-CUDA error, re-raise immediately
                raise
        except Exception as e:
            # Any other error, re-raise immediately
            logger.info(f"Non-CUDA error during model loading: {e}")
            raise

    if base_model is None:
        raise RuntimeError(f"Failed to load model after {max_retries} attempts")

    # ============================================================================
    # PHASE 3: TOKENIZER SETUP AND DATASET PREPARATION
    # ============================================================================
    # Configure the tokenizer and prepare the dataset for this specific test.
    # The tokenizer needs a pad_token for batched operations, and we need to
    # extract the conversation template and target tokens for evaluation.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Extract dataset-specific conversation template, training data, and evaluation targets.
    # This determines what the model will be asked to do and how we'll measure success.
    conversation, train_dataset, target_tokens = get_data(dataset)
    # Create the base scenario prompt from the conversation template.
    # This will be used for testing the control vector at different strengths.
    scenario = tokenizer.apply_chat_template(
        conversation=conversation,
        continue_final_message=(
            True if conversation[-1]["role"] == "assistant" else False
        ),
        tokenize=False,
        enable_thinking=enable_thinking,
    )

    # ============================================================================
    # PHASE 4: TENSORBOARD WRITER SETUP
    # ============================================================================
    # Create a unique TensorBoard writer for this specific configuration.
    # This allows us to track and visualize results for each parameter combination.
    zones_tag = format_layer_zones_for_filename(layer_zones)
    model_tag = model_name.replace("/", "_").replace("-", "_")
    normalize_tag = "norm" if normalize else "nonorm"
    rescaling_tag = rescaling if rescaling else "norescale"
    thinking_tag = "thinking" if enable_thinking else "nothinking"
    run_name = f"{model_tag}_{dataset}_{method}_zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}"
    writer = SummaryWriter(f"./tensorboard_logs/grid_search/{run_name}")

    # Initialize correlation variables to ensure they're always defined for return dict
    correlation_coeff = 0.0
    correlation_p_value = 1.0

    try:
        # ========================================================================
        # PHASE 5: CONTROL MODEL CREATION AND VECTOR TRAINING
        # ========================================================================
        # Wrap the base model with ControlModel to enable control vector injection.
        # Then train a control vector using the specified method and dataset.
        # Note: ControlModel mutates the base model, so we work with the same instance
        control_model = ControlModel(
            base_model,
            layer_zones=layer_zones,
        )

        # Train the control vector using the specified method (mean, median, PCA, etc.)
        # This learns the direction in activation space that corresponds to the target concept.
        logger.info("Training control vector...")
        result = ControlVector.train(
            control_model,
            tokenizer,
            train_dataset,
            batch_size=batch_size,
            method=method,
            rescaling=rescaling,
            cache_path="./model_cache",
            output_training_avg_logprob=True,
            enable_thinking=enable_thinking,
        )
        trained_vector, avg_logprobs = result

        # Log training statistics to TensorBoard (only once per model+dataset combination
        # to avoid duplicate logging across different methods/layer_zones)
        if logged_training_logprobs is not None:
            model_dataset_key = f"{model_name}_{dataset}"
            if model_dataset_key not in logged_training_logprobs:
                logger.info(f"Logging training avg logprobs for {model_dataset_key}")
                model_tag = model_name.replace("/", "_").replace("-", "_")
                for sample_idx, avg_logprob in enumerate(avg_logprobs):
                    writer.add_scalar(
                        f"training_logprobs/{model_tag}_{dataset}/avg_logprob_per_sample",
                        avg_logprob,
                        global_step=sample_idx,
                    )
                logged_training_logprobs.add(model_dataset_key)

        # ========================================================================
        # PHASE 6: CONTROL VECTOR EVALUATION AT DIFFERENT STRENGTHS
        # ========================================================================
        # Test the trained control vector at various strength coefficients to see
        # how it affects model behavior. We use a two-stage generation process:
        # Stage 1: Free generation to let the model elaborate on the topic
        # Stage 2: Constrained generation to extract a specific numerical answer
        scores = {}
        logprob_data = {}
        extracted_answers = {}

        for strength in strengths:
            logger.info(f"Testing strength: {strength}")
            # Apply the control vector at the specified strength coefficient.
            # Higher absolute values = stronger control effect.
            control_model.set_control(trained_vector, strength, normalize=normalize)

            # ====================================================================
            # STAGE 1: FREE GENERATION
            # ====================================================================
            # Let the model generate freely in response to the base scenario.
            # This allows it to elaborate on the topic before we ask for a
            # specific numerical answer, which often produces more natural
            # and consistent responses.
            input_ids = tokenizer.encode(scenario, return_tensors="pt")
            if hasattr(control_model, "device"):
                input_ids = input_ids.to(control_model.device)

            # Generate initial response
            logger.info(
                f"  Starting Stage 1: Free generation for strength {strength}..."
            )
            with torch.no_grad():
                initial_generated_ids = control_model.generate(
                    input_ids,
                    max_new_tokens=512,  # Allow longer initial generation
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Extract the freely generated text (excluding the input prompt)
            initial_new_tokens = initial_generated_ids[0][len(input_ids[0]) :]
            initial_generated_text = tokenizer.decode(
                initial_new_tokens, skip_special_tokens=True
            )

            # ====================================================================
            # STAGE 2: CONSTRAINED ANSWER GENERATION
            # ====================================================================
            # Add a conclusion prompt to elicit a specific numerical answer.
            # This two-stage approach helps ensure we get both natural elaboration
            # and a clear extractable answer for evaluation.
            conclusion_prompt = "\nHence, my answer in the A or B format is: "
            extended_conversation = conversation.copy()
            extended_conversation[-1]["content"] += (
                initial_generated_text + conclusion_prompt
            )

            # Apply chat template with continue_final_message=True
            final_scenario = tokenizer.apply_chat_template(
                conversation=extended_conversation,
                continue_final_message=True,
                tokenize=False,
                enable_thinking=False,
            )

            # Generate final answer tokens
            final_input_ids = tokenizer.encode(final_scenario, return_tensors="pt")
            if hasattr(control_model, "device"):
                final_input_ids = final_input_ids.to(control_model.device)

            logger.info(
                f"  Starting Stage 2: Constrained answer generation for strength {strength}..."
            )
            with torch.no_grad():
                final_generated_ids = control_model.generate(
                    final_input_ids,
                    max_new_tokens=5,  # Just a few tokens for the final answer
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Extract the final answer portion (excluding the extended prompt)
            final_new_tokens = final_generated_ids[0][len(final_input_ids[0]) :]
            final_answer_text = tokenizer.decode(
                final_new_tokens, skip_special_tokens=True
            )
            # Combine both generation stages for complete logging
            complete_generated_text = (
                initial_generated_text + conclusion_prompt + final_answer_text
            )
            generated_text = complete_generated_text  # For backward compatibility

            # ====================================================================
            # LOGPROB EXTRACTION AND SCORING
            # ====================================================================
            # Extract log probabilities for target tokens from the final answer.
            # This measures how likely the model was to generate our target
            # tokens (e.g., "25" vs "20" for age questions) under the control.
            if len(final_new_tokens) > 0:
                # Run model on full sequence to get logits for generated positions
                with torch.no_grad():
                    outputs = control_model(final_generated_ids)
                    # Get logits for the generated token positions
                    generated_logits = outputs.logits[
                        0,
                        len(final_input_ids[0])
                        - 1 : len(final_input_ids[0])
                        - 1
                        + len(final_new_tokens),
                        :,
                    ]
                    generated_log_probs = torch.nn.functional.log_softmax(
                        generated_logits, dim=-1
                    )

                # Initialize logprobs dict
                logprobs = {
                    target: generated_log_probs[0, tokenizer.encode(target)[0]].item()
                    for target in target_tokens
                }

                # Check each generated token against our targets
                for i, token_id in enumerate(final_new_tokens):
                    token_text = tokenizer.decode(
                        [token_id], skip_special_tokens=True
                    ).strip()
                    if not token_text:
                        continue
                    # Check if this token matches any of our targets
                    for target in target_tokens:
                        if (
                            target.lower() in token_text.lower()
                            or token_text.lower() in target.lower()
                        ):
                            # Get the logprob of this specific token at this position
                            token_logprob = generated_log_probs[i, token_id].item()
                            # Use the maximum logprob if we find multiple matches
                            if token_logprob > logprobs[target]:
                                logprobs[target] = token_logprob
                                logger.info(
                                    f"  Found target '{target}' in generated token '{token_text}' with logprob {token_logprob:.4f}"
                                )
            else:
                # No final tokens generated
                logprobs = {target: float("-inf") for target in target_tokens}
                logger.info("Warning: No final tokens generated for logprob extraction")

            # Extract answer using regex
            extracted_answer = None
            for target in target_tokens:
                if re.search(
                    rf"\b{re.escape(target)}\b", final_answer_text, re.IGNORECASE
                ):
                    extracted_answer = target
                    break

            # Convert to int if possible, otherwise keep as string, or use -1/"None" if not found
            if extracted_answer is not None:
                try:
                    extracted_answer_value = int(extracted_answer)
                except ValueError:
                    extracted_answer_value = extracted_answer
            else:
                # Determine if targets are numeric to decide between -1 and "None"
                try:
                    int(target_tokens[0])  # Test if first target is numeric
                    extracted_answer_value = -1
                except ValueError:
                    extracted_answer_value = "None"

            # ====================================================================
            # SCORE CALCULATION
            # ====================================================================
            # Compute the final score as the difference between target token logprobs.
            # For binary tasks: score = P(high_value) - P(low_value)
            # Positive scores indicate the model prefers the "higher" value.
            if len(target_tokens) >= 2:
                higher_token = target_tokens[1]  # e.g., "25" or "135"
                lower_token = target_tokens[0]  # e.g., "20" or "125"
                score = logprobs[higher_token] - logprobs[lower_token]
            else:
                score = logprobs[target_tokens[0]]

            # Store results for this strength coefficient
            scores[strength] = score
            logprob_data[strength] = logprobs
            extracted_answers[strength] = extracted_answer_value

            logger.info(f"  Initial generated text: {initial_generated_text}")
            logger.info(f"  Final answer text: {final_answer_text}")
            logger.info(f"  Complete generated text: {complete_generated_text}")
            logger.info(f"  Extracted answer: {extracted_answer_value}")
            logger.info(f"  Logprobs: {logprobs}")
            logger.info(f"  Score (difference): {score}")

            # Log the generated text and logprobs to tensorboard
            zones_tag = format_layer_zones_for_filename(layer_zones)
            model_tag = model_name.replace("/", "_").replace("-", "_")
            normalize_tag = "norm" if normalize else "nonorm"
            rescaling_tag = rescaling if rescaling else "norescale"
            thinking_tag = "thinking" if enable_thinking else "nothinking"

            # Log the complete generated text (both generations) to tensorboard
            writer.add_text(
                f"{model_tag}_{dataset}_{method}/zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}/complete_generated_text",
                f"Strength {strength}: {complete_generated_text}",
                global_step=int(strength * strengths_multiplier_tensorboard),
            )

            # Also log individual components for debugging
            writer.add_text(
                f"{model_tag}_{dataset}_{method}/zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}/initial_generation",
                f"Strength {strength}: {initial_generated_text}",
                global_step=int(strength * strengths_multiplier_tensorboard),
            )
            writer.add_text(
                f"{model_tag}_{dataset}_{method}/zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}/final_answer",
                f"Strength {strength}: {final_answer_text}",
                global_step=int(strength * strengths_multiplier_tensorboard),
            )

            # Log extracted answer to tensorboard
            if isinstance(extracted_answer_value, int):
                writer.add_scalar(
                    f"extracted_answers/value",
                    extracted_answer_value,
                    global_step=int(strength * strengths_multiplier_tensorboard),
                )
            else:
                # For non-numeric answers, log as text
                writer.add_text(
                    f"extracted_answers/text",
                    str(extracted_answer_value),
                    global_step=int(strength * strengths_multiplier_tensorboard),
                )

            # Log all target token logprobs
            logprobs_text = ", ".join(
                [f"{token}: {logprob:.4f}" for token, logprob in logprobs.items()]
            )
            writer.add_text(
                f"{model_tag}_{dataset}_{method}/zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}/logprobs",
                f"Strength {strength}: {logprobs_text}",
                global_step=int(strength * strengths_multiplier_tensorboard),
            )

            # Log individual data point to TensorBoard for native plotting
            # This creates an interactive plot for this specific configuration
            writer.add_scalar(
                "logprob_score_vs_strength",
                score,
                global_step=int(strength * strengths_multiplier_tensorboard),
            )

            # Log individual target token logprobs
            for token, logprob in logprobs.items():
                writer.add_scalar(
                    f"individual_logprobs/{token}",
                    logprob,
                    global_step=int(strength * strengths_multiplier_tensorboard),
                )

        # ========================================================================
        # PHASE 7: ANALYSIS AND VISUALIZATION
        # ========================================================================
        # Create plots and calculate correlation statistics to evaluate how well
        # the control vector influences model behavior in the expected direction.
        strengths_list = sorted(scores.keys())
        scores_list = [scores[s] for s in strengths_list]
        extracted_answers_list = [extracted_answers[s] for s in strengths_list]

        logger.info(f"  Creating plot with {len(scores_list)} logprob data points")

        # Debug: logger.info the data being plotted
        logger.info(f"  Plotting strengths: {strengths_list}")
        logger.info(f"  Plotting logprob scores: {scores_list}")
        logger.info(f"  Plotting extracted answers: {extracted_answers_list}")

        # Calculate correlation between control strength and logprob score
        try:
            if len(strengths_list) > 1 and len(scores_list) > 1:
                correlation_coeff, correlation_p_value = pearsonr(
                    strengths_list, scores_list
                )
                correlation_coeff_squared = correlation_coeff**2
                logger.info(
                    f"  Correlation coefficient: {correlation_coeff:.4f} (p-value: {correlation_p_value:.4f})"
                )
                logger.info(f"  R-squared: {correlation_coeff_squared:.4f}")
        except Exception as e:
            logger.info(f"  Error calculating correlation: {e}")
            if debug:
                raise
            correlation_coeff = 0.0
            correlation_p_value = 1.0
            correlation_coeff_squared = 0.0

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(strengths_list, scores_list, "bo-", linewidth=2, markersize=6)

        # Add extracted answer annotations next to each point
        for i, (x, y, answer) in enumerate(
            zip(strengths_list, scores_list, extracted_answers_list)
        ):
            ax.annotate(
                str(answer),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        ax.set_xlabel("Control Strength", fontsize=12)
        ax.set_ylabel(
            f"Logprob Diff: '{target_tokens[1]}' - '{target_tokens[0]}'", fontsize=12
        )
        ax.set_title(
            f"Token Log Probability vs Control Strength ({dataset} dataset)\n"
            f"Method: {method}, Layer zones: {layer_zones}, Normalize: {normalize}, Rescaling: {rescaling}, Thinking: {enable_thinking}\n"
            f"Model: {model_name}\n"
            f"Correlation: r={correlation_coeff:.3f}, p={correlation_p_value:.3f}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)

        # Fix x-axis to show full range of possible strength values
        ax.set_xlim(min(strengths), max(strengths))
        # Set x-axis ticks to show key strength values for better readability
        ax.set_xticks(
            [s for s in strengths if s % 0.5 == 0]
        )  # Show every 0.5 increment

        # Add reference line for no control
        ax.axvline(x=0, color="g", linestyle="--", alpha=0.5, label="No Control (0)")

        ax.legend()
        plt.tight_layout()

        # Explicitly draw the figure to ensure it's rendered
        fig.canvas.draw()

        # Log summary metrics to the individual configuration writer for filtering
        try:
            # Log correlation metrics with full configuration context
            writer.add_scalar(
                f"correlation/correlation_coeff",
                correlation_coeff,
                global_step=0,
            )
            writer.add_scalar(
                f"correlation/correlation_coeff_squared",
                correlation_coeff_squared,
                global_step=0,
            )
            writer.add_scalar(
                f"correlation/correlation_p_value",
                correlation_p_value,
                global_step=0,
            )

            # Log summary statistics with configuration context
            writer.add_scalar(
                f"summary_stats/mean_logprob_score",
                sum(scores_list) / len(scores_list),
                global_step=0,
            )
            writer.add_scalar(
                f"summary_stats/max_logprob_score",
                max(scores_list),
                global_step=0,
            )
            writer.add_scalar(
                f"summary_stats/min_logprob_score",
                min(scores_list),
                global_step=0,
            )
            writer.add_scalar(
                f"summary_stats/logprob_score_range",
                max(scores_list) - min(scores_list),
                global_step=0,
            )
            writer.add_scalar(
                f"summary_stats/num_logprob_scores",
                len(scores_list),
                global_step=0,
            )

            # Log configuration parameters as scalars for easy filtering
            writer.add_scalar(
                f"config/normalize", 1.0 if normalize else 0.0, global_step=0
            )
            writer.add_scalar(
                f"config/enable_thinking",
                1.0 if enable_thinking else 0.0,
                global_step=0,
            )
            writer.add_scalar(
                f"config/rescaling_enabled", 1.0 if rescaling else 0.0, global_step=0
            )
            writer.add_scalar(
                f"config/num_layer_zones", len(layer_zones), global_step=0
            )

            # Log individual zone boundaries for filtering
            for zone_idx, zone in enumerate(layer_zones):
                writer.add_scalar(
                    f"config/zone_{zone_idx}_start", zone[0], global_step=0
                )
                writer.add_scalar(f"config/zone_{zone_idx}_end", zone[1], global_step=0)
                writer.add_scalar(
                    f"config/zone_{zone_idx}_width", zone[1] - zone[0], global_step=0
                )

            # Log method and dataset as text for reference
            writer.add_text(f"config/method", method, global_step=0)
            writer.add_text(f"config/dataset", dataset, global_step=0)
            writer.add_text(f"config/model_name", model_name, global_step=0)
            writer.add_text(
                f"config/rescaling_type",
                rescaling if rescaling else "none",
                global_step=0,
            )

            logger.info("  Summary metrics logged to individual configuration writer")
        except Exception as e:
            logger.info(f"  Error logging summary metrics to individual writer: {e}")
            if debug:
                raise

        # Log plot to tensorboard
        try:
            # Use configuration-specific tag and unique global step
            plot_tag = f"plots/{model_tag}_{dataset}_{method}_zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}"
            writer.add_figure(
                plot_tag,
                fig,
                global_step=combo_idx,  # Use combo_idx for unique global step
            )
            # Explicitly flush the writer to ensure data is written
            writer.flush()
            logger.info(
                f"  Plot successfully logged to TensorBoard with tag: {plot_tag}"
            )
        except Exception as e:
            logger.info(f"  Error logging plot to TensorBoard: {e}")
            if debug:
                raise

        # Save plot in both locations
        zones_tag = format_layer_zones_for_filename(layer_zones)
        model_tag = model_name.replace("/", "_").replace("-", "_")
        normalize_tag = "norm" if normalize else "nonorm"
        rescaling_tag = rescaling if rescaling else "norescale"
        thinking_tag = "thinking" if enable_thinking else "nothinking"

        # Original plot location
        plot_filename = f"./tensorboard_logs/grid_search/logprob_score_{model_tag}_{dataset}_{method}_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}.png"

        # TensorBoard logs plot location
        tensorboard_plot_filename = (
            f"./tensorboard_logs/grid_search/{run_name}/logprob_score_plot.png"
        )

        try:
            # Save to original location
            fig.savefig(plot_filename, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"  Plot saved: {plot_filename}")

            # Check if file was actually created and has content
            if os.path.exists(plot_filename):
                file_size = os.path.getsize(plot_filename)
                logger.info(f"  Plot file size: {file_size} bytes")
            else:
                logger.info("  Warning: Plot file was not created!")

            # Save to TensorBoard logs directory
            os.makedirs(f"./tensorboard_logs/grid_search/{run_name}", exist_ok=True)
            fig.savefig(
                tensorboard_plot_filename,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            logger.info(
                f"  Plot also saved in TensorBoard logs: {tensorboard_plot_filename}"
            )

        except Exception as e:
            logger.info(f"  Error saving plot: {e}")
            if debug:
                raise

        plt.close(fig)  # Close the specific figure to save memory

        # ========================================================================
        # PHASE 8: CLEANUP AND MEMORY MANAGEMENT
        # ========================================================================
        # Reset the model state and explicitly free GPU memory to prevent OOM
        # errors in subsequent configurations. This is critical for grid search.
        control_model.reset()
        unwrapped_model = control_model.unwrap()

        # Flush and close the TensorBoard writer for this combination
        writer.flush()
        writer.close()

        # Explicitly delete all model references to free GPU memory
        del trained_vector, control_model, unwrapped_model, base_model

        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return {
            "model_name": model_name,
            "method": method,
            "dataset": dataset,
            "layer_zones": layer_zones,
            "normalize": normalize,
            "rescaling": rescaling,
            "enable_thinking": enable_thinking,
            "scores": scores,
            "logprob_data": logprob_data,
            "correlation_coeff": correlation_coeff,
            "correlation_coeff_squared": correlation_coeff_squared,
            "correlation_p_value": correlation_p_value,
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
            "normalize": normalize,
            "rescaling": rescaling,
            "enable_thinking": enable_thinking,
            "scores": {},
            "logprob_data": {},
            "correlation_coeff": 0.0,
            "correlation_coeff_squared": 0.0,
            "correlation_p_value": 1.0,
            "success": False,
            "error": str(e),
        }


# Create directories
os.makedirs("./tensorboard_logs", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Grid search script version for tracking experiments
grid_search_script_version = "1.0.0"


def should_skip_run(run_name: str) -> bool:
    """
    Check if a run should be skipped based on existing folders and done files.

    Returns True if the run should be skipped (already completed).
    If an incomplete run is found (folder exists without done file),
    it cleans up the folder and returns False to allow re-running.
    """
    run_dir = f"./tensorboard_logs/grid_search/{run_name}"
    done_file = os.path.join(run_dir, "done")

    if not os.path.exists(run_dir):
        # No folder exists, should run
        return False

    if os.path.exists(done_file):
        # Done file exists, should skip
        logger.info(f"Skipping completed run: {run_name}")
        return True

    # Folder exists but no done file - incomplete run, clean up
    logger.info(f"Cleaning up incomplete run: {run_name}")
    shutil.rmtree(run_dir)
    return False


def mark_run_complete(run_name: str) -> None:
    """Mark a run as complete by creating a done file in its tensorboard directory."""
    run_dir = f"./tensorboard_logs/grid_search/{run_name}"
    done_file = os.path.join(run_dir, "done")

    # Ensure the directory exists (it should, but just in case)
    os.makedirs(run_dir, exist_ok=True)

    # Create the done file
    with open(done_file, "w") as f:
        f.write("completed\n")

    logger.info(f"Marked run as complete: {run_name}")


def main(
    debug: bool = False,
    grid_reduction: bool = False,
    batch_size: int = 1,
    cuda_visible_devices: str | None = None,
    reduction_factor: float = 0.5,
):
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
    grid_reduction : bool, default=False
        If True, uses GridSearchReductor to reduce the parameter grid size
        while maintaining good coverage of the parameter space. This significantly
        reduces computational cost but may miss some parameter interactions.
    batch_size : int, default=1
        Batch size for training control vectors.
    cuda_visible_devices : str | None, default=None
        If provided, sets CUDA_VISIBLE_DEVICES environment variable to control
        which GPU(s) to use. For example: "0" for GPU 0, "0,1" for GPUs 0 and 1.
    reduction_factor : float, default=0.5
        When grid_reduction is True, the factor by which to reduce the parameter
        grid size. Must be between 0 and 1, where smaller values result in
        smaller reduced grids.

    Notes
    -----
    The function creates several output directories:
    - ./tensorboard_logs/grid_search/ : TensorBoard logs for analysis
    - ./logs/ : Text logs from loguru

    GPU memory is explicitly managed by deleting model references and calling
    torch.cuda.empty_cache() after each configuration to prevent OOM errors.

    Results include correlation analysis between control strength and extracted
    values, which helps identify effective control directions.
    """
    # Set CUDA device visibility if specified
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        logger.info(f"Set CUDA_VISIBLE_DEVICES to: {cuda_visible_devices}")
    # Create main writer for overall grid search logging
    main_writer = SummaryWriter("./tensorboard_logs/grid_search/main")

    # Track model+dataset combinations for which we've already logged training logprobs
    logged_training_logprobs = set()

    # Log version information as metadata
    main_writer.add_text(
        "metadata/grid_search_script_version", grid_search_script_version, 0
    )
    main_writer.add_text("metadata/repeng_version", repeng_version, 0)
    # Note: model_name is now part of the grid and logged per combination

    # Grid search
    grid = ParameterGrid(param_grid)

    if grid_reduction:
        # Use GridSearchReductor to reduce the size of the grid
        converter = GridSearchReductor()
        old_grid = grid
        grid: List[Dict] = converter.fit_transform(
            param_grid, reduction_factor=reduction_factor
        )
        assert len(grid) <= len(old_grid)
        total_combinations = len(grid)

        logger.info(
            f"Starting grid search with {total_combinations} combinations (before reduction: {len(old_grid)})"
        )
    else:
        total_combinations = len(grid)
        logger.info(
            f"Starting grid search with {total_combinations} combinations (no grid reduction)"
        )

    # sort the grid to make sure that we switch model as little as possible
    grid = list(grid)
    grid = sorted(
        grid,
        key=lambda dictparam: str(
            dictparam["model_name"] + str(dictparam["layer_zones"])
        ),
    )

    all_results = []

    for i, params in enumerate(
        tqdm(grid, desc="Grid Search Progress", colour="magenta")
    ):
        model_name = params["model_name"]
        method = params["method"]
        dataset = params["dataset"]
        layer_zones = params["layer_zones"]
        normalize = params["normalize"]
        rescaling = params["rescaling"]
        enable_thinking = params["enable_thinking"]

        # Generate run name for resume functionality (same logic as in test_configuration)
        zones_tag = format_layer_zones_for_filename(layer_zones)
        model_tag = model_name.replace("/", "_").replace("-", "_")
        normalize_tag = "norm" if normalize else "nonorm"
        rescaling_tag = rescaling if rescaling else "norescale"
        thinking_tag = "thinking" if enable_thinking else "nothinking"
        run_name = f"{model_tag}_{dataset}_{method}_zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}"

        # Check if this run should be skipped (already completed)
        if should_skip_run(run_name):
            # Create a placeholder result for skipped runs
            result = {
                "model_name": model_name,
                "method": method,
                "dataset": dataset,
                "layer_zones": layer_zones,
                "normalize": normalize,
                "rescaling": rescaling,
                "enable_thinking": enable_thinking,
                "scores": {},
                "logprob_data": {},
                "correlation_coeff": 0.0,
                "correlation_coeff_squared": 0.0,
                "correlation_p_value": 1.0,
                "success": True,  # Mark as success since it was previously completed
                "skipped": True,
            }
            all_results.append(result)
            logger.info(
                f"Skipped combination {i+1}/{total_combinations} (already completed)"
            )
            continue

        # Test this configuration
        result = test_configuration(
            model_name,
            method,
            layer_zones,
            dataset,
            normalize,
            rescaling,
            enable_thinking,
            i,
            total_combinations,
            debug,
            batch_size,
            logged_training_logprobs,
        )
        all_results.append(result)

        # Mark run as complete if successful
        if result["success"]:
            mark_run_complete(run_name)

        # Only do tensorboard logging for runs that were actually executed (not skipped)
        if not result.get("skipped", False):
            if result["success"] and result["scores"]:
                scores = result["scores"]

                # Log individual points to tensorboard with full configuration context
                zones_tag = format_layer_zones_for_filename(layer_zones)
                model_tag = model_name.replace("/", "_").replace("-", "_")
                normalize_tag = "norm" if normalize else "nonorm"
                rescaling_tag = rescaling if rescaling else "norescale"
                thinking_tag = "thinking" if enable_thinking else "nothinking"

                # Create comprehensive tag for this configuration
                full_config_tag = f"{model_tag}_{dataset}_{method}_zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}"

                for strength, score in scores.items():
                    main_writer.add_scalar(
                        f"individual_scores/{full_config_tag}/logprob_score",
                        score,
                        global_step=int(strength * strengths_multiplier_tensorboard),
                    )

                # Log summary statistics to tensorboard
                if result["success"] and result["scores"]:
                    scores = result["scores"]
                    # Logprobs are always valid (no NaN values to filter)
                    strengths_list = sorted(scores.keys())
                    scores_list = [scores[s] for s in strengths_list]

                    mean_score = sum(scores_list) / len(scores_list)
                    max_score = max(scores_list)
                    min_score = min(scores_list)
                    score_range = max_score - min_score

                    # Get correlation values from the result
                    correlation_coeff = result.get("correlation_coeff", 0.0)
                    correlation_coeff_squared = result.get(
                        "correlation_coeff_squared", 0.0
                    )
                    correlation_p_value = result.get("correlation_p_value", 1.0)

                    # Log hyperparameters and metrics for easy filtering
                    hparam_dict = {
                        "model_name": model_name,
                        "method": method,
                        "dataset": dataset,
                        "normalize": normalize,
                        "rescaling": rescaling if rescaling else "none",
                        "enable_thinking": enable_thinking,
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
                        "hparam/mean_logprob_score": mean_score,
                        "hparam/max_logprob_score": max_score,
                        "hparam/min_logprob_score": min_score,
                        "hparam/logprob_score_range": score_range,
                        "hparam/correlation_coeff": correlation_coeff,
                        "hparam/correlation_coeff_squared": correlation_coeff_squared,
                        "hparam/correlation_p_value": correlation_p_value,
                        "hparam/num_logprob_scores": len(scores_list),
                    }

                    # Log hyperparameters with metrics - this allows filtering in TensorBoard
                    main_writer.add_hparams(hparam_dict, metric_dict)

                    # Also log individual parameters as scalars for time-series analysis
                    main_writer.add_scalar("params/combo_idx", i, i)
                    main_writer.add_scalar(
                        "params/num_layer_zones", len(layer_zones), i
                    )
                    for zone_idx, zone in enumerate(layer_zones):
                        main_writer.add_scalar(
                            f"params/zone_{zone_idx}_start", zone[0], i
                        )
                        main_writer.add_scalar(
                            f"params/zone_{zone_idx}_end", zone[1], i
                        )
                        main_writer.add_scalar(
                            f"params/zone_{zone_idx}_width", zone[1] - zone[0], i
                        )

                    # Use combination index as the x-axis for summary stats with full configuration context
                    model_tag = model_name.replace("/", "_").replace("-", "_")
                    normalize_tag = "norm" if normalize else "nonorm"
                    rescaling_tag = rescaling if rescaling else "norescale"
                    thinking_tag = "thinking" if enable_thinking else "nothinking"

                    # Create comprehensive tag for this configuration
                    full_config_tag = f"{model_tag}_{dataset}_{method}_zones_{zones_tag}_{normalize_tag}_{rescaling_tag}_{thinking_tag}"

                    # Log all summary metrics with full configuration context
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/mean_logprob_score",
                        mean_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/max_logprob_score",
                        max_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/min_logprob_score",
                        min_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/logprob_score_range",
                        score_range,
                        i,
                    )
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/correlation_coeff",
                        correlation_coeff,
                        i,
                    )
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )
                    main_writer.add_scalar(
                        f"summary/{full_config_tag}/correlation_p_value",
                        correlation_p_value,
                        i,
                    )

                    # Log by individual configuration components for easy filtering
                    main_writer.add_scalar(
                        f"by_model/{model_tag}/mean_logprob_score", mean_score, i
                    )
                    main_writer.add_scalar(
                        f"by_model/{model_tag}/correlation_coeff", correlation_coeff, i
                    )
                    main_writer.add_scalar(
                        f"by_model/{model_tag}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

                    main_writer.add_scalar(
                        f"by_method/{method}/mean_logprob_score", mean_score, i
                    )
                    main_writer.add_scalar(
                        f"by_method/{method}/correlation_coeff", correlation_coeff, i
                    )
                    main_writer.add_scalar(
                        f"by_method/{method}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

                    main_writer.add_scalar(
                        f"by_dataset/{dataset}/mean_logprob_score", mean_score, i
                    )
                    main_writer.add_scalar(
                        f"by_dataset/{dataset}/correlation_coeff", correlation_coeff, i
                    )
                    main_writer.add_scalar(
                        f"by_dataset/{dataset}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

                    main_writer.add_scalar(
                        f"by_normalize/{normalize_tag}/mean_logprob_score",
                        mean_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_normalize/{normalize_tag}/correlation_coeff",
                        correlation_coeff,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_normalize/{normalize_tag}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

                    main_writer.add_scalar(
                        f"by_rescaling/{rescaling_tag}/mean_logprob_score",
                        mean_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_rescaling/{rescaling_tag}/correlation_coeff",
                        correlation_coeff,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_rescaling/{rescaling_tag}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

                    main_writer.add_scalar(
                        f"by_thinking/{thinking_tag}/mean_logprob_score", mean_score, i
                    )
                    main_writer.add_scalar(
                        f"by_thinking/{thinking_tag}/correlation_coeff",
                        correlation_coeff,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_thinking/{thinking_tag}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

                    # Also log combined model+dataset+method for comparison across layer zones
                    main_writer.add_scalar(
                        f"by_model_dataset_method/{model_tag}_{dataset}_{method}/mean_logprob_score",
                        mean_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_model_dataset_method/{model_tag}_{dataset}_{method}/max_logprob_score",
                        max_score,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_model_dataset_method/{model_tag}_{dataset}_{method}/logprob_score_range",
                        score_range,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_model_dataset_method/{model_tag}_{dataset}_{method}/correlation_coeff",
                        correlation_coeff,
                        i,
                    )
                    main_writer.add_scalar(
                        f"by_model_dataset_method/{model_tag}_{dataset}_{method}/correlation_coeff_squared",
                        correlation_coeff_squared,
                        i,
                    )

            logger.info(f"Completed combination {i+1}/{total_combinations}")

    # Log final summary
    successful_runs = [r for r in all_results if r["success"]]
    logger.info("\nGrid search completed!")
    logger.info(f"Successful runs: {len(successful_runs)}/{total_combinations}")

    # Create summary report
    summary_file = "./tensorboard_logs/grid_search/summary_report.txt"
    with open(summary_file, "w") as f:
        f.write("Grid Search Summary Report\n")
        f.write("==========================\n\n")
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
            f.write(f"  Normalize: {result.get('normalize', 'unknown')}\n")
            f.write(f"  Rescaling: {result.get('rescaling', 'unknown')}\n")
            f.write(f"  Enable thinking: {result.get('enable_thinking', 'unknown')}\n")
            f.write(f"  Success: {result['success']}\n")
            if result["success"]:
                scores = result["scores"]
                if scores:
                    scores_list = list(scores.values())
                    strengths_list = list(scores.keys())
                    f.write(f"  Logprob scores found: {len(scores)}/{len(strengths)}\n")
                    f.write(
                        f"  Mean logprob score: {sum(scores_list)/len(scores_list):.4f}\n"
                    )
                    f.write(
                        f"  Logprob score range: {min(scores_list):.4f} - {max(scores_list):.4f}\n"
                    )

                    # Calculate and report correlation
                    try:
                        if len(strengths_list) > 1 and len(scores_list) > 1:
                            correlation_coeff, correlation_p_value = pearsonr(
                                strengths_list, scores_list
                            )
                            correlation_coeff_squared = correlation_coeff**2
                            f.write(
                                f"  Correlation coefficient: {correlation_coeff:.4f} (p-value: {correlation_p_value:.4f})\n"
                            )
                            f.write(f"  R-squared: {correlation_coeff_squared:.4f}\n")
                        else:
                            f.write(
                                "  Correlation coefficient: N/A (insufficient data)\n"
                            )
                    except Exception as e:
                        f.write(f"  Correlation coefficient: Error - {e}\n")
                        if debug:
                            raise
                else:
                    f.write("  No logprob scores found\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

    # Flush and close the main writer
    main_writer.flush()
    main_writer.close()

    logger.info(f"Summary report saved to: {summary_file}")
    logger.info(
        "Results logged to tensorboard. Run: tensorboard --logdir=./tensorboard_logs"
    )


if __name__ == "__main__":
    Fire(main)
