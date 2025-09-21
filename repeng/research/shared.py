"""
Shared configuration and utilities for repeng research experiments.

This module centralizes common experiment parameters to ensure consistency
across different research scripts and avoid duplication.
"""

import re
from typing import List
from repeng.research import datasets


# Standard strength values for control vector testing
# These represent the coefficient multipliers applied to control vectors
DEFAULT_STRENGTHS: List[float] = [x for x in range(-10, 11, 1)]

# Alternative fine-grained strength values (increments of 0.05 from -0.5 to +0.5)
FINE_GRAINED_STRENGTHS: List[float] = [x / 100 for x in range(-50, 55, 5)]

# For quicker test
SHORT_TEST_STRENGTHS: List[float] = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]


def find_matching_token_ids(
    tokenizer, target_tokens: List[str]
) -> dict[str, List[int]]:
    """
    Find all token IDs in the vocabulary that match each target token.

    For each target token, finds all vocabulary tokens that:
    - When converted to string and lowercased, contain the target string
    - Don't contain any digits other than those in the target
    - Are at most len(target_string) + 2 characters long

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer to search vocabulary of
    target_tokens : List[str]
        List of target tokens to find matches for

    Returns
    -------
    dict[str, List[int]]
        Dictionary mapping each target token to list of matching token IDs
    """

    result = {}

    for target in target_tokens:
        target_lower = str(target).lower()
        target_digits = set(re.findall(r"\d", target_lower))
        max_length = len(target_lower) + 2

        matching_ids = []

        # Iterate through vocabulary
        for token_id in range(len(tokenizer)):
            try:
                # Get token string representation
                token_str = str(tokenizer.decode([token_id])).lower()

                # Check if token contains target and meets criteria
                if target_lower in token_str and len(token_str) <= max_length:

                    # Check that no other digits are present
                    token_digits = set(re.findall(r"\d", token_str))
                    if token_digits.issubset(target_digits):
                        matching_ids.append(token_id)

            except Exception:
                # Skip tokens that can't be decoded
                continue

        # If no matches found, try to find exact target token as fallback
        if not matching_ids:
            try:
                # Try to encode the target and see if it results in a single token
                encoded = tokenizer.encode(target, add_special_tokens=False)
                if len(encoded) == 1:
                    matching_ids.append(encoded[0])
                elif len(encoded) > 1:
                    # If target encodes to multiple tokens, use the first one
                    matching_ids.append(encoded[0])
            except Exception:
                pass

        # If still no matches, find any token that contains the target (without restrictions)
        if not matching_ids:
            for token_id in range(len(tokenizer)):
                try:
                    token_str = str(tokenizer.decode([token_id])).lower()
                    if target_lower in token_str:
                        matching_ids.append(token_id)
                        break  # Just need one fallback
                except Exception:
                    continue

        result[target] = matching_ids

    # Assert that no lists are empty
    for target, token_ids in result.items():
        assert token_ids, f"No matching token IDs found for target '{target}'"

    return result


def extract_token_logprobs(
    model,
    tokenizer,
    target_tokens: List[str],
    input_text: str = None,
    input_ids=None,
    num_generated_tokens: int = 0,
    normalize: bool = True,
    sum_across_positions: bool = False,
) -> dict[str, float]:
    """
    Extract log probabilities for specific target tokens.

    This function finds all vocabulary tokens that match each target token
    (containing the target string and no other digits) and computes their probabilities.
    Can handle both single-position (next token prediction) and multi-position
    (across generated tokens) scenarios.

    Parameters
    ----------
    model : ControlModel
        The wrapped model to get predictions from
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for the model
    target_tokens : List[str]
        List of target tokens to extract logprobs for
    input_text : str, optional
        Input text to get predictions for. Either this or input_ids must be provided.
    input_ids : torch.Tensor, optional
        Pre-tokenized input. Either this or input_text must be provided.
    num_generated_tokens : int, default=0
        Number of generated token positions to consider from the end.
        If 0, uses next token prediction (last position only).
        If > 0, uses the last N positions where tokens were generated.
    normalize : bool, default=True
        Whether to apply softmax normalization to get proper probabilities
    sum_across_positions : bool, default=False
        If True and num_generated_tokens > 0, sums logprobs across all positions.
        If False, uses only the last position.

    Returns
    -------
    dict[str, float]
        Dictionary mapping each target token to its log probability
    """
    import torch
    import torch.nn.functional as F

    # Prepare input
    if input_ids is None:
        if input_text is None:
            raise ValueError("Either input_text or input_ids must be provided")
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
    else:
        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids)

        if num_generated_tokens > 0:
            # Multi-position case: get logits for the last N generated positions
            if num_generated_tokens > outputs.logits.shape[1]:
                raise ValueError(
                    f"num_generated_tokens ({num_generated_tokens}) cannot be larger than sequence length ({outputs.logits.shape[1]})"
                )
            logits = outputs.logits[0, -num_generated_tokens:, :]
        else:
            # Single position case: get logits for the last position (next token prediction)
            logits = outputs.logits[0, -1:, :]  # Keep dimension for consistency

    # Apply normalization
    if normalize:
        log_probs = F.log_softmax(logits, dim=-1)
    else:
        log_probs = logits  # Assume logits are already log probabilities

    # Find all matching token IDs for each target
    token_id_mapping = find_matching_token_ids(tokenizer, target_tokens)

    # Extract and compute probabilities for matching tokens
    result = {}
    for target, matching_ids in token_id_mapping.items():
        if matching_ids:
            if sum_across_positions and num_generated_tokens > 0:
                # Sum logprobs across all positions and all matching token IDs
                total_logprob = 0.0
                for pos in range(log_probs.shape[0]):
                    for token_id in matching_ids:
                        total_logprob += log_probs[pos, token_id].item()
                result[target] = total_logprob
            else:
                # Use only the last position, sum across matching token IDs
                last_pos_log_probs = log_probs[-1, :]  # Last position
                total_logprob = 0.0
                for token_id in matching_ids:
                    total_logprob += torch.exp(last_pos_log_probs[token_id]).item()
                # Convert back to log probability
                result[target] = torch.log(torch.tensor(total_logprob)).item()
        else:
            # If no matching tokens found, assign very low probability
            result[target] = float("-inf")
            breakpoint()

    return result


def get_data(dataset: str) -> tuple[list[dict], list, list[str], str]:
    """
    Get scenario conversation, dataset and target tokens based on dataset name.

    Parameters
    ----------
    dataset : str
        Dataset name ("age", "iq", etc.)

    Returns
    -------
    tuple[list[dict], list, list[str], str]
        Tuple containing (conversation_as_list_of_dicts, dataset_entries, target_tokens
    """
    target_tokens = ["A", "B"]
    if dataset == "age":
        conversation = [
            {
                "role": "system",
                "content": "You take a wild guess about the user with very little prior. Your final answer must be either A or B.",
            },
            {
                "role": "user",
                "content": "Guess my age. Please answer A if 20 years old or B for 25 years old.",
            },
        ]
        return conversation, datasets.young_old_paragraph, target_tokens
    elif dataset == "iq":
        conversation = [
            {
                "role": "system",
                "content": "You take a wild guess about the user with very little prior. Your final answer must be either A or B.",
            },
            {
                "role": "user",
                "content": "Guess your own IQ. Please answer A for 125 or B for 135.",
            },
        ]
        return conversation, datasets.dumb_genius_paragraph, target_tokens
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
