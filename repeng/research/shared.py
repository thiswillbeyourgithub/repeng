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

        result[target] = matching_ids

    return result


def extract_token_logprobs(
    model,
    tokenizer,
    input_text: str,
    target_tokens: List[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Extract log probabilities for specific target tokens at the next position.

    This function finds all vocabulary tokens that match each target token
    (containing the target string and no other digits) and sums their probabilities.

    Parameters
    ----------
    model : ControlModel
        The wrapped model to get predictions from
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for the model
    input_text : str
        Input text to get next token predictions for
    target_tokens : List[str]
        List of target tokens to extract logprobs for
    normalize : bool, default=True
        Whether to apply softmax normalization to get proper probabilities

    Returns
    -------
    dict[str, float]
        Dictionary mapping each target token to its log probability
    """
    import torch
    import torch.nn.functional as F

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the last position (next token prediction)
        next_token_logits = outputs.logits[0, -1, :]

    # Convert to probabilities for summing, then back to log probabilities
    if normalize:
        probs = F.softmax(next_token_logits, dim=-1)
    else:
        probs = torch.exp(next_token_logits)  # Assume logits are log probabilities

    # Find all matching token IDs for each target
    token_id_mapping = find_matching_token_ids(tokenizer, target_tokens)

    # Extract and sum probabilities for matching tokens
    result = {}
    for target, matching_ids in token_id_mapping.items():
        if matching_ids:
            # Sum probabilities of all matching tokens
            total_prob = sum(probs[token_id].item() for token_id in matching_ids)
            # Convert back to log probability
            result[target] = torch.log(torch.tensor(total_prob)).item()
        else:
            # If no matching tokens found, assign very low probability
            result[target] = float("-inf")

    return result


def get_data(dataset: str) -> tuple[list[dict], list, list[str], str]:
    """
    Get scenario conversation, dataset, target tokens, and score token based on dataset name.

    Parameters
    ----------
    dataset : str
        Dataset name ("age", "iq", etc.)

    Returns
    -------
    tuple[list[dict], list, list[str], str]
        Tuple containing (conversation_as_list_of_dicts, dataset_entries, target_tokens, score_token)
    """
    if dataset == "age":
        conversation = [
            {
                "role": "user",
                "content": "You have to take a wild guess: what age group do you think I belong to? Please choose either 20 or 25.",
            },
            {
                "role": "assistant",
                "content": "Tough one. I think you are most likely ",
            },
        ]
        target_tokens = ["20", "25"]
        score_token = (
            "25"  # Use logprob of "25" as the score (higher = more likely old)
        )
        return conversation, datasets.young_old_paragraph, target_tokens, score_token
    elif dataset == "iq":
        conversation = [
            {
                "role": "user",
                "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, what is the IQ score of this human? Please choose either 125 or 135.",
            },
            {
                "role": "assistant",
                "content": "Alright. I got it. The IQ of the human I pick is ",
            },
        ]
        target_tokens = ["125", "135"]
        score_token = (
            "135"  # Use logprob of "135" as the score (higher = more likely genius)
        )
        return conversation, datasets.dumb_genius_paragraph, target_tokens, score_token
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
