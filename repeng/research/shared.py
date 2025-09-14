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


def extract_token_logprobs(
    model,
    tokenizer,
    input_text: str,
    target_tokens: List[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Extract log probabilities for specific target tokens at the next position.

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

    # Convert to log probabilities
    if normalize:
        log_probs = F.log_softmax(next_token_logits, dim=-1)
    else:
        log_probs = next_token_logits

    # Extract logprobs for target tokens
    result = {}
    for token in target_tokens:
        # Tokenize the target token to get its ID
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = token_ids[0]
            result[token] = log_probs[token_id].item()
        else:
            # Handle multi-token case by taking average (or you could modify this logic)
            logprobs_sum = sum(log_probs[tid].item() for tid in token_ids)
            result[token] = logprobs_sum / len(token_ids)

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
                "content": "You have to take a wild guess: what age group do you think I belong to? Please choose one: 20, 30, 40, or 50.",
            },
            {
                "role": "assistant",
                "content": "Tough one. I think you are most likely ",
            },
        ]
        target_tokens = ["20", "30", "40", "50"]
        score_token = (
            "20"  # Use logprob of "20" as the score (higher = more likely young)
        )
        return conversation, datasets.young_old_paragraph, target_tokens, score_token
    elif dataset == "iq":
        conversation = [
            {
                "role": "user",
                "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, what is the IQ score of this human? Please choose one: 100, 110, 120, or 130.",
            },
            {
                "role": "assistant",
                "content": "Alright. I got it. The IQ of the human I pick is around ",
            },
        ]
        target_tokens = ["100", "110", "120", "130"]
        score_token = (
            "130"  # Use logprob of "130" as the score (higher = more likely genius)
        )
        return conversation, datasets.dumb_genius_paragraph, target_tokens, score_token
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
