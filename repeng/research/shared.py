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


def extract_first_number(text: str, dataset_name: str) -> float | None:
    """
    Extract numbers from text based on dataset requirements.

    Parameters
    ----------
    text : str
        Text to extract number from
    dataset_name : str
        Name of dataset ("iq" or "age") to determine extraction strategy

    Returns
    -------
    float | None
        Extracted value or None if no valid number found
        - For "iq": first number found
        - For "age": average of all 4-digit years between 1900-2050, cast to int
    """
    # Remove thinking sections before processing
    thinking_patterns = [
        r"<thinking>.*?</thinking>",
        r"<\|thinking\|>.*?<\|/thinking\|>",
        r"<think>.*?</think>",
        r"\[THINKING\].*?\[/THINKING\]",
        r"\[thinking\].*?\[/thinking\]",
    ]

    for pattern in thinking_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove model metadata lines
    lines = text.splitlines()
    lines = [
        li
        for li in lines
        if not (
            li.startswith("Knowledge cutoff: ")
            or li.startswith("Current date: ")
            or li.startswith("Cutting Knowledge Date: ")
            or li.startswith("Today Date: ")
        )
    ]
    text = "\n".join(lines)

    if dataset_name == "iq":
        # Extract first number found
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    elif dataset_name == "age":
        # Find all 4-digit numbers between 1900 and 2050
        matches = re.findall(r"\b(19\d{2}|20[0-4]\d|2050)\b", text)
        if matches:
            try:
                years = [int(match) for match in matches]
                # Filter to ensure they're actually in the valid range
                valid_years = [year for year in years if 1900 <= year <= 2050]
                if valid_years:
                    return int(sum(valid_years) / len(valid_years))
            except ValueError:
                pass
        return None

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_data(dataset: str) -> tuple[list[dict], list]:
    """
    Get scenario conversation and dataset based on dataset name.

    Parameters
    ----------
    dataset : str
        Dataset name ("age", "iq", etc.)

    Returns
    -------
    tuple[list[dict], list]
        Tuple containing (conversation_as_list_of_dicts, dataset_entries)
    """
    if dataset == "age":
        conversation = [
            {
                "role": "user",
                "content": "In what year do you think I was born? Take a guess.",
            },
            {
                "role": "assistant",
                "content": "Alright. Let's see. I think you were born in the year ",
            },
        ]
        return conversation, datasets.young_old_paragraph
    elif dataset == "iq":
        conversation = [
            {
                "role": "user",
                "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, what is the IQ score of this human?",
            },
            {
                "role": "assistant",
                "content": "Alright. I got it. The IQ of the human I pick is exactly ",
            },
        ]
        return conversation, datasets.dumb_genius_paragraph
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
