"""
Shared configuration and utilities for repeng research experiments.

This module centralizes common experiment parameters to ensure consistency
across different research scripts and avoid duplication.
"""

import re
from typing import List


# Standard strength values for control vector testing
# These represent the coefficient multipliers applied to control vectors
DEFAULT_STRENGTHS: List[float] = [x for x in range(-10, 11, 1)]

# Alternative fine-grained strength values (increments of 0.05 from -0.5 to +0.5)
FINE_GRAINED_STRENGTHS: List[float] = [x / 100 for x in range(-50, 55, 5)]


def extract_first_number(text: str, max_value: float | None = None) -> float | None:
    """
    Extract the first number from text using regex.

    This function handles various number formats including comma/dot thousand
    separators and applies preprocessing to filter out common model metadata lines.

    Parameters
    ----------
    text : str
        Text to extract number from
    max_value : float | None, optional
        Maximum value to return - if extracted value exceeds this, returns max_value

    Returns
    -------
    float | None
        First number found in text, or None if no valid number found
    """
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

    # Try to match numbers with comma thousand separators first: 1,000 or 1,000.50
    match = re.search(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", text)
    if match:
        number_str = match.group()
        # Remove commas (thousand separators) and convert to float
        cleaned_number = number_str.replace(",", "")
        try:
            value = float(cleaned_number)
            if max_value is not None and value > max_value:
                return max_value
            return value
        except ValueError:
            pass

    # Try to match numbers with dot thousand separators (European style): 10.000
    # Only match if it looks like thousand separators (groups of 3 digits)
    match = re.search(r"\d{1,3}(?:\.\d{3})+(?!\.\d)", text)
    if match:
        number_str = match.group()
        # Remove dots (treating as thousand separators) and convert to float
        cleaned_number = number_str.replace(".", "")
        try:
            value = float(cleaned_number)
            if max_value is not None and value > max_value:
                return max_value
            return value
        except ValueError:
            pass

    # Fall back to original pattern for simple numbers: 123 or 123.45
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        try:
            value = float(match.group())
            if max_value is not None and value > max_value:
                return max_value
            return value
        except ValueError:
            pass

    return None
