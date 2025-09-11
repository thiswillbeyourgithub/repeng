"""
Shared configuration and utilities for repeng research experiments.

This module centralizes common experiment parameters to ensure consistency
across different research scripts and avoid duplication.
"""

from typing import List


# Standard strength values for control vector testing
# These represent the coefficient multipliers applied to control vectors
DEFAULT_STRENGTHS: List[float] = [
    # -5,
    # -4,
    # -3,
    # -2,
    -1.5,
    -1.4,
    -1.3,
    -1.2,
    -1.1,
    -1.0,
    -0.9,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
]

# Alternative fine-grained strength values (increments of 0.05 from -0.5 to +0.5)
FINE_GRAINED_STRENGTHS: List[float] = [
    x / 100
    for x in range(-50, 55, 5)
]
