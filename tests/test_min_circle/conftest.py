"""Pytest configuration and fixtures for the min_circle test suite.

This module contains shared fixtures that can be used across multiple test files.
"""

import numpy as np
import pytest


@pytest.fixture
def points() -> list[np.ndarray]:
    """Create a list of three 2D points forming a triangle.

    Returns:
        List[np.ndarray]: A list containing three 2D points at coordinates
            [0,0], [4,0], and [2,4].
    """
    p1 = np.array([0, 0])
    p2 = np.array([4, 0])
    p3 = np.array([2, 4])
    return [p1, p2, p3]
