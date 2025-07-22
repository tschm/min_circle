"""Hexaly implementation for finding the minimum enclosing circle.

This module provides functions to find the minimum enclosing circle
using the Hexaly optimization library.
"""

from typing import Any

import hexaly.optimizer
import numpy as np

from .utils.circle import Circle


def min_circle_hexaly(points: np.ndarray, **kwargs: Any) -> Circle:
    """Find the minimum enclosing circle using Hexaly optimizer.

    Uses the Hexaly optimization library to formulate and solve the
    minimum enclosing circle problem.

    Args:
        points: Array of 2D points with shape (n, 2)
        **kwargs: Additional keyword arguments to pass to the solver

    Returns:
        Circle object containing the center and radius of the minimum enclosing circle
    """
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        # Declare the optimization model
        model = optimizer.model

        # Create variables for the center coordinates within the bounds of the points
        z = np.array([model.float(np.min(points[:, j]), np.max(points[:, j])) for j in range(points.shape[1])])

        # Calculate squared distances from each point to the center
        radius = [np.sum((z - point) ** 2) for point in points]

        # Minimize the maximum distance (radius)
        r = model.sqrt(model.max(radius))
        model.minimize(r)
        model.close()

        # Solve the optimization problem
        optimizer.solve(**kwargs)

        # Return the circle with the optimal center and radius
        return Circle(radius=r.value, center=np.array([z[0].value, z[1].value]))
