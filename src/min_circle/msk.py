"""MOSEK implementation for finding the minimum enclosing circle.

This module provides functions to find the minimum enclosing circle
using the MOSEK Fusion optimization library.
"""

from typing import Any

import mosek.fusion as mf
import numpy as np

from .utils.circle import Circle


def min_circle_mosek(points: np.ndarray, **kwargs: Any) -> Circle:
    """Find the minimum enclosing circle using MOSEK Fusion.

    Uses the MOSEK Fusion API to formulate and solve the minimum enclosing circle
    problem as a second-order cone program (SOCP).

    Args:
        points: Array of 2D points with shape (n, 2)
        **kwargs: Additional keyword arguments to pass to the solver

    Returns:
        Circle object containing the center and radius of the minimum enclosing circle

    Notes:
        Implementation based on MOSEK's minimum ellipsoid tutorial:
        https://github.com/MOSEK/Tutorials/blob/master/minimum-ellipsoid/minimum-ellipsoid.ipynb
    """
    with mf.Model() as model:
        # Create variables for radius and center
        r = model.variable("Radius", 1)
        x = model.variable("Midpoint", [1, points.shape[1]])

        # Number of points
        k = points.shape[0]

        # Repeat the radius and center variables for each point
        r0 = mf.Var.repeat(r, k)
        x0 = mf.Var.repeat(x, k)

        # Create second-order cone constraints ensuring all points are within the circle
        model.constraint(mf.Expr.hstack(r0, mf.Expr.sub(x0, points)), mf.Domain.inQCone())

        # Set the objective to minimize the radius
        model.objective("obj", mf.ObjectiveSense.Minimize, r)

        # Solve the optimization problem
        model.solve(**kwargs)

        # Return the circle with the optimal center and radius
        return Circle(radius=r.level(), center=x.level())
