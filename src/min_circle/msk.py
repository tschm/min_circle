"""MOSEK implementation for finding the minimum enclosing circle.

This module provides functions to find the minimum enclosing circle
using the MOSEK Fusion optimization library.

Note:
    The MOSEK dependency is optional. Importing this module no longer requires
    MOSEK to be installed. The import happens lazily inside the solver function.
"""

from typing import Any

import numpy as np

from .utils.circle import Circle

_MOSEK_IMPORT_ERROR = (
    "MOSEK is required for min_circle_mosek(). Install with `pip install 'min_circle[solvers]'` "
    "or install the `mosek` package manually."
)


def min_circle_mosek(points: np.ndarray, **kwargs: Any) -> Circle:
    """Find the minimum enclosing circle using MOSEK Fusion.

    Uses the MOSEK Fusion API to formulate and solve the minimum enclosing circle
    problem as a second-order cone program (SOCP).

    Args:
        points: Array of 2D points with shape (n, 2)
        **kwargs: Additional keyword arguments to pass to the solver

    Returns:
        Circle object containing the center and radius of the minimum enclosing circle

    Raises:
        ImportError: If the `mosek` package is not installed. Install with
            `pip install 'min_circle[solvers]'` or add the `solvers` extra.

    Notes:
        Implementation based on MOSEK's minimum ellipsoid tutorial:
        https://github.com/MOSEK/Tutorials/blob/master/minimum-ellipsoid/minimum-ellipsoid.ipynb
    """
    try:
        import mosek.fusion as mf  # type: ignore
    except Exception as exc:  # pragma: no cover - only hit when MOSEK is missing
        raise ImportError(_MOSEK_IMPORT_ERROR) from exc

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
