"""CVXPY implementation for finding the minimum enclosing circle.

This module provides functions to find the minimum enclosing circle
using convex optimization with the CVXPY library.
"""

import statistics
import timeit
from typing import Any

import cvxpy as cp
import numpy as np

from .utils.circle import Circle


def min_circle_cvx(points: np.ndarray, **kwargs: Any) -> Circle:
    """Find the minimum enclosing circle using convex optimization.

    Uses CVXPY to formulate and solve the minimum enclosing circle problem
    as a second-order cone program (SOCP).

    Args:
        points: Array of 2D points with shape (n, 2)
        **kwargs: Additional keyword arguments to pass to the solver

    Returns:
        Circle object containing the center and radius of the minimum enclosing circle
    """
    # CVXPY variable for the radius
    r = cp.Variable(name="Radius")
    # CVXPY variable for the midpoint
    x = cp.Variable(points.shape[1], name="Midpoint")

    # Minimize the radius
    objective = cp.Minimize(r)

    # Second-order cone constraints ensuring all points are within the circle
    constraints = [
        cp.SOC(
            r * np.ones(points.shape[0]),
            points - cp.outer(np.ones(points.shape[0]), x),
            axis=1,
        )
    ]

    # Create and solve the problem
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(**kwargs)

    return Circle(radius=float(r.value), center=x.value)


def min_circle_cvx_3() -> cp.Problem:
    """Create a CVXPY problem for finding the minimum circle for 3 points.

    Creates a parameterized problem that can be reused with different point sets.

    Returns:
        CVXPY Problem object ready to be solved
    """
    # CVXPY variable for the radius
    r = cp.Variable(name="Radius")
    # CVXPY variable for the midpoint
    x = cp.Variable((1, 2), name="Midpoint")
    # Parameter for the 3 points
    p = cp.Parameter((3, 2), "points")

    # Minimize the radius
    objective = cp.Minimize(r)

    # Second-order cone constraints for 3 points
    constraints = [
        cp.SOC(
            cp.hstack([r, r, r]),
            p - cp.vstack([x, x, x]),
            axis=1,
        )
    ]

    problem = cp.Problem(objective=objective, constraints=constraints)
    return problem


def min_circle_cvx_2() -> cp.Problem:
    """Create a CVXPY problem for finding the minimum circle for 2 points.

    Creates a parameterized problem that can be reused with different point sets.

    Returns:
        CVXPY Problem object ready to be solved
    """
    # CVXPY variable for the radius
    r = cp.Variable(name="Radius")
    # CVXPY variable for the midpoint
    x = cp.Variable((1, 2), name="Midpoint")
    # Parameter for the 2 points
    p = cp.Parameter((2, 2), "points")

    # Minimize the radius
    objective = cp.Minimize(r)

    # Second-order cone constraints for 2 points
    constraints = [
        cp.SOC(
            cp.hstack([r, r]),
            p - cp.vstack([x, x]),
            axis=1,
        )
    ]

    problem = cp.Problem(objective=objective, constraints=constraints)
    return problem


if __name__ == "__main__":
    # Example with 3 points
    p = np.array([[0, 0], [0.0, 2.0], [4.0, 4.0]])
    problem = min_circle_cvx_3()
    problem.param_dict["points"].value = p
    problem.solve(solver="CLARABEL")
    print(problem.var_dict["Radius"].value)

    # Performance testing with 2 points
    p = np.random.randn(2, 2)
    problem = min_circle_cvx_2()

    def f() -> None:
        """Benchmark function for parameterized problem."""
        problem.param_dict["points"].value = p
        problem.solve(solver="CLARABEL")

    results = timeit.repeat(f, number=1, repeat=10)
    print(f"Parameterized problem mean time: {statistics.mean(results):.6f} seconds")

    def g() -> Circle:
        """Benchmark function for direct problem formulation."""
        return min_circle_cvx(p, solver="CLARABEL")

    results = timeit.repeat(g, number=1, repeat=10)
    print(f"Direct problem mean time: {statistics.mean(results):.6f} seconds")
