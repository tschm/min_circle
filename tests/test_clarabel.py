import numpy as np
import cvxpy as cp

import pytest


def con_3(points, x, r):
    return [
        cp.SOC(
            r * np.ones(points.shape[0]),
            points - cp.outer(np.ones(points.shape[0]), x),
            axis=1,
        )
    ]


def min_circle_cvx(points, **kwargs):
    # Use con_1 if no constraint construction is defined
    fct = con_3
    # cvxpy variable for the radius
    r = cp.Variable(1, name="Radius")
    # cvxpy variable for the midpoint
    x = cp.Variable(points.shape[1], name="Midpoint")

    objective = cp.Minimize(r)
    constraints = fct(points, x, r)

    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(**kwargs)

    return {"Radius": r.value, "Midpoint": x.value}


def test_clarabel(np_points):
    results = min_circle_cvx(np_points, solver="CLARABEL")
    assert results["Radius"] == pytest.approx(2.5)
    assert results["Midpoint"] == pytest.approx(np.array([2.0, 1.5]))
