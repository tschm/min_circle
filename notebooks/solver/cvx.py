import statistics
import timeit

import cvxpy as cp
import numpy as np

from .utils.circle import Circle


def min_circle_cvx(points, **kwargs):
    # Use con_1 if no constraint construction is defined
    # cvxpy variable for the radius
    r = cp.Variable(name="Radius")
    # cvxpy variable for the midpoint
    x = cp.Variable(points.shape[1], name="Midpoint")
    objective = cp.Minimize(r)
    constraints = [
        cp.SOC(
            r * np.ones(points.shape[0]),
            points - cp.outer(np.ones(points.shape[0]), x),
            axis=1,
        )
    ]

    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(**kwargs)

    return Circle(radius=float(r.value), center=x.value)


def min_circle_cvx_3():
    # Use con_1 if no constraint construction is defined
    # cvxpy variable for the radius
    r = cp.Variable(name="Radius")
    # cvxpy variable for the midpoint
    x = cp.Variable((1, 2), name="Midpoint")
    p = cp.Parameter((3, 2), "points")
    objective = cp.Minimize(r)
    constraints = [
        cp.SOC(
            cp.hstack([r, r, r]),
            p - cp.vstack([x, x, x]),
            axis=1,
        )
    ]

    problem = cp.Problem(objective=objective, constraints=constraints)
    print(problem)
    return problem


def min_circle_cvx_2():
    # Use con_1 if no constraint construction is defined
    # cvxpy variable for the radius
    r = cp.Variable(name="Radius")
    # cvxpy variable for the midpoint
    x = cp.Variable((1, 2), name="Midpoint")
    p = cp.Parameter((2, 2), "points")
    objective = cp.Minimize(r)
    constraints = [
        cp.SOC(
            cp.hstack([r, r]),
            p - cp.vstack([x, x]),
            axis=1,
        )
    ]

    problem = cp.Problem(objective=objective, constraints=constraints)
    return problem
    # problem.solve(**kwargs)

    # return Circle(radius=float(r.value), center=x.value)


if __name__ == "__main__":
    p = np.array([[0, 0], [0.0, 2.0], [4.0, 4.0]])
    problem = min_circle_cvx_3()
    problem.param_dict["points"].value = p
    problem.solve(solver="CLARABEL")
    print(problem.var_dict["Radius"].value)

    p = np.random.randn(2, 2)
    problem = min_circle_cvx_2()

    def f():
        problem.param_dict["points"].value = p
        problem.solve(solver="CLARABEL")

    results = timeit.repeat(f, number=1, repeat=10)
    print(statistics.mean(results))

    def g():
        min_circle_cvx(p, solver="CLARABEL")

    results = timeit.repeat(g, number=1, repeat=10)
    print(statistics.mean(results))
