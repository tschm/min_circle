"""Marimo app for demonstrating minimum enclosing circle algorithms.

This module provides an interactive Marimo application that demonstrates
the computation of the minimum enclosing circle for a set of randomly
generated points using various algorithms, particularly focusing on
the CVXPY implementation with the CLARABEL solver.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import statistics as stats
    import timeit as tt

    import marimo as mo
    import numpy as np

    from min_circle.cvx import min_circle_cvx

    pos = np.random.randn(2600, 2)


@app.cell
def _():
    mo.md("""# Problem""")
    return


@app.cell
def _():
    mo.md(
        """We compute the radius and center of the smallest enclosing
        ball for $N$ points in $d$ dimensions.
        We use a variety of tools and compare their performance.
        For fun we included the recursive algorithm by Emo Welzl.
        Hence we work with $d=2$."""
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Generate a cloud of points""")
    return


@app.cell
def _():
    # Create the figure
    from min_circle.utils.figure import create_figure

    fig = create_figure()
    return (fig,)


@app.cell
def _(fig):
    # add the cloud plot
    from src.min_circle.utils.cloud import Cloud

    cloud = Cloud(points=pos)

    fig.add_trace(cloud.scatter())
    return


@app.cell
def _():
    mo.md("""## Compute with cvxpy""")
    return


@app.cell
def _(fig):
    print(min_circle_cvx(points=pos, solver="CLARABEL"))
    circle = min_circle_cvx(points=pos, solver="CLARABEL")
    fig.add_trace(circle.scatter())
    fig.show()

    def cvx1():
        min_circle_cvx(points=pos, solver="CLARABEL")

    # Run each 1000 times
    times_clarabel = tt.repeat(cvx1, number=1, repeat=50)

    print(f"Implementation cvxpy/clarabel: {stats.mean(times_clarabel):.6f} seconds")
    return


if __name__ == "__main__":
    app.run()
