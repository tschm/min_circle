import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import statistics as stats
    import timeit as tt

    import numpy as np
    from .solver.cvx import min_circle_cvx
    from .solver.mosek import min_circle_mosek
    from .solver.welzl import min_circle_welzl


@app.cell
def _(mo):
    mo.md("""# Problem""")
    return


@app.cell
def _(mo):
    mo.md(
        """We compute the radius and center of the smallest enclosing
        ball for $N$ points in $d$ dimensions.
        We use a variety of tools and compare their performance.
        For fun we included the recursive algorithm by Emo Welzl.
        Hence we work with $d=2$."""
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate a cloud of points""")
    return


@app.cell
def _():
    # generate random points in space
    pos = np.random.randn(2600, 2)
    return (pos,)


@app.cell
def _():
    # Create the figure
    from solver.utils.figure import create_figure

    fig = create_figure()
    return (fig,)


@app.cell
def _(fig, pos):
    # add the cloud plot
    from solver.utils.cloud import Cloud

    cloud = Cloud(points=pos)

    fig.add_trace(cloud.scatter())
    return


@app.cell
def _(mo):
    mo.md("""## Compute with cvxpy""")
    return


@app.cell
def _(fig, pos):
    print(min_circle_cvx(points=pos, solver="CLARABEL"))
    print(min_circle_cvx(points=pos, solver="MOSEK"))
    circle = min_circle_cvx(points=pos, solver="CLARABEL")
    fig.add_trace(circle.scatter())
    fig.show()

    def cvx1():
        min_circle_cvx(points=pos, solver="CLARABEL")

    def cvx2():
        min_circle_cvx(points=pos, solver="MOSEK")

    # Run each 1000 times
    times_clarabel = tt.repeat(cvx1, number=1, repeat=50)
    times_cvx_mosek = tt.repeat(cvx2, number=1, repeat=50)

    print(f"Implementation cvxpy/clarabel: {stats.mean(times_clarabel):.6f} seconds")
    print(f"Implementation cvxpy/mosek: {stats.mean(times_cvx_mosek):.6f} seconds")
    return


@app.cell
def _(mo):
    mo.md("""## Compute with Mosek""")
    return


@app.cell
def _(pos):
    print(min_circle_mosek(points=pos))

    def mosek():
        min_circle_mosek(pos)

    # Run each 1000 times
    times_mosek = tt.repeat(mosek, number=1, repeat=50)

    print(f"Implementation average: {stats.mean(times_mosek):.6f} seconds")
    return


@app.cell
def _(mo):
    mo.md("""## Compute using Welzl's algorithm""")
    return


@app.cell
def _(pos):
    print(min_circle_welzl(points=pos))

    def welzl():
        min_circle_welzl(points=pos)

    # Run each 50 times
    times_welzl = tt.repeat(welzl, number=1, repeat=100)

    print(f"Implementation average: {stats.mean(times_welzl):.6f} seconds")
    return


if __name__ == "__main__":
    app.run()
