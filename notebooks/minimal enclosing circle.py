import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""# Problem""")
    return


@app.cell
def _(mo):
    mo.md(
        """We compute the radius and center of the smallest enclosing ball for $N$ points in $d$ dimensions. We use a variety of tools and compare their performance. For fun we included the recursive algorithm by Emo Welzl. Hence we work with $d=2$."""
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
    import plotly.graph_objects as go
    import numpy as np
    import timeit as tt
    import statistics as stats

    return go, np, stats, tt


@app.cell
def _(np):
    # generate random points in space
    pos = np.random.randn(2600, 2)
    return (pos,)


@app.cell
def _(go, pos):
    # Create the scatter plot
    fig = go.Figure(
        data=go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode="markers", marker=dict(symbol="x", size=10)
        )
    )

    # Update layout for equal aspect ratio and axis labels
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    # Show the plot
    fig.show()

    # plot makes really only sense when using d=2
    return (fig,)


@app.cell
def _(mo):
    mo.md("""## Compute with cvxpy""")
    return


@app.cell
def _(np):
    import cvxpy as cp

    # We compare 3 equivalent ways to create the constraint that
    # each point has to be within the ball of radius r centered at x
    def con_1(points, x, r):
        return [cp.norm(point - x) <= r for point in points]

    def con_2(points, x, r):
        return [cp.SOC(r, point - x) for point in points]

    def con_3(points, x, r):
        return [
            cp.SOC(
                r * np.ones(points.shape[0]),
                points - cp.outer(np.ones(points.shape[0]), x),
                axis=1,
            )
        ]

    def min_circle_cvx(points, fct=None, **kwargs):
        # Use con_1 if no constraint construction is defined
        fct = fct or con_3
        # cvxpy variable for the radius
        r = cp.Variable(1, name="Radius")
        # cvxpy variable for the midpoint
        x = cp.Variable(points.shape[1], name="Midpoint")

        objective = cp.Minimize(r)
        constraints = fct(points, x, r)

        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(**kwargs)

        return {"Radius": r.value, "Midpoint": x.value}

    return con_1, con_2, con_3, cp, min_circle_cvx


@app.cell
def _(min_circle_cvx, pos, stats, tt):
    print(min_circle_cvx(points=pos, solver="CLARABEL"))
    print(min_circle_cvx(points=pos, solver="MOSEK"))

    def cvx1():
        min_circle_cvx(points=pos, solver="CLARABEL")

    def cvx2():
        min_circle_cvx(points=pos, solver="MOSEK")

    # Run each 1000 times
    times_clarabel = tt.repeat(cvx1, number=1, repeat=50)
    times_cvx_mosek = tt.repeat(cvx2, number=1, repeat=50)

    print(f"Implementation cvxpy/clarabel: {stats.mean(times_clarabel):.6f} seconds")
    print(f"Implementation cvxpy/mosek: {stats.mean(times_cvx_mosek):.6f} seconds")
    return cvx1, cvx2, times_clarabel, times_cvx_mosek


@app.cell
def _(mo):
    mo.md("""## Compute with Mosek""")
    return


@app.cell
def _():
    import mosek.fusion as mf

    def min_circle_mosek(points, **kwargs):
        with mf.Model() as M:
            r = M.variable("Radius", 1)
            x = M.variable("Midpoint", [1, points.shape[1]])

            k = points.shape[0]

            # repeat the quantities
            R0 = mf.Var.repeat(r, k)
            X0 = mf.Var.repeat(x, k)

            # https://github.com/MOSEK/Tutorials/blob/master/minimum-ellipsoid/minimum-ellipsoid.ipynb
            M.constraint(
                mf.Expr.hstack(R0, mf.Expr.sub(X0, points)), mf.Domain.inQCone()
            )

            M.objective("obj", mf.ObjectiveSense.Minimize, r)
            M.solve()
            return {"Radius": r.level(), "Midpoint": x.level()}

    return mf, min_circle_mosek


@app.cell
def _(min_circle_mosek, pos, stats, tt):
    print(min_circle_mosek(points=pos))

    def mosek():
        min_circle_mosek(pos)

    # Run each 1000 times
    times_mosek = tt.repeat(mosek, number=1, repeat=50)

    print(f"Implementation average: {stats.mean(times_mosek):.6f} seconds")
    return mosek, times_mosek


@app.cell
def _(mo):
    mo.md("""## Compute with Hexaly""")
    return


@app.cell
def _(np):
    import hexaly.optimizer

    def min_circle_hexaly(points, **kwargs):
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            #
            # Declare the optimization model
            #
            model = optimizer.model

            z = np.array(
                [
                    model.float(np.min(points[:, j]), np.max(points[:, j]))
                    for j in range(points.shape[1])
                ]
            )

            radius = [np.sum((z - point) ** 2) for point in points]

            # Minimize the radius r
            r = model.sqrt(model.max(radius))
            model.minimize(r)
            model.close()

            optimizer.solve()
            return {
                "Radius": r.value,
                "Midpoint x": z[0].value,
                "Midpoint y": z[1].value,
            }

    return hexaly, min_circle_hexaly


@app.cell
def _(min_circle_hexaly, pos):
    min_circle_hexaly(points=pos)
    return


@app.cell
def _(mo):
    mo.md("""## Compute using Welzl's algorithm""")
    return


@app.cell
def _(pos, stats, tt):
    from solver.welzl import welzl_min_circle

    print(welzl_min_circle(points=list(pos)))

    def welzl():
        welzl_min_circle(points=list(pos))

    # Run each 50 times
    times_welzl = tt.repeat(welzl, number=1, repeat=100)

    print(f"Implementation average: {stats.mean(times_welzl):.6f} seconds")
    return times_welzl, welzl, welzl_min_circle


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
