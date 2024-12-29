import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Implement with cvxpy""")
    return


@app.cell
def _():
    import plotly.graph_objects as go
    import numpy as np
    return go, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Generate a cloud of points""")
    return


@app.cell
def _(np):
    pos = np.random.randn(1000,3)
    return (pos,)


@app.cell
def _(go, pos):
    # Create the scatter plot
    fig = go.Figure(data=go.Scatter(
        x=pos[:,0],
        y=pos[:,1],
        mode='markers',
        marker=dict(
            symbol='x',
            size=10
        )
    ))

    # Update layout for equal aspect ratio and axis labels
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )

    # Show the plot
    fig.show()
    return (fig,)


@app.cell
def _():
    import cvxpy as cp

    def min_circle_cvx(points, **kwargs):
        # cvxpy variable for the radius
        r = cp.Variable(1, name="Radius")
        # cvxpy variable for the midpoint
        x = cp.Variable(points.shape[1], name="Midpoint")

        objective = cp.Minimize(r)
        constraints = [cp.norm(point - x) <= r for point in points]

        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(**kwargs)

        return {"Radius": r.value, "Midpoint": x.value}
    return cp, min_circle_cvx


@app.cell
def _(cp):
    def min_circle_cvx_cone(points, **kwargs):
        # cvxpy variable for the radius
        r = cp.Variable(1, name="Radius")
        # cvxpy variable for the midpoint
        x = cp.Variable(points.shape[1], name="Midpoint")

        objective = cp.Minimize(r)
        #constraints = [cp.SOC(np.ones(points.shape[0])*r, points - x, axis=1)]
        constraints = [cp.SOC(r, point - x) for point in points]

        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(**kwargs)

        return {"Radius": r.value, "Midpoint": x.value}
    return (min_circle_cvx_cone,)


@app.cell
def _(min_circle_cvx, pos):
    min_circle_cvx(points=pos, solver="CLARABEL")
    return


@app.cell
def _(min_circle_cvx_cone, pos):
    min_circle_cvx_cone(points=pos, solver="CLARABEL")
    return


@app.cell
def _():
    import mosek.fusion as mf

    def min_circle_mosek(points, **kwargs):
        with mf.Model() as M:
            r = M.variable("Radius", 1)
            x = M.variable("Midpoint", points.shape[1])

            # see https://docs.mosek.com/latest/pythonfusion/modeling.html#vectorization
            for i, p in enumerate(points):
                M.constraint(f"point_{i}", mf.Expr.vstack(r, mf.Expr.sub(x,p)), mf.Domain.inQCone())

            M.objective('obj', mf.ObjectiveSense.Minimize, r)
            M.solve()
            return {"Radius": r.level(), "Midpoint": x.level()}
    return mf, min_circle_mosek


@app.cell
def _(min_circle_mosek, pos):
    min_circle_mosek(points=pos)
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

            z = np.array([model.float(np.min(points[:,j]), np.max(points[:,j])) for j in range(points.shape[1])])

            radius = [np.sum((z - point) ** 2) for point in points]

            # Minimize the radius r
            r = model.sqrt(model.max(radius))
            model.minimize(r)
            model.close()

            optimizer.solve()
            return {"Radius": r.value, "Midpoint": z.level}
    return hexaly, min_circle_hexaly


@app.cell
def _(min_circle_hexaly, pos):
    min_circle_hexaly(points=pos)
    return


@app.cell
def _():
    return


@app.cell
def _(min_circle_cvx, pos):
    min_circle_cvx(points=pos, solver="MOSEK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Implement with Mosek Fusion""")
    return


if __name__ == "__main__":
    app.run()
