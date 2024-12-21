import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import plotly.graph_objects as go
    import numpy as np
    import cvxpy as cp
    return cp, go, np


@app.cell
def _(go, np):
    pos = np.random.randn(2000,20)

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
    return fig, pos


@app.cell
def _(cp):
    def min_circle(points, **kwargs):
        # cvxpy variable for the radius
        r = cp.Variable(1, name="Radius")
        # cvxpy variable for the midpoint
        x = cp.Variable(points.shape[1], name="Midpoint")

        objective = cp.Minimize(r)
        constraints = [cp.norm(point - x) <= r for point in points]

        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(**kwargs)

        return {"Radius": r.value, "Midpoint": x.value}

    return (min_circle,)


@app.cell
def _(min_circle, pos):
    min_circle(points=pos)
    return


if __name__ == "__main__":
    app.run()
