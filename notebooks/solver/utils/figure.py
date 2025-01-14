import plotly.graph_objects as go


def create_figure():
    # Create the scatter plot
    fig = go.Figure()

    # Update layout for equal aspect ratio and axis labels
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    return fig
