"""Figure utility module for creating and configuring Plotly figures.

This module provides functions for creating pre-configured Plotly figures
for visualizing circles and point clouds.
"""

import plotly.graph_objects as go


def create_figure() -> go.Figure:
    """Create a pre-configured Plotly figure with equal aspect ratio.

    Creates a new Plotly figure with an equal aspect ratio between x and y axes,
    which is important for correctly visualizing circles.

    Returns:
        A configured Plotly Figure object ready for adding traces
    """
    # Create an empty figure
    fig = go.Figure()

    # Update layout for equal aspect ratio and axis labels
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        # Ensure equal scaling for x and y axes (important for circles)
        yaxis={
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )

    return fig
