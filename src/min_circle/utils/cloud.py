"""Point cloud utility class for representing and visualizing sets of points.

This module provides a Cloud class for representing collections of 2D points
and methods for visualizing them.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class Cloud:
    """A 2D point cloud representation.

    This class represents a collection of points in 2D space and provides
    methods for visualizing them using Plotly.

    Attributes:
        points: A numpy array of shape (n, 2) representing n points in 2D space
    """

    points: np.ndarray

    def __post_init__(self) -> None:
        """Validate that the points array has the correct shape.

        Raises:
            AssertionError: If the points array doesn't have shape (n, 2)
        """
        assert len(self.points.shape) == 2, "Points must be a 2D array"
        assert self.points.shape[1] == 2, "Points must have shape (n, 2)"

    def scatter(self, size: int = 10) -> go.Scatter:
        """Create a Plotly Scatter trace representing the point cloud.

        Args:
            size: Size of the marker points

        Returns:
            A Plotly Scatter object that can be added to a figure
        """
        return go.Scatter(
            x=self.points[:, 0],
            y=self.points[:, 1],
            mode="markers",
            marker={"symbol": "x", "size": size, "color": "blue"},
            name=f"Points ({len(self.points)})",
        )
