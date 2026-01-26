"""Circle utility class for representing and visualizing circles.

This module provides a Circle class for representing circles in 2D space,
with methods for checking point containment and visualization.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class Circle:
    """A 2D circle representation.

    This class represents a circle in 2D space, defined by its center coordinates
    and radius. It provides methods for checking if points are contained within
    the circle and for visualizing the circle using Plotly.

    Attributes:
        center: A numpy array of shape (2,) representing the x,y coordinates of the center
        radius: The radius of the circle
    """

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        """Validate that the center is a 2D point.

        Raises:
            AssertionError: If the center is not a 2D point (shape != (2,))
        """
        assert self.center.shape == (2,), "Center must be a 2D point with shape (2,)"

    def contains(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a point lies within or on the circle.

        Args:
            point: A numpy array representing the point to check
            tolerance: A small value to account for floating-point errors

        Returns:
            True if the point is inside or on the circle boundary, False otherwise
        """
        return np.linalg.norm(point - self.center) <= self.radius + tolerance

    def scatter(self, num: int = 100, color: str = "red") -> go.Scatter:
        """Create a Plotly Scatter trace representing the circle.

        Args:
            num: Number of points to use for drawing the circle
            color: Color of the circle line

        Returns:
            A Plotly Scatter object that can be added to a figure
        """
        # Generate points along the circle
        t = np.linspace(0, 2 * np.pi, num=num)
        radius = self.radius
        circle_x = self.center[0] + radius * np.cos(t)
        circle_y = self.center[1] + radius * np.sin(t)

        # Create and return the Scatter trace
        return go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line={"color": color, "width": 2},
            name=f"Circle(r = {self.radius:.3f})",
        )
