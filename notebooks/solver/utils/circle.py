from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class Circle:
    center: np.ndarray
    radius: float

    def __post_init__(self):
        assert self.center.shape == (2,)

    def contains(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a point lies within or on the circle."""
        return np.linalg.norm(point - self.center) <= self.radius + tolerance

    def scatter(self, num=100, color="red"):
        t = np.linspace(0, 2 * np.pi, num=num)
        radius = self.radius
        circle_x = self.center[0] + radius * np.cos(t)
        circle_y = self.center[1] + radius * np.sin(t)

        return go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line={"color": color, "width": 2},
            name=f"Circle(r = {self.radius})",
        )
