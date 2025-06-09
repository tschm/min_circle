from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class Cloud:
    points: np.ndarray

    def scatter(self, size=10):
        return go.Scatter(
            x=self.points[:, 0],
            y=self.points[:, 1],
            mode="markers",
            marker={"symbol": "x", "size": size, "color": "blue"},
        )
