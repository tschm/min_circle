import hexaly.optimizer
import numpy as np

from .utils.circle import Circle


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
        return Circle(radius=r.value, center=np.array([z[0].value, z[1].value]))
