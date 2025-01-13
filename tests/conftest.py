import numpy as np
import pytest

from notebooks.solver.welzl import Point


@pytest.fixture()
def points():
    p1 = Point(x=0, y=0)
    p2 = Point(x=4, y=0)
    p3 = Point(x=2, y=4)
    return list([p1, p2, p3])


@pytest.fixture()
def np_points(points):
    return np.array([[p.x, p.y] for p in points])
