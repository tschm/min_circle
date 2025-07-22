import numpy as np
import pytest


@pytest.fixture()
def points():
    p1 = np.array([0, 0])
    p2 = np.array([4, 0])
    p3 = np.array([2, 4])
    return list([p1, p2, p3])
