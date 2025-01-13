import numpy as np
import pytest

from notebooks.solver.welzl import make_circle_n_points


def test_one_point():
    p = np.array([[3.0, 2.0]])
    circle = make_circle_n_points(p)
    assert circle.radius == pytest.approx(0.0, rel=1e-9)
    assert circle.center == pytest.approx([3.0, 2.0], rel=1e-9)


def test_two_points():
    p = np.array([[0.0, 0.0], [4.0, 2.0]])
    circle = make_circle_n_points(np.array(p))

    assert circle.radius == pytest.approx(np.sqrt(5), rel=1e-9)
    assert circle.center == pytest.approx([2.0, 1.0], rel=1e-9)


def test_three_points():
    p = np.array([[0, 0], [4.0, 0.0], [2.0, 4.0]])
    circle = make_circle_n_points(p)

    assert circle.radius == pytest.approx(2.5, rel=1e-9)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-9)


def test_null_points():
    p = np.array([])
    circle = make_circle_n_points(p)
    assert circle is None


def test_collinear():
    p = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    assert make_circle_n_points(p) is None
