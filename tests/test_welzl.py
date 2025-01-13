import statistics
import timeit

import numpy as np
import pytest

from notebooks.solver.welzl import (
    make_circle_n_points,
    welzl_min_circle,
    welzl_helper,
)


def test_one_point():
    p = np.array([3.0, 2.0])

    circle = make_circle_n_points([p])
    assert circle.radius == pytest.approx(0.0, rel=1e-9)
    assert circle.center == pytest.approx([3.0, 2.0], rel=1e-9)


def test_two_points():
    p = np.array([[0.0, 0.0], [4.0, 2.0]])

    circle = make_circle_n_points(p)
    print(circle)
    # assert circle.radius == pytest.approx(np.sqrt(5), rel=1e-9)
    assert circle.center == pytest.approx([2.0, 1.0], rel=1e-9)


def test_three_points(points):
    circle = make_circle_n_points(points)

    # assert circle.radius == pytest.approx(2.5, rel=1e-9)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-9)


def test_null_points():
    p = []
    circle = make_circle_n_points(p)
    assert np.isinf(circle.radius)


def test_collinear():
    # p = [Point(x=0.0, y=0.0), Point(x=2.0, y=2.0), Point(x=4.0, y=4.0)]
    p = [np.array([0, 0]), np.array([2.0, 2.0]), np.array([4.0, 4.0])]

    circle = make_circle_n_points(p)

    assert circle.radius == pytest.approx(np.sqrt(8), rel=1e-9)
    assert circle.center == pytest.approx(np.array([2.0, 2.0]), rel=1e-9)


def test_helper(points):
    circle = welzl_helper(points, [], 3)
    assert circle.radius == pytest.approx(2.5, rel=1e-9)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-9)


def test_welzl_min_circle(points):
    circle = welzl_min_circle(points)

    assert circle.radius == pytest.approx(2.5, rel=1e-9)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-9)


def test_multiple_points():
    np.random.seed(0)
    pos = np.random.randn(800, 2)

    def f():
        welzl_min_circle(list(pos))

    # print(circle)
    times_welzl = timeit.repeat(f, number=1, repeat=100)
    print(statistics.mean(times_welzl))
