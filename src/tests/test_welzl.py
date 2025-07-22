import statistics
import timeit

import numpy as np
import pytest

from min_circle.welzl import (
    make_circle_n_points,
    min_circle_welzl,
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
    p = [np.array([0, 0]), np.array([2.0, 2.0]), np.array([4.0, 4.0])]

    circle = make_circle_n_points(p)

    assert circle.radius == pytest.approx(np.sqrt(8), rel=1e-9)
    assert circle.center == pytest.approx(np.array([2.0, 2.0]), rel=1e-9)


def test_helper(points):
    circle = welzl_helper(points, [], 3)
    assert circle.radius == pytest.approx(2.5, rel=1e-6)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-6)


def test_welzl_min_circle(points):
    circle = min_circle_welzl(points)

    assert circle.radius == pytest.approx(2.5, rel=1e-6)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-6)


def test_multiple_points():
    np.random.seed(0)
    pos = np.random.randn(800, 2)

    def f():
        min_circle_welzl(list(pos))

    # print(circle)
    times_welzl = timeit.repeat(f, number=1, repeat=100)
    print(statistics.mean(times_welzl))


# def test_vertical_12():
#     p = [np.array([0, 0]), np.array([0.0, 2.0]), np.array([4.0, 4.0])]
#
#     circle = make_circle_n_points(p)
#
#     assert circle.radius == pytest.approx(3.1622776601683795, rel=1e-9)
#     assert circle.center == pytest.approx(np.array([3.0, 1.0]), rel=1e-9)


# def test_vertical_23():
#    p = [np.array([1, 0]), np.array([4.0, 2.0]), np.array([4.0, 4.0])]
#
#    circle = make_circle_n_points(p)
#
#    assert circle.radius == pytest.approx(np.sqrt(8), rel=1e-9)
#    assert circle.center == pytest.approx(np.array([2.0, 2.0]), rel=1e-9)
