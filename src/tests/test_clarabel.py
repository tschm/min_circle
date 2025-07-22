import numpy as np
import pytest

from min_circle.cvx import min_circle_cvx
from min_circle.utils.cloud import Cloud
from min_circle.utils.figure import create_figure
from min_circle.welzl import make_circle_n_points


def test_clarabel(points):
    cloud = Cloud(np.array(points))
    circle = min_circle_cvx(np.array(points), solver="CLARABEL")
    assert circle.radius == pytest.approx(2.5)
    assert circle.center == pytest.approx(np.array([2.0, 1.5]))

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    fig.show()


def test_vertical_12():
    p = np.array([np.array([0, 0]), np.array([0.0, 2.0]), np.array([4.0, 4.0])])
    cloud = Cloud(p)
    circle = min_circle_cvx(p, solver="CLARABEL")

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    fig.show()

    # assert circle.radius == pytest.approx(3.1622776601683795, rel=1e-9)
    # assert circle.center == pytest.approx(np.array([3.0, 1.0]), rel=1e-9)

    # p = [np.array([0, 0]), np.array([0.0, 2.0]), np.array([4.0, 4.0])]

    circle = make_circle_n_points(list(p))

    fig.add_trace(circle.scatter(color="black"))
    fig.show()


def test_vertical_23():
    p = np.array([[4.0, 4.0], [0, 0], [0.0, 2.0]])
    cloud = Cloud(p)
    circle = min_circle_cvx(p, solver="CLARABEL")

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    fig.show()

    # assert circle.radius == pytest.approx(3.1622776601683795, rel=1e-9)
    # assert circle.center == pytest.approx(np.array([3.0, 1.0]), rel=1e-9)

    # p = [np.array([0, 0]), np.array([0.0, 2.0]), np.array([4.0, 4.0])]

    circle = make_circle_n_points(list(p))

    fig.add_trace(circle.scatter(color="black"))
    fig.show()


def test_vertical():
    p = np.array([[0.0, 4.0], [0, 0], [0.0, 2.0]])
    cloud = Cloud(p)
    circle = min_circle_cvx(p, solver="CLARABEL")

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    fig.show()

    circle = make_circle_n_points(list(p))

    fig.add_trace(circle.scatter(color="black"))
    fig.show()


def test_random():
    p = np.array([[2.0, 4.0], [0, 0], [2.5, 2.0]])
    cloud = Cloud(p)
    circle = min_circle_cvx(p, solver="CLARABEL")

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    fig.show()

    circle = make_circle_n_points(list(p))

    fig.add_trace(circle.scatter(color="black"))
    fig.show()


def test_random_2():
    p = np.array([[0, 0.0], [3, 2], [6, 0.0]])
    cloud = Cloud(p)
    circle = min_circle_cvx(p, solver="CLARABEL")

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    fig.show()

    circle = make_circle_n_points(list(p))

    fig.add_trace(circle.scatter(color="black"))
    fig.show()
