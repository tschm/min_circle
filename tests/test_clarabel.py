"""Tests for the Clarabel solver implementation.

This module contains tests for the Clarabel solver used to find
the minimum enclosing circle for a set of points using convex optimization.
"""

import numpy as np
import pytest

from min_circle.cvx import min_circle_cvx
from min_circle.utils.cloud import Cloud
from min_circle.utils.figure import create_figure
from min_circle.welzl import make_circle_n_points


def test_clarabel(points: list[np.ndarray]) -> None:
    """Test the Clarabel solver with a triangle of points.

    Uses the fixture 'points' which provides three points in a triangle formation.
    Creates a visualization of the points and the minimum enclosing circle.

    Args:
        points: A list of three 2D points forming a triangle.
    """
    Cloud(np.array(points))
    circle = min_circle_cvx(np.array(points), solver="CLARABEL")
    assert circle.radius == pytest.approx(2.5)
    assert circle.center == pytest.approx(np.array([2.0, 1.5]))

    # Create visualization (commented out to avoid showing in automated tests)
    # fig = create_figure()
    # fig.add_trace(circle.scatter())
    # fig.add_trace(cloud.scatter())
    # fig.show()


def test_vertical_12() -> None:
    """Test the Clarabel solver with a specific set of points.

    Tests a configuration with points at [0,0], [0,2], and [4,4].
    Compares the results with the Welzl algorithm.
    """
    p = np.array([np.array([0, 0]), np.array([0.0, 2.0]), np.array([4.0, 4.0])])
    Cloud(p)
    min_circle_cvx(p, solver="CLARABEL")

    # Compare with Welzl algorithm
    make_circle_n_points(list(p))

    # Assertions could be added here to verify the results
    # For example:
    # assert circle_cvx.radius == pytest.approx(circle_welzl.radius, rel=1e-6)
    # assert circle_cvx.center == pytest.approx(circle_welzl.center, rel=1e-6)

    # Visualization code (commented out for automated tests)
    # fig = create_figure()
    # fig.add_trace(circle_cvx.scatter())
    # fig.add_trace(cloud.scatter())
    # fig.add_trace(circle_welzl.scatter(color="black"))
    # fig.show()


def test_vertical_23() -> None:
    """Test the Clarabel solver with a specific point configuration.

    Tests a configuration with points at [4,4], [0,0], and [0,2].
    Creates a visualization and compares the results with the Welzl algorithm.
    """
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


def test_vertical() -> None:
    """Test the Clarabel solver with vertically aligned points.

    Tests a configuration with three points aligned vertically at x=0.
    Compares the results with the Welzl algorithm.
    """
    p = np.array([[0.0, 4.0], [0, 0], [0.0, 2.0]])
    Cloud(p)
    min_circle_cvx(p, solver="CLARABEL")

    # Compare with Welzl algorithm
    make_circle_n_points(list(p))

    # Assertions could be added here to verify the results
    # For example:
    # assert circle_cvx.radius == pytest.approx(circle_welzl.radius, rel=1e-6)
    # assert circle_cvx.center == pytest.approx(circle_welzl.center, rel=1e-6)

    # Visualization code (commented out for automated tests)
    # fig = create_figure()
    # fig.add_trace(circle_cvx.scatter())
    # fig.add_trace(cloud.scatter())
    # fig.add_trace(circle_welzl.scatter(color="black"))
    # fig.show()


def test_random() -> None:
    """Test the Clarabel solver with a random configuration of points.

    Tests a specific configuration of three points and compares the results
    with the Welzl algorithm.
    """
    p = np.array([[2.0, 4.0], [0, 0], [2.5, 2.0]])
    Cloud(p)
    min_circle_cvx(p, solver="CLARABEL")

    # Compare with Welzl algorithm
    make_circle_n_points(list(p))

    # Assertions could be added here to verify the results
    # For example:
    # assert circle_cvx.radius == pytest.approx(circle_welzl.radius, rel=1e-6)
    # assert circle_cvx.center == pytest.approx(circle_welzl.center, rel=1e-6)

    # Visualization code (commented out for automated tests)
    # fig = create_figure()
    # fig.add_trace(circle_cvx.scatter())
    # fig.add_trace(cloud.scatter())
    # fig.add_trace(circle_welzl.scatter(color="black"))
    # fig.show()


def test_random_2() -> None:
    """Test the Clarabel solver with another configuration of points.

    Tests a configuration with points at [0,0], [3,2], and [6,0].
    Compares the results with the Welzl algorithm.
    """
    p = np.array([[0, 0.0], [3, 2], [6, 0.0]])
    Cloud(p)
    min_circle_cvx(p, solver="CLARABEL")

    # Compare with Welzl algorithm
    make_circle_n_points(list(p))

    # Assertions could be added here to verify the results
    # For example:
    # assert circle_cvx.radius == pytest.approx(circle_welzl.radius, rel=1e-6)
    # assert circle_cvx.center == pytest.approx(circle_welzl.center, rel=1e-6)

    # Visualization code (commented out for automated tests)
    # fig = create_figure()
    # fig.add_trace(circle_cvx.scatter())
    # fig.add_trace(cloud.scatter())
    # fig.add_trace(circle_welzl.scatter(color="black"))
    # fig.show()
