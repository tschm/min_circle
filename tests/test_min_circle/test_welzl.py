"""Tests for the Welzl algorithm implementation.

This module contains tests for the Welzl algorithm functions used to find
the minimum enclosing circle for a set of points.
"""

import statistics
import timeit

import numpy as np
import pytest

from min_circle.welzl import (
    Circle,
    make_circle_n_points,
    min_circle_welzl,
    welzl_helper,
)


def test_one_point() -> None:
    """Test creating a circle from a single point.

    The circle should have radius 0 and center at the point.
    """
    p = np.array([3.0, 2.0])

    circle = make_circle_n_points([p])
    assert circle.radius == pytest.approx(0.0, rel=1e-9)
    assert circle.center == pytest.approx([3.0, 2.0], rel=1e-9)


def test_two_points() -> None:
    """Test creating a circle from two points.

    The circle should have its center at the midpoint between the two points.
    """
    p = np.array([[0.0, 0.0], [4.0, 2.0]])

    circle = make_circle_n_points(p)
    assert circle.center == pytest.approx([2.0, 1.0], rel=1e-9)


def test_three_points(points: list[np.ndarray]) -> None:
    """Test creating a circle from three points.

    Uses the fixture 'points' which provides three points in a triangle formation.

    Args:
        points: A list of three 2D points forming a triangle.
    """
    circle = make_circle_n_points(points)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-9)


def test_null_points() -> None:
    """Test creating a circle from an empty list of points.

    The circle should have an infinite radius when no points are provided.
    """
    p: list[np.ndarray] = []
    circle = make_circle_n_points(p)
    assert np.isinf(circle.radius)


def test_collinear() -> None:
    """Test creating a circle from three collinear points.

    When points are collinear, the circle should have its center at the middle point
    and radius equal to the distance to the furthest point.
    """
    p = [np.array([0, 0]), np.array([2.0, 2.0]), np.array([4.0, 4.0])]

    circle = make_circle_n_points(p)

    assert circle.radius == pytest.approx(np.sqrt(8), rel=1e-9)
    assert circle.center == pytest.approx(np.array([2.0, 2.0]), rel=1e-9)


def test_helper(points: list[np.ndarray]) -> None:
    """Test the welzl_helper function directly.

    This tests the recursive helper function used in the Welzl algorithm.

    Args:
        points: A list of three 2D points forming a triangle.
    """
    circle = welzl_helper(points, [], 3)
    assert circle.radius == pytest.approx(2.5, rel=1e-6)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-6)


def test_welzl_min_circle(points: list[np.ndarray]) -> None:
    """Test the min_circle_welzl function.

    This tests the main entry point for the Welzl algorithm.

    Args:
        points: A list of three 2D points forming a triangle.
    """
    circle = min_circle_welzl(points)

    assert circle.radius == pytest.approx(2.5, rel=1e-6)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-6)


def test_multiple_points() -> None:
    """Test the performance of the Welzl algorithm with many points.

    This test measures the execution time of the algorithm on a larger dataset
    of 800 random points.
    """
    np.random.seed(0)
    pos = np.random.randn(800, 2)

    def f() -> Circle:
        return min_circle_welzl(list(pos))

    times_welzl = timeit.repeat(f, number=1, repeat=100)
    # Calculate and print the mean execution time
    print(f"Mean execution time: {statistics.mean(times_welzl):.6f} seconds")


def test_too_many_points_raises_error() -> None:
    """Test that make_circle_n_points raises ValueError for more than 3 points."""
    p = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    with pytest.raises(ValueError, match="Expected 0-3 points, got 4"):
        make_circle_n_points(p)


def test_perpendicular_slope_horizontal() -> None:
    """Test perpendicular_slope with a horizontal line returns infinity."""
    from min_circle.welzl import perpendicular_slope

    p1 = np.array([0, 0])
    p2 = np.array([4, 0])  # Horizontal line
    slope = perpendicular_slope(p1, p2)
    assert np.isinf(slope)


def test_perpendicular_slope_diagonal() -> None:
    """Test perpendicular_slope with a diagonal line."""
    from min_circle.welzl import perpendicular_slope

    p1 = np.array([0, 0])
    p2 = np.array([2, 2])  # 45-degree line, slope = 1
    slope = perpendicular_slope(p1, p2)
    # Perpendicular to slope 1 should be -1
    assert slope == pytest.approx(-1.0, rel=1e-9)


def test_min_circle_welzl_with_numpy_array() -> None:
    """Test min_circle_welzl accepts numpy array input."""
    p = np.array([[0, 0], [4, 0], [2, 4]])
    circle = min_circle_welzl(p)
    assert circle.radius == pytest.approx(2.5, rel=1e-6)
    assert circle.center == pytest.approx([2.0, 1.5], rel=1e-6)
