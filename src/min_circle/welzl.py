"""Welzl's algorithm implementation for finding the minimum enclosing circle.

This module provides an implementation of Welzl's randomized algorithm
for finding the minimum enclosing circle of a set of points in 2D space.
"""

import secrets

import numpy as np

from .cvx import min_circle_cvx
from .utils.circle import Circle


def perpendicular_slope(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate the slope of the perpendicular bisector between two points.

    Args:
        p1: First point as a numpy array [x, y]
        p2: Second point as a numpy array [x, y]

    Returns:
        The slope of the perpendicular bisector line
    """
    if p2[1] == p1[1]:  # horizontal line
        return float(np.inf)  # Perpendicular bisector is vertical
    return float(-(p2[0] - p1[0]) / (p2[1] - p1[1]))


def make_circle_n_points(matrix: list[np.ndarray]) -> Circle:
    """Construct a circle from 0 to 3 points.

    Args:
        matrix: List of points (up to 3) as numpy arrays

    Returns:
        Circle object containing the center and radius of the circle
    """
    num_points = len(matrix)

    if num_points == 0:
        return Circle(np.array([0.0, 0.0]), -np.inf)

    if num_points == 1:
        return Circle(matrix[0], 0)  # Single point, radius 0

    if num_points == 2 or num_points == 3:
        return min_circle_cvx(np.array(matrix), solver="CLARABEL")

    raise ValueError(f"Expected 0-3 points, got {num_points}")

    # if num_points == 2:
    #    # Two points: the center is the midpoint, radius is half the distance
    #    center = (matrix[0] + matrix[1]) / 2
    #    radius = np.linalg.norm(matrix[0] - matrix[1]) / 2
    #
    #    return Circle(center=center, radius=radius)

    # if num_points == 3:
    #    # For 3 points: use the circumcenter and circumradius formula
    #    # p1, p2, p3 = matrix
    #    p = np.array(matrix)
    #
    #    # Midpoints of the sides
    #    mid12 = (p[0] + p[1]) * 0.5
    #    mid23 = (p[1] + p[2]) * 0.5
    #
    #    # Slopes of the perpendicular bisectors
    #    # Perpendicular slope to line 1-2: m1
    #    m1 = perpendicular_slope(p[1], p[0])
    #
    #    # Perpendicular slope to line 2-3: m2
    #    m2 = perpendicular_slope(p[1], p[2])
    #
    #    # Use line equations to solve for the intersection (circumcenter)
    #    if m1 == np.inf:  # Line 1-2 is vertical, so we solve for x = mid12[0]
    #        center_x = mid12[0]
    #        center_y = m2 * (center_x - mid23[0]) + mid23[1]
    #    elif m2 == np.inf:  # Line 2-3 is vertical, so we solve for x = mid23[0]
    #        center_x = mid23[0]
    #        center_y = m1 * (center_x - mid12[0]) + mid12[1]
    #    else:
    #        # Calculate the intersection of the two perpendicular bisectors
    #        A, B = m1, -1
    #        C, D = m2, -1
    #        E = m1 * mid12[0] - mid12[1]
    #        F = m2 * mid23[0] - mid23[1]
    #
    #        # Construct the coefficient matrix and the right-hand side vector
    #        coeff_matrix = np.array([[A, B], [C, D]])
    #        rhs = np.array([E, F])

    #        try:
    #            center_x, center_y = np.linalg.solve(coeff_matrix, rhs)
    #        except np.linalg.LinAlgError:
    #            max_x = p[:, 0].max()
    #            min_x = p[:, 0].min()
    #
    #            max_y = p[:, 1].max()
    #            min_y = p[:, 1].min()

    #            center_x = (max_x + min_x) / 2
    #            center_y = (max_y + min_y) / 2

    #    center = np.array([center_x, center_y])
    #    radius = np.linalg.norm(p - center, axis=1).max()

    #    return Circle(center=center, radius=radius)


def welzl_helper(points: list[np.ndarray], r: list[np.ndarray], n: int) -> Circle:
    """Recursive helper function for Welzl's algorithm."""
    if n == 0 or len(r) == 3:
        return make_circle_n_points(r)

    # Remove a random point by shuffling it to the end
    # we know at this stage that n > 0
    idx = secrets.SystemRandom().randrange(n)
    p = points[idx]
    points[idx], points[n - 1] = points[n - 1], points[idx]

    # Recursively compute the minimum circle without p
    # This is drilling down and open welzl_helper for each individual p!
    # R remains empty for now
    circle = welzl_helper(points, r, n - 1)
    # finally it will arrive at n == 0 and R = []
    # It now calls make_circle_n_points with R = []
    # It returns a circle with radius -inf
    # and lands back here where
    # n = 1
    # p is the final surviving point
    # and the circle has radius -inf
    # obviously p is not(!) contained in the circle!

    # If p is inside the circle, we're done
    if circle.contains(p):
        return circle

    # Otherwise, p must be on the boundary of the minimum enclosing circle
    # now we add this final point, points is still empty
    r.append(p)
    circle = welzl_helper(points, r, n - 1)
    r.pop()

    return circle


def min_circle_welzl(points: np.ndarray | list[np.ndarray]) -> Circle:
    """Find the minimum enclosing circle using Welzl's algorithm.

    This is the main entry point for Welzl's randomized algorithm. It takes a set of points
    and returns the smallest circle that contains all the points.

    Args:
        points: Array or list of 2D points

    Returns:
        Circle object containing the center and radius of the minimum enclosing circle
    """
    if isinstance(points, np.ndarray):
        points = list(points)

    # Make a copy of points to avoid modifying the input
    points = points.copy()
    # Shuffle the points randomly
    secrets.SystemRandom().shuffle(points)

    return welzl_helper(points, [], len(points))
