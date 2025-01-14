import random
import numpy as np
from typing import List


from .utils.circle import Circle


# Calculate slopes of perpendicular bisectors
def perpendicular_slope(p1, p2):
    if p2[1] == p1[1]:  # horizontal line
        return np.inf  # Perpendicular line is horizontal
    return -(p2[0] - p1[0]) / (p2[1] - p1[1])


def make_circle_n_points(matrix: List[np.ndarray]) -> Circle:
    """Construct a circle with n points."""

    """Construct a circle from 1 to 3 points."""
    num_points = len(matrix)

    if num_points == 0:
        return Circle(np.array([0.0, 0.0]), -np.inf)

    if num_points == 1:
        return Circle(matrix[0], 0)  # Single point, radius 0

    if num_points == 2:
        # Two points: the center is the midpoint, radius is half the distance
        center = (matrix[0] + matrix[1]) / 2
        radius = np.linalg.norm(matrix[0] - matrix[1]) / 2

        return Circle(center=center, radius=radius)

    if num_points == 3:
        # For 3 points: use the circumcenter and circumradius formula
        # p1, p2, p3 = matrix
        p = np.array(matrix)

        # Midpoints of the sides
        mid12 = (p[0] + p[1]) * 0.5
        mid23 = (p[1] + p[2]) * 0.5

        # Slopes of the perpendicular bisectors
        # Perpendicular slope to line 1-2: m1
        m1 = perpendicular_slope(p[1], p[0])

        # Perpendicular slope to line 2-3: m2
        m2 = perpendicular_slope(p[1], p[2])

        # Use line equations to solve for the intersection (circumcenter)
        if m1 == np.inf:  # Line 1-2 is vertical, so we solve for x = mid12[0]
            center_x = mid12[0]
            center_y = m2 * (center_x - mid23[0]) + mid23[1]
        elif m2 == np.inf:  # Line 2-3 is vertical, so we solve for x = mid23[0]
            center_x = mid23[0]
            center_y = m1 * (center_x - mid12[0]) + mid12[1]
        else:
            # Calculate the intersection of the two perpendicular bisectors
            A, B = m1, -1
            C, D = m2, -1
            E = m1 * mid12[0] - mid12[1]
            F = m2 * mid23[0] - mid23[1]

            # Construct the coefficient matrix and the right-hand side vector
            coeff_matrix = np.array([[A, B], [C, D]])
            rhs = np.array([E, F])

            try:
                center_x, center_y = np.linalg.solve(coeff_matrix, rhs)
            except np.linalg.LinAlgError:
                max_x = p[:, 0].max()
                min_x = p[:, 0].min()

                max_y = p[:, 1].max()
                min_y = p[:, 1].min()

                center_x = (max_x + min_x) / 2
                center_y = (max_y + min_y) / 2

        center = np.array([center_x, center_y])
        radius = np.linalg.norm(p - center, axis=1).max()

        return Circle(center=center, radius=radius)


def welzl_helper(points: List[np.ndarray], R: List[np.ndarray], n: int) -> Circle:
    """Recursive helper function for Welzl's algorithm."""
    if n == 0 or len(R) == 3:
        return make_circle_n_points(R)

    # Remove a random point
    idx = random.randrange(n)
    p = points[idx]
    points[idx], points[n - 1] = points[n - 1], points[idx]

    # Recursively compute the minimum circle without p
    circle = welzl_helper(points, R, n - 1)

    # If p is inside the circle, we're done
    if circle.contains(p):
        return circle

    # Otherwise, p must be on the boundary of the minimum enclosing circle
    R.append(p)
    circle = welzl_helper(points, R, n - 1)
    R.pop()

    return circle


def min_circle_welzl(points: np.ndarray) -> Circle:
    """
    Find the minimum enclosing circle using Welzl's algorithm.

    Args:
        points: List of points as numpy arrays

    Returns:
        Circle object containing center coordinates and radius
    """
    if isinstance(points, np.ndarray):
        points = list(points)

    # Make a copy of points to avoid modifying the input
    points = points.copy()
    # Shuffle the points randomly
    random.shuffle(points)

    return welzl_helper(points, [], len(points))
