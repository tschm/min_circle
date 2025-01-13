from dataclasses import dataclass
import random
import numpy as np
from typing import List, Optional


@dataclass(frozen=True)
class Circle:
    center: np.ndarray
    radius: float

    def contains(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a point lies within or on the circle."""
        return np.linalg.norm(point - self.center) <= self.radius + tolerance


def make_circle_n_points(matrix: np.ndarray) -> Circle:
    """Construct a circle with n points."""
    assert matrix.shape[0] <= 3

    if matrix.shape[0] == 0:
        return None

    if matrix.shape[0] == 1:
        return Circle(matrix[0], 0)

    if matrix.shape[0] == 2:
        center = (matrix[0] + matrix[1]) / 2
        radius = np.linalg.norm(matrix[0] - center)
        return Circle(center, radius)

    if matrix.shape[0] == 3:
        p1 = matrix[0, :]
        p2 = matrix[1, :]
        p3 = matrix[2, :]

        # Midpoints of the sides
        mid12 = (p1 + p2) / 2
        mid23 = (p2 + p3) / 2

        # Slopes of the perpendicular bisectors
        # Perpendicular slope to line 1-2: m1
        m1 = -(p2[0] - p1[0]) / (p2[1] - p1[1]) if p2[1] - p1[1] != 0 else np.inf
        # Perpendicular slope to line 2-3: m2
        m2 = -(p3[0] - p2[0]) / (p3[1] - p2[1]) if p3[1] - p2[1] != 0 else np.inf

        # Use line equations to solve for the intersection (circumcenter)
        if m1 == np.inf:  # Line 1-2 is vertical, so we solve for x = mid12[0]
            center_x = mid12[0]
            center_y = m2 * (center_x - mid23[0]) + mid23[1]
        elif m2 == np.inf:  # Line 2-3 is vertical, so we solve for x = mid23[0]
            center_x = mid23[0]
            center_y = m1 * (center_x - mid12[0]) + mid12[1]
        else:
            # Calculate the intersection of the two perpendicular bisectors
            # y = m1 * (x - mid12[0]) + mid12[1]
            # y = m2 * (x - mid23[0]) + mid23[1]
            A = m1
            B = -1
            C = m2
            D = -1
            E = m1 * mid12[0] - mid12[1]
            F = m2 * mid23[0] - mid23[1]

            # Solving for intersection using Cramer's rule:
            det = A * D - B * C

            if abs(det) < 1e-10:
                #    # take the third point out
                return None
            #    raise AssertionError("Points are collinear")

            center_x = (E * D - B * F) / det
            center_y = (A * F - E * C) / det

        center = np.array([center_x, center_y])

        # Calculate radius using the distance from the center to any vertex (e.g., p1)
        radius = np.linalg.norm(p1 - center)

        return Circle(center, radius)

        # return make_circle_three_points(matrix[0], matrix[1], matrix[2])


def minimum_circle_with_points(
    points: List[np.ndarray], R: List[np.ndarray], shuffle: bool = True
) -> Circle:
    """
    Find the minimum enclosing circle for points with points R on the boundary.

    Args:
        points: List of points to process
        R: List of points known to be on the boundary
        shuffle: Whether to shuffle the points (should be True only for initial call)
    """
    # Base case: If 3 boundary points are available, return a circle
    # if len(R) == 3:
    #    circle = make_circle_n_points(np.array(R[0], R[1], R[2]))
    #    if circle is None:
    #        # If the points are collinear, fall back to using two points
    #        return make_circle_n_points(np.array(R[0], R[1]))
    #    return circle

    # Base case: If no points are left, return the smallest circle possible
    # if len(points) == 0:
    #    if len(R) == 0:
    #        return None
    #    elif len(R) == 1:
    #        return make_circle_n_points(np.array(R))
    # else:  # len(R) == 2
    #    return make_circle_n_points(np.array(R[0], R[1]))

    if len(R) <= 3:
        return make_circle_n_points(np.array(R))

    # Take the last point (no need to shuffle again)
    p = points.pop()

    # Recursively compute minimum circle without p
    circle = minimum_circle_with_points(points, R, shuffle=False)

    # If p is not in the circle, it must be on the boundary
    if circle is None or not circle.contains(p):
        R.append(p)
        circle = minimum_circle_with_points(points, R, shuffle=False)

    return circle


def welzl_min_circle(points: List[np.ndarray], seed: Optional[int] = None) -> Circle:
    """
    Find the minimum enclosing circle using Welzl's algorithm.

    Args:
        points: List of points as numpy arrays
        seed: Random seed for reproducible results

    Returns:
        Circle object containing center coordinates and radius
    """
    # Ensure consistent randomization by setting a fixed seed
    if seed is not None:
        random.seed(seed)

    return minimum_circle_with_points(points, [], shuffle=True)
