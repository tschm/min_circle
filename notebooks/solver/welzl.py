from dataclasses import dataclass
import random
import numpy as np
from typing import List


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def array(self):
        return np.array([self.x, self.y])

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        assert isinstance(other, float)
        return Point(self.x * other, self.y * other)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


def norm(point: Point):
    return np.linalg.norm(point.array)


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float

    def contains(self, point: Point, tolerance: float = 1e-10) -> bool:
        """Check if a point lies within or on the circle."""
        assert isinstance(point, Point)
        return norm(point - self.center) <= self.radius + tolerance


# Calculate slopes of perpendicular bisectors
def perpendicular_slope(p1, p2):
    if p2[1] == p1[1]:  # horizontal line
        return np.inf  # Perpendicular line is horizontal
    return -(p2[0] - p1[0]) / (p2[1] - p1[1])


def make_circle_n_points(matrix: List[Point]) -> Circle:
    """Construct a circle with n points."""

    """Construct a circle from 1 to 3 points."""
    num_points = len(matrix)

    if num_points == 0:
        return Circle(Point(x=0, y=0), -np.inf)
        # return None  # No points, no circle

    if num_points == 1:
        return Circle(matrix[0], 0)  # Single point, radius 0

    if num_points == 2:
        # Two points: the center is the midpoint, radius is half the distance
        return Circle((matrix[0] + matrix[1]) * 0.5, norm(matrix[0] - matrix[1]) / 2.0)

    if num_points == 3:
        # For 3 points: use the circumcenter and circumradius formula
        # p1, p2, p3 = matrix
        p = np.array([x.array for x in matrix])

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
            # y = m1 * (x - mid12[0]) + mid12[1]
            # y = m2 * (x - mid23[0]) + mid23[1]
            A, B = m1, -1
            C, D = m2, -1
            E = m1 * mid12[0] - mid12[1]
            F = m2 * mid23[0] - mid23[1]

            # Solving for intersection using Cramer's rule:
            det = A * D - B * C

            if abs(det) < 1e-10:
                # compute the smallest and largest x
                max_x = p[:, 0].max()
                min_x = p[:, 0].min()

                max_y = p[:, 1].max()
                min_y = p[:, 1].min()

                center_x = (max_x + min_x) / 2
                center_y = (max_y + min_y) / 2

            else:
                center_x = (E * D - B * F) / det
                center_y = (A * F - E * C) / det

        center = Point(x=center_x, y=center_y)
        radius = np.linalg.norm(p - center.array, axis=1).max()

        return Circle(center=center, radius=radius)


def welzl_helper(points: List[Point], R: List[Point], n: int) -> Circle:
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


def welzl_min_circle(points: List[Point]) -> Circle:
    """
    Find the minimum enclosing circle using Welzl's algorithm.

    Args:
        points: List of points as numpy arrays
        seed: Random seed for reproducible results

    Returns:
        Circle object containing center coordinates and radius
    """
    # Make a copy of points to avoid modifying the input
    points = points.copy()
    # Shuffle the points randomly
    random.shuffle(points)

    return welzl_helper(points, [], len(points))

    # return minimum_circle_with_points(points, [], shuffle=True)
