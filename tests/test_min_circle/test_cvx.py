"""Tests for the CVXPY implementation.

This module contains tests for the parameterized CVXPY problem functions
and additional coverage for the cvx module.
"""

import numpy as np
import pytest

from min_circle.cvx import min_circle_cvx, min_circle_cvx_2, min_circle_cvx_3


class TestMinCircleCvx2:
    """Tests for the min_circle_cvx_2 parameterized problem."""

    def test_creates_problem(self) -> None:
        """Test that min_circle_cvx_2 creates a valid CVXPY problem."""
        problem = min_circle_cvx_2()
        assert problem is not None
        assert "points" in problem.param_dict
        assert "Radius" in problem.var_dict
        assert "Midpoint" in problem.var_dict

    def test_solves_two_points(self) -> None:
        """Test that the problem solves correctly for two points."""
        p = np.array([[0.0, 0.0], [4.0, 0.0]])
        problem = min_circle_cvx_2()
        problem.param_dict["points"].value = p
        problem.solve(solver="CLARABEL")

        radius = problem.var_dict["Radius"].value
        center = problem.var_dict["Midpoint"].value

        assert radius == pytest.approx(2.0, rel=1e-6)
        assert center.flatten() == pytest.approx([2.0, 0.0], rel=1e-6)


class TestMinCircleCvx3:
    """Tests for the min_circle_cvx_3 parameterized problem."""

    def test_creates_problem(self) -> None:
        """Test that min_circle_cvx_3 creates a valid CVXPY problem."""
        problem = min_circle_cvx_3()
        assert problem is not None
        assert "points" in problem.param_dict
        assert "Radius" in problem.var_dict
        assert "Midpoint" in problem.var_dict

    def test_solves_three_points(self) -> None:
        """Test that the problem solves correctly for three points."""
        p = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
        problem = min_circle_cvx_3()
        problem.param_dict["points"].value = p
        problem.solve(solver="CLARABEL")

        radius = problem.var_dict["Radius"].value
        assert radius > 0

    def test_solves_equilateral_triangle(self) -> None:
        """Test with an equilateral triangle."""
        # Equilateral triangle with side length 2
        p = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3)]])
        problem = min_circle_cvx_3()
        problem.param_dict["points"].value = p
        problem.solve(solver="CLARABEL")

        radius = problem.var_dict["Radius"].value
        # Circumradius of equilateral triangle = side / sqrt(3)
        expected_radius = 2.0 / np.sqrt(3)
        assert radius == pytest.approx(expected_radius, rel=1e-5)


class TestMinCircleCvxDirect:
    """Additional tests for the direct min_circle_cvx function."""

    def test_single_point(self) -> None:
        """Test with a single point."""
        p = np.array([[1.0, 2.0]])
        circle = min_circle_cvx(p, solver="CLARABEL")
        assert circle.radius == pytest.approx(0.0, abs=1e-6)
        assert circle.center == pytest.approx([1.0, 2.0], rel=1e-6)

    def test_many_points(self) -> None:
        """Test with many random points."""
        np.random.seed(42)
        p = np.random.randn(50, 2)
        circle = min_circle_cvx(p, solver="CLARABEL")

        # All points should be inside or on the circle
        distances = np.linalg.norm(p - circle.center, axis=1)
        assert all(d <= circle.radius + 1e-5 for d in distances)
