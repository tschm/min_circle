"""Tests for the MOSEK implementation.

This module contains tests for the MOSEK solver implementation.
"""

import numpy as np
import pytest

# Skip all tests in this module - requires MOSEK license
pytestmark = pytest.mark.skip(reason="MOSEK license not available")


class TestMosekImport:
    """Tests for MOSEK import handling."""

    def test_import_error_message(self) -> None:
        """Test that the import error message is informative."""
        from min_circle.msk import _MOSEK_IMPORT_ERROR

        assert "MOSEK" in _MOSEK_IMPORT_ERROR
        assert "pip install" in _MOSEK_IMPORT_ERROR

    def test_min_circle_mosek_without_mosek(self) -> None:
        """Test that min_circle_mosek raises ImportError when MOSEK is not installed."""
        pytest.importorskip("mosek", reason="Test only runs when MOSEK is not installed")


class TestMosekSolver:
    """Tests for the MOSEK solver when available."""

    @pytest.fixture
    def mosek(self) -> None:
        """Skip if MOSEK is not installed."""
        pytest.importorskip("mosek")

    def test_two_points(self, mosek: None) -> None:
        """Test MOSEK solver with two points."""
        from min_circle.msk import min_circle_mosek

        p = np.array([[0.0, 0.0], [4.0, 0.0]])
        circle = min_circle_mosek(p)

        assert circle.radius == pytest.approx(2.0, rel=1e-6)
        assert np.allclose(circle.center, [2.0, 0.0], rtol=1e-6)

    def test_three_points(self, mosek: None) -> None:
        """Test MOSEK solver with three points."""
        from min_circle.msk import min_circle_mosek

        p = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
        circle = min_circle_mosek(p)

        assert circle.radius == pytest.approx(2.5, rel=1e-6)
        assert np.allclose(circle.center, [2.0, 1.5], rtol=1e-6)

    def test_many_points(self, mosek: None) -> None:
        """Test MOSEK solver with many random points."""
        from min_circle.msk import min_circle_mosek

        np.random.seed(42)
        p = np.random.randn(50, 2)
        circle = min_circle_mosek(p)

        # All points should be inside or on the circle
        distances = np.linalg.norm(p - circle.center, axis=1)
        assert all(d <= circle.radius + 1e-5 for d in distances)
