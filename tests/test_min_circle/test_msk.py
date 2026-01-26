"""Tests for the MOSEK implementation.

This module contains tests for the MOSEK solver implementation.
"""

import numpy as np
import pytest

from min_circle.msk import _MOSEK_IMPORT_ERROR, min_circle_mosek


class TestMosekImport:
    """Tests for MOSEK import handling."""

    def test_import_error_message(self) -> None:
        """Test that the import error message is informative."""
        assert "MOSEK" in _MOSEK_IMPORT_ERROR
        assert "pip install" in _MOSEK_IMPORT_ERROR


def _has_mosek_license() -> bool:
    """Check if MOSEK license is available."""
    try:
        import mosek.fusion as mf

        with mf.Model() as model:
            r = model.variable("test", 1)
            model.objective("obj", mf.ObjectiveSense.Minimize, r)
            model.solve()
    except Exception:
        return False
    else:
        return True


# Skip MOSEK solver tests if license is not available
skip_without_license = pytest.mark.skipif(not _has_mosek_license(), reason="MOSEK license not available")


class TestMosekSolver:
    """Tests for the MOSEK solver when license is available."""

    @skip_without_license
    def test_two_points(self) -> None:
        """Test MOSEK solver with two points."""
        p = np.array([[0.0, 0.0], [4.0, 0.0]])
        circle = min_circle_mosek(p)

        assert circle.radius == pytest.approx(2.0, rel=1e-6)
        assert np.allclose(circle.center, [2.0, 0.0], rtol=1e-6)

    @skip_without_license
    def test_three_points(self) -> None:
        """Test MOSEK solver with three points."""
        p = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
        circle = min_circle_mosek(p)

        assert circle.radius == pytest.approx(2.5, rel=1e-6)
        assert np.allclose(circle.center, [2.0, 1.5], rtol=1e-6)

    @skip_without_license
    def test_many_points(self) -> None:
        """Test MOSEK solver with many random points."""
        np.random.seed(42)
        p = np.random.randn(50, 2)
        circle = min_circle_mosek(p)

        # All points should be inside or on the circle
        distances = np.linalg.norm(p - circle.center, axis=1)
        assert all(d <= circle.radius + 1e-5 for d in distances)


class TestMosekSolverWithoutLicense:
    """Tests that run even without a MOSEK license (to verify error handling)."""

    @pytest.mark.skipif(_has_mosek_license(), reason="Test only runs without MOSEK license")
    def test_license_error(self) -> None:
        """Test that we get an appropriate error when license is not available."""
        import mosek.fusion as mf

        p = np.array([[0.0, 0.0], [4.0, 0.0]])

        with pytest.raises(mf.OptimizeError, match="license"):
            min_circle_mosek(p)
