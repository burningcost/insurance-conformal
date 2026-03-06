"""
Tests for non-conformity score functions in insurance_conformal.scores.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_conformal.scores import (
    NonconformityScore,
    anscombe_score,
    compute_score,
    deviance_score,
    invert_score,
    pearson_score,
    pearson_weighted_score,
    raw_score,
)


class TestRawScore:
    def test_basic(self):
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 1.5, 3.5])
        scores = raw_score(y, yhat)
        np.testing.assert_allclose(scores, [0.5, 0.5, 0.5])

    def test_always_nonnegative(self, rng=np.random.default_rng(0)):
        y = rng.poisson(2.0, size=200)
        yhat = rng.uniform(0.1, 5.0, size=200)
        assert np.all(raw_score(y, yhat) >= 0)

    def test_zero_residual(self):
        y = np.array([2.0, 3.0])
        yhat = np.array([2.0, 3.0])
        np.testing.assert_allclose(raw_score(y, yhat), [0.0, 0.0])


class TestPearsonScore:
    def test_basic(self):
        y = np.array([4.0])
        yhat = np.array([1.0])
        # |4 - 1| / sqrt(1) = 3.0
        np.testing.assert_allclose(pearson_score(y, yhat), [3.0])

    def test_normalised_by_sqrt_yhat(self):
        y = np.array([5.0])
        yhat = np.array([4.0])
        expected = abs(5.0 - 4.0) / np.sqrt(4.0)
        np.testing.assert_allclose(pearson_score(y, yhat), [expected])

    def test_always_nonnegative(self, rng=np.random.default_rng(1)):
        y = rng.poisson(3.0, size=300)
        yhat = rng.uniform(0.5, 6.0, size=300)
        assert np.all(pearson_score(y, yhat) >= 0)

    def test_warns_on_zero_prediction(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([0.0, 1.0])  # zero prediction
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pearson_score(y, yhat)
            assert len(w) == 1
            assert "clipped" in str(w[0].message).lower()

    def test_smaller_than_raw_for_large_yhat(self):
        """Pearson score should be smaller than raw score when yhat > 1."""
        y = np.array([10.0, 20.0])
        yhat = np.array([9.0, 19.0])
        assert np.all(pearson_score(y, yhat) < raw_score(y, yhat))


class TestPearsonWeightedScore:
    def test_reduces_to_pearson_at_p1(self):
        y = np.array([3.0, 5.0, 1.0])
        yhat = np.array([2.0, 4.0, 1.5])
        p1 = pearson_weighted_score(y, yhat, tweedie_power=1.0)
        p_std = pearson_score(y, yhat)
        np.testing.assert_allclose(p1, p_std)

    def test_p2_gives_cv_like_residual(self):
        y = np.array([4.0])
        yhat = np.array([2.0])
        # |4 - 2| / 2^(2/2) = 2 / 2 = 1.0
        score = pearson_weighted_score(y, yhat, tweedie_power=2.0)
        np.testing.assert_allclose(score, [1.0])

    def test_narrower_than_raw(self):
        """Weighted score is narrower than raw when yhat > 1 and p > 0."""
        rng = np.random.default_rng(2)
        yhat = rng.uniform(2.0, 10.0, size=100)
        y = yhat + rng.normal(0, 0.5, size=100)
        assert np.mean(pearson_weighted_score(y, yhat, 1.5)) < np.mean(raw_score(y, yhat))

    def test_shape_preserved(self):
        y = np.ones(50)
        yhat = np.ones(50) * 2
        assert pearson_weighted_score(y, yhat).shape == (50,)


class TestDevianceScore:
    def test_poisson_zero_at_yhat(self):
        """Deviance score at y=yhat should be 0."""
        yhat = np.array([1.0, 2.0, 5.0])
        scores = deviance_score(yhat, yhat, distribution="poisson")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_gamma_zero_at_yhat(self):
        yhat = np.array([1.0, 3.0])
        scores = deviance_score(yhat, yhat, distribution="gamma")
        np.testing.assert_allclose(scores, np.zeros(2), atol=1e-10)

    def test_tweedie_zero_at_yhat(self):
        yhat = np.array([2.0, 4.0])
        scores = deviance_score(yhat, yhat, distribution="tweedie", tweedie_power=1.5)
        np.testing.assert_allclose(scores, np.zeros(2), atol=1e-10)

    def test_always_nonnegative(self):
        rng = np.random.default_rng(3)
        y = rng.exponential(scale=2.0, size=200)
        yhat = rng.uniform(0.5, 5.0, size=200)
        assert np.all(deviance_score(y, yhat, "tweedie", 1.5) >= 0)

    def test_poisson_zero_observations(self):
        """y=0 is valid for Poisson; score should be finite."""
        y = np.array([0.0, 0.0])
        yhat = np.array([1.0, 2.0])
        scores = deviance_score(y, yhat, distribution="poisson")
        assert np.all(np.isfinite(scores))
        assert np.all(scores > 0)

    def test_tweedie_p1_equals_poisson(self):
        y = np.array([2.0, 3.0, 5.0])
        yhat = np.array([1.5, 2.5, 4.5])
        d_tweedie = deviance_score(y, yhat, "tweedie", tweedie_power=1.0)
        d_poisson = deviance_score(y, yhat, "poisson")
        np.testing.assert_allclose(d_tweedie, d_poisson, rtol=1e-6)

    def test_tweedie_p2_equals_gamma(self):
        y = np.array([2.0, 3.0, 5.0])
        yhat = np.array([1.5, 2.5, 4.5])
        d_tweedie = deviance_score(y, yhat, "tweedie", tweedie_power=2.0)
        d_gamma = deviance_score(y, yhat, "gamma")
        np.testing.assert_allclose(d_tweedie, d_gamma, rtol=1e-6)

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            deviance_score(np.array([1.0]), np.array([1.0]), distribution="normal")


class TestAnscombeScore:
    def test_zero_at_yhat_poisson(self):
        yhat = np.array([1.0, 4.0, 9.0])
        scores = anscombe_score(yhat, yhat, distribution="poisson")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_always_nonnegative(self):
        rng = np.random.default_rng(4)
        y = rng.exponential(scale=3.0, size=200)
        yhat = rng.uniform(0.5, 6.0, size=200)
        assert np.all(anscombe_score(y, yhat, "poisson") >= 0)

    def test_tweedie_p1_equals_poisson(self):
        y = np.array([2.0, 5.0])
        yhat = np.array([3.0, 4.0])
        a1 = anscombe_score(y, yhat, "tweedie", tweedie_power=1.0)
        ap = anscombe_score(y, yhat, "poisson")
        np.testing.assert_allclose(a1, ap, rtol=1e-6)


class TestComputeScore:
    def test_dispatches_correctly(self):
        y = np.array([2.0, 3.0])
        yhat = np.array([1.5, 2.5])

        for name in ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"]:
            scores = compute_score(y, yhat, name, "tweedie", 1.5)
            assert scores.shape == (2,)
            assert np.all(scores >= 0)

    def test_invalid_name(self):
        with pytest.raises(ValueError):
            compute_score(np.array([1.0]), np.array([1.0]), "banana")


class TestInvertScore:
    """Test that score inversion recovers y values with the correct score."""

    def _check_inversion(self, nonconformity: str, yhat_vals, quantile, p=1.5):
        lower, upper = invert_score(
            yhat_vals, quantile, nonconformity, "tweedie", p, clip_lower=0.0
        )
        assert np.all(lower >= 0), "Lower bound must be >= 0"
        assert np.all(upper >= yhat_vals), "Upper >= yhat"

        # Score at upper bound should equal quantile (within tolerance)
        upper_scores = compute_score(upper, yhat_vals, nonconformity, "tweedie", p)
        np.testing.assert_allclose(upper_scores, quantile, rtol=0.05, atol=1e-3)

    def test_raw_inversion(self):
        yhat = np.array([1.0, 2.0, 5.0])
        lower, upper = invert_score(yhat, 1.0, "raw", clip_lower=0.0)
        np.testing.assert_allclose(lower, [0.0, 1.0, 4.0])
        np.testing.assert_allclose(upper, [2.0, 3.0, 6.0])

    def test_pearson_inversion(self):
        yhat = np.array([4.0])
        quantile = 1.0
        lower, upper = invert_score(yhat, quantile, "pearson", clip_lower=0.0)
        # half_width = 1.0 * sqrt(4) = 2.0
        np.testing.assert_allclose(upper, [6.0])
        np.testing.assert_allclose(lower, [2.0])

    def test_pearson_weighted_inversion(self):
        yhat = np.array([4.0])
        quantile = 1.0
        lower, upper = invert_score(yhat, quantile, "pearson_weighted", tweedie_power=2.0, clip_lower=0.0)
        # half_width = 1.0 * 4^(2/2) = 4.0
        np.testing.assert_allclose(upper, [8.0])
        np.testing.assert_allclose(lower, [0.0])  # clipped at 0

    def test_lower_clipped_at_zero(self):
        yhat = np.array([0.1])
        quantile = 10.0  # very wide interval
        lower, upper = invert_score(yhat, quantile, "raw", clip_lower=0.0)
        assert lower[0] == 0.0

    def test_deviance_inversion_roundtrip(self):
        yhat = np.array([2.0, 5.0])
        quantile = 0.5
        self._check_inversion("deviance", yhat, quantile, p=1.5)

    def test_anscombe_inversion_roundtrip(self):
        yhat = np.array([2.0, 5.0])
        quantile = 0.5
        self._check_inversion("anscombe", yhat, quantile, p=1.5)
