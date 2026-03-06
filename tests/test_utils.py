"""
Tests for utility functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_conformal.utils import (
    as_numpy,
    conformal_quantile,
    extract_tweedie_power,
    temporal_split,
)


class TestConformalQuantile:
    def test_basic(self):
        # 10 scores, alpha=0.1: ceil(11 * 0.9) / 10 = ceil(9.9)/10 = 10/10 = 1.0
        # so we want the 100th percentile, i.e., max
        scores = np.arange(1, 11, dtype=float)  # [1, 2, ..., 10]
        q = conformal_quantile(scores, alpha=0.1)
        assert q == 10.0

    def test_alpha_05(self):
        scores = np.arange(1, 11, dtype=float)
        q = conformal_quantile(scores, alpha=0.5)
        # ceil(11 * 0.5)/10 = ceil(5.5)/10 = 6/10 = 0.6 quantile
        expected = np.quantile(scores, 0.6, method="higher")
        assert q == expected

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            conformal_quantile(np.array([]), alpha=0.1)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.array([1.0, 2.0]), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.array([1.0, 2.0]), alpha=1.0)

    def test_single_observation(self):
        q = conformal_quantile(np.array([5.0]), alpha=0.1)
        assert q == 5.0

    def test_quantile_nondecreasing_in_alpha(self):
        """Smaller alpha (wider interval) should give larger quantile."""
        scores = np.random.default_rng(0).uniform(0, 10, size=200)
        q_small = conformal_quantile(scores, alpha=0.01)
        q_large = conformal_quantile(scores, alpha=0.50)
        assert q_small >= q_large


class TestExtractTweediepower:
    def test_returns_none_for_unknown(self):
        class UnknownModel:
            pass

        result = extract_tweedie_power(UnknownModel())
        assert result is None

    def test_sklearn_tweedie_regressor(self):
        try:
            from sklearn.linear_model import TweedieRegressor

            model = TweedieRegressor(power=1.5)
            assert extract_tweedie_power(model) == 1.5

            model2 = TweedieRegressor(power=2.0)
            assert extract_tweedie_power(model2) == 2.0
        except ImportError:
            pytest.skip("sklearn not installed")


class TestTemporalSplit:
    def test_basic_split(self):
        n = 100
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.arange(n, dtype=float)

        X_train, X_cal, y_train, y_cal, e_train, e_cal = temporal_split(
            X, y, calibration_frac=0.2
        )

        assert len(X_cal) == 20
        assert len(X_train) == 80
        assert len(y_cal) == 20
        assert len(y_train) == 80
        assert e_train is None
        assert e_cal is None

    def test_with_exposure(self):
        n = 100
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.ones(n)
        exposure = np.ones(n) * 0.5

        X_train, X_cal, y_train, y_cal, e_train, e_cal = temporal_split(
            X, y, calibration_frac=0.2, exposure=exposure
        )
        assert e_cal is not None
        assert len(e_cal) == 20
        assert np.all(e_cal == 0.5)

    def test_date_col_split(self):
        n = 100
        dates = np.arange(n)
        X = pd.DataFrame({"date": dates, "feature": np.random.randn(n)})
        y = np.ones(n)

        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
            X, y, calibration_frac=0.2, date_col="date"
        )

        # Calibration set should contain the most recent 20 dates
        cal_dates = X_cal["date"].to_numpy()
        assert cal_dates.min() >= 80  # dates 80-99

    def test_total_observations_preserved(self):
        n = 200
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.arange(n, dtype=float)

        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
            X, y, calibration_frac=0.25
        )
        assert len(X_train) + len(X_cal) == n


class TestAsNumpy:
    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = as_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_pandas_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = as_numpy(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_list(self):
        result = as_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
