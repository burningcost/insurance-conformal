"""
Utility functions for insurance-conformal.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def extract_tweedie_power(model: Any) -> Optional[float]:
    """
    Attempt to extract the Tweedie power parameter from a fitted model.

    Supports LightGBM and sklearn TweedieRegressor. Returns None if
    the parameter cannot be determined automatically.

    Parameters
    ----------
    model : fitted model
        A fitted sklearn-compatible model.

    Returns
    -------
    float or None
        The Tweedie power parameter, or None if not detectable.
    """
    # LightGBM Booster or sklearn wrapper
    try:
        import lightgbm as lgb

        if isinstance(model, lgb.Booster):
            params = model.params
            if params.get("objective") in ("tweedie", "mape"):
                return float(params.get("tweedie_variance_power", 1.5))
            elif params.get("objective") == "poisson":
                return 1.0
            elif params.get("objective") == "gamma":
                return 2.0

        if isinstance(model, lgb.LGBMRegressor):
            objective = model.objective_
            if objective == "tweedie":
                return float(model.tweedie_variance_power)
            elif objective == "poisson":
                return 1.0
    except ImportError:
        pass

    # sklearn TweedieRegressor
    try:
        from sklearn.linear_model import TweedieRegressor

        if isinstance(model, TweedieRegressor):
            return float(model.power)
    except ImportError:
        pass

    # Wrapped model (e.g. Pipeline) — try the last step
    try:
        if hasattr(model, "steps"):
            return extract_tweedie_power(model.steps[-1][1])
    except Exception:
        pass

    return None


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the split conformal quantile from calibration scores.

    For n calibration samples, the conformal quantile is the
    ceil((n+1)(1-alpha))/n quantile of the calibration scores. This
    guarantees marginal coverage >= 1-alpha for any exchangeable data.

    This is the finite-sample correction from Venn prediction / split conformal
    prediction — the (n+1) rather than n in the numerator is essential.

    Parameters
    ----------
    scores : np.ndarray
        Non-conformity scores from the calibration set.
    alpha : float
        Miscoverage rate. Must be in (0, 1).

    Returns
    -------
    float
        The calibration quantile.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute quantile from empty calibration set.")

    # The ceil((n+1)(1-alpha))/n quantile
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def apply_exposure(
    y: np.ndarray,
    yhat: np.ndarray,
    exposure: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply exposure offset to observed and predicted values.

    For frequency models with an exposure offset, y is a count and yhat
    is the expected count (already multiplied by exposure in the model's
    predict call). If the user passes raw rates, we need to convert.

    This function assumes yhat already incorporates exposure (which is
    standard for LightGBM offset models). The exposure parameter here
    is used only for validation and documentation.

    Parameters
    ----------
    y : np.ndarray
        Observed values (counts or rates, depending on model).
    yhat : np.ndarray
        Predicted values from model.predict().
    exposure : np.ndarray or None
        Exposure weights. If None, no adjustment is made.

    Returns
    -------
    y, yhat : both as np.ndarray
        Potentially adjusted values.
    """
    if exposure is None:
        return y, yhat

    exposure = np.asarray(exposure, dtype=float)
    if np.any(exposure <= 0):
        raise ValueError("All exposure values must be positive.")

    return y, yhat


def as_numpy(x: Any) -> np.ndarray:
    """Convert pandas Series/DataFrame or list to numpy array."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


def temporal_split(
    X: Any,
    y: Any,
    calibration_frac: float,
    date_col: Optional[Union[str, np.ndarray]] = None,
    exposure: Optional[Any] = None,
) -> tuple:
    """
    Split data into training and calibration sets.

    If date_col is provided, the calibration set is the most recent
    calibration_frac of observations (by date). This matches the insurance
    use-case where you calibrate on recent data to capture recent loss trends.

    If date_col is None, uses a simple tail split (last calibration_frac rows).

    Parameters
    ----------
    X : array-like
        Features.
    y : array-like
        Target.
    calibration_frac : float
        Fraction of data to use as calibration set.
    date_col : str or array-like, optional
        Date column name (if X is a DataFrame) or array of dates.
        If provided, calibration uses the most recent observations.
    exposure : array-like, optional
        Exposure weights.

    Returns
    -------
    X_train, X_cal, y_train, y_cal, exp_train, exp_cal
    """
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y = as_numpy(y)
    n = len(y)
    cal_n = max(1, int(np.ceil(n * calibration_frac)))

    if date_col is not None:
        if isinstance(date_col, str):
            dates = X[date_col].to_numpy()
        else:
            dates = as_numpy(date_col)

        order = np.argsort(dates)
        train_idx = order[:-cal_n]
        cal_idx = order[-cal_n:]
    else:
        # Simple tail split — assumes data is in temporal order
        train_idx = np.arange(n - cal_n)
        cal_idx = np.arange(n - cal_n, n)

    X_train = X.iloc[train_idx]
    X_cal = X.iloc[cal_idx]
    y_train = y[train_idx]
    y_cal = y[cal_idx]

    if exposure is not None:
        exp = as_numpy(exposure)
        exp_train = exp[train_idx]
        exp_cal = exp[cal_idx]
    else:
        exp_train = exp_cal = None

    return X_train, X_cal, y_train, y_cal, exp_train, exp_cal
