"""
Shared fixtures for insurance-conformal tests.

All synthetic data is generated from true Tweedie/Poisson distributions
with known parameters so we can verify coverage guarantees analytically.
"""

from __future__ import annotations

import numpy as np
import pytest


class LinearPoissonModel:
    """
    Minimal sklearn-compatible Poisson model for testing.

    True model: E[Y] = exp(X @ beta), Y ~ Poisson(E[Y]).
    This lets us test coverage with a model that has no misspecification.
    """

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.exp(X @ self.beta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearPoissonModel":
        # Not actually fitting — beta is fixed for testing.
        return self


class LinearTweedieModel:
    """Minimal model returning yhat = exp(X @ beta) for Tweedie(p)."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.exp(X @ self.beta)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def poisson_data(rng):
    """
    Synthetic Poisson dataset with known true model.

    n=2000 observations, 3 features, true log-linear model.
    Split into train (1200), calibration (400), test (400).
    """
    n = 2000
    p = 3
    beta = np.array([0.5, -0.3, 0.2])

    X = rng.normal(size=(n, p))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu)

    train_end = 1200
    cal_end = 1600

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_cal": X[train_end:cal_end],
        "y_cal": y[train_end:cal_end],
        "X_test": X[cal_end:],
        "y_test": y[cal_end:],
        "mu_test": mu[cal_end:],
        "beta": beta,
        "model": LinearPoissonModel(beta),
    }


@pytest.fixture(scope="session")
def small_cal_data(rng):
    """Very small calibration set for edge case testing."""
    beta = np.array([0.3, -0.2])
    X = rng.normal(size=(50, 2))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu)
    return {
        "X": X[:40],
        "y": y[:40],
        "X_cal": X[40:],
        "y_cal": y[40:],
        "model": LinearPoissonModel(beta),
    }
