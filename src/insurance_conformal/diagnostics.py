"""
Coverage diagnostics for conformal prediction intervals.

The marginal coverage guarantee from split conformal prediction is a floor,
not a promise of uniformity. In insurance, the pathological failure mode is
correct average coverage hiding poor tail coverage: a model may achieve 90%
overall but only 65% for the highest-risk decile, which is exactly where
uncertainty quantification matters most.

These diagnostics expose that problem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from insurance_conformal.utils import as_numpy

if TYPE_CHECKING:
    from insurance_conformal.predictor import InsuranceConformalPredictor


class CoverageDiagnostics:
    """
    Standalone coverage diagnostics that work with any set of prediction intervals.

    Can be used independently of InsuranceConformalPredictor — useful if you have
    intervals from another source and want to apply the same diagnostic framework.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed values. Accepts numpy, pandas Series, or polars Series.
    y_lower : array-like of shape (n,)
        Lower bounds of prediction intervals.
    y_upper : array-like of shape (n,)
        Upper bounds of prediction intervals.
    y_pred : array-like of shape (n,), optional
        Point predictions. Used for decile-binning and width analysis.
        If None, uses (y_lower + y_upper) / 2 as a proxy.
    alpha : float, optional
        Nominal miscoverage rate. Used only for labelling in plots and summaries.

    Examples
    --------
    >>> diag = CoverageDiagnostics(y_true, lower, upper, y_pred=yhat, alpha=0.10)
    >>> diag.coverage_by_decile()
    >>> diag.coverage_plot()
    >>> diag.interval_width_distribution()
    """

    def __init__(
        self,
        y_true: Any,
        y_lower: Any,
        y_upper: Any,
        y_pred: Optional[Any] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self.y_true = as_numpy(y_true)
        self.y_lower = as_numpy(y_lower)
        self.y_upper = as_numpy(y_upper)
        self.y_pred = (
            as_numpy(y_pred)
            if y_pred is not None
            else (self.y_lower + self.y_upper) / 2.0
        )
        self.alpha = alpha

        self._covered = (self.y_true >= self.y_lower) & (self.y_true <= self.y_upper)
        self._widths = self.y_upper - self.y_lower

    @property
    def marginal_coverage(self) -> float:
        """Empirical marginal coverage rate."""
        return float(self._covered.mean())

    @property
    def mean_width(self) -> float:
        """Mean interval width."""
        return float(self._widths.mean())

    @property
    def median_width(self) -> float:
        """Median interval width."""
        return float(np.median(self._widths))

    def coverage_by_decile(self, n_bins: int = 10) -> pl.DataFrame:
        """
        Compute empirical coverage by decile of predicted value.

        Parameters
        ----------
        n_bins : int, default 10
            Number of equal-frequency bins.

        Returns
        -------
        pl.DataFrame
            Columns: decile, mean_predicted, mean_width, n_obs,
            coverage, target_coverage.
        """
        try:
            decile_labels = pd.qcut(self.y_pred, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(self.y_pred, bins=n_bins, labels=False, duplicates="drop")

        results = []
        unique_deciles = np.unique(decile_labels[~np.isnan(decile_labels.astype(float))])
        for d in unique_deciles:
            mask = decile_labels == d
            target_cov = (1 - self.alpha) if self.alpha is not None else float("nan")
            results.append(
                {
                    "decile": int(d) + 1,
                    "mean_predicted": float(self.y_pred[mask].mean()),
                    "mean_width": float(self._widths[mask].mean()),
                    "n_obs": int(mask.sum()),
                    "coverage": float(self._covered[mask].mean()),
                    "target_coverage": target_cov,
                }
            )

        return pl.DataFrame(results)

    def coverage_plot(
        self,
        n_bins: int = 10,
        figsize: tuple = (10, 6),
        title: Optional[str] = None,
    ) -> Any:
        """
        Matplotlib visualisation of coverage by decile with confidence bands.

        The confidence bands are Wilson score intervals for the binomial
        proportion — they represent uncertainty in the coverage estimate
        due to finite sample size, not uncertainty in the model.

        Parameters
        ----------
        n_bins : int, default 10
            Number of deciles.
        figsize : tuple, default (10, 6)
            Figure size.
        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import binom
        except ImportError as e:
            raise ImportError(
                "matplotlib and scipy are required for coverage_plot(). "
                "Install with: pip install insurance-conformal[plot]"
            ) from e

        df = self.coverage_by_decile(n_bins=n_bins)
        target = (1 - self.alpha) if self.alpha is not None else None

        fig, ax = plt.subplots(figsize=figsize)

        # Compute Wilson score confidence bands
        ns = df["n_obs"].to_numpy()
        ps = df["coverage"].to_numpy()
        z = 1.96  # 95% CI

        denom = 1 + z**2 / ns
        centre = (ps + z**2 / (2 * ns)) / denom
        half = z * np.sqrt(ps * (1 - ps) / ns + z**2 / (4 * ns**2)) / denom

        lower_ci = np.clip(centre - half, 0, 1)
        upper_ci = np.clip(centre + half, 0, 1)

        x = df["decile"].to_numpy()

        ax.fill_between(x, lower_ci, upper_ci, alpha=0.2, color="steelblue", label="95% CI (Wilson)")
        ax.plot(x, ps, "o-", color="steelblue", linewidth=2, markersize=6, label="Empirical coverage")

        if target is not None:
            ax.axhline(target, color="red", linestyle="--", linewidth=1.5, label=f"Target ({target:.0%})")

        ax.set_xlabel("Decile of predicted value (1=lowest risk, 10=highest risk)", fontsize=11)
        ax.set_ylabel("Coverage rate", fontsize=11)
        ax.set_ylim(max(0, (target or 0.5) - 0.3), 1.02)
        ax.set_xticks(x)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        t = title or (
            f"Coverage by risk decile"
            + (f" (target: {target:.0%})" if target is not None else "")
        )
        ax.set_title(t, fontsize=13)

        fig.tight_layout()
        return fig

    def interval_width_distribution(
        self,
        n_bins: int = 40,
        figsize: tuple = (10, 5),
        log_scale: bool = False,
        title: Optional[str] = None,
    ) -> Any:
        """
        Histogram of prediction interval widths.

        Useful for spotting heteroscedasticity — a good score should produce
        intervals that widen proportionally with the point prediction. If
        the width distribution is bimodal or has a very heavy tail, the score
        is not well-matched to the data's variance structure.

        Parameters
        ----------
        n_bins : int, default 40
            Number of histogram bins.
        figsize : tuple, default (10, 5)
            Figure size.
        log_scale : bool, default False
            Use log scale on x-axis. Useful when widths span several orders
            of magnitude (common in pure premium models).
        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for interval_width_distribution(). "
                "Install with: pip install insurance-conformal[plot]"
            ) from e

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left panel: histogram of widths
        ax = axes[0]
        widths = self._widths
        if log_scale:
            pos_widths = widths[widths > 0]
            ax.hist(np.log10(pos_widths + 1e-10), bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.set_xlabel("log10(interval width)", fontsize=11)
        else:
            ax.hist(widths, bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Interval width", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Distribution of interval widths", fontsize=12)
        ax.axvline(np.median(widths), color="red", linestyle="--", label=f"Median: {np.median(widths):.3f}")
        ax.axvline(widths.mean(), color="orange", linestyle="--", label=f"Mean: {widths.mean():.3f}")
        ax.legend(fontsize=9)

        # Right panel: width vs predicted value (scatter)
        ax2 = axes[1]
        sample_n = min(5000, len(self.y_pred))
        idx = np.random.choice(len(self.y_pred), size=sample_n, replace=False)
        ax2.scatter(
            self.y_pred[idx],
            widths[idx],
            alpha=0.3,
            s=8,
            color="steelblue",
        )
        ax2.set_xlabel("Point prediction", fontsize=11)
        ax2.set_ylabel("Interval width", fontsize=11)
        ax2.set_title("Width vs. predicted value", fontsize=12)
        ax2.grid(alpha=0.2)

        if title:
            fig.suptitle(title, fontsize=13)

        fig.tight_layout()
        return fig
