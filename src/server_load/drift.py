"""Regime change and drift monitoring (Q11–Q13)."""

import pandas as pd


def inject_regime_change(
    df: pd.DataFrame, start: str, column: str, shift: float
) -> pd.DataFrame:
    """Return a copy of df with a synthetic regime change from ``start`` onward.

    Only needed if the real machine shows no clear regime change.

    Args:
        df: Regularly sampled DataFrame.
        start: Change-point as offset from trace start, e.g. "6D".
        column: Variable affected, e.g. "net_in" (workload surge).
        shift: Additive change, clipped to [0, 100].

    Returns:
        Modified copy; the original df is left untouched.
    """
    # TODO(Q11)
    raise NotImplementedError("Q11: regime change")


def rolling_error(y_true: pd.Series, y_pred: pd.Series, window: str) -> pd.Series:
    """Rolling mean absolute error — the drift indicator (Q12).

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        window: Rolling window, e.g. "1h".

    Returns:
        Rolling MAE series.
    """
    # TODO(Q12)
    raise NotImplementedError("Q12: drift indicator")


def alert_threshold(validation_errors: pd.Series, n_sigma: float) -> float:
    """Alert threshold derived from validation errors only (e.g. mean + n·std).

    Args:
        validation_errors: Rolling error measured on the validation period.
        n_sigma: Number of standard deviations above the mean.

    Returns:
        Threshold value; an alert fires when the rolling error exceeds it.
    """
    # TODO(Q12)
    raise NotImplementedError("Q12: alert threshold")
