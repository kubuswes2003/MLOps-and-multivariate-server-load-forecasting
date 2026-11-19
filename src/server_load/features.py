"""Lagged features and forecasting target (Q4)."""

import pandas as pd


def make_lag_features(df: pd.DataFrame, columns: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    """Add lagged copies of ``columns`` (e.g. ``cpu_lag1``, ``mem_lag5``).

    Only past values may be used: a lag of k at row t is the value at t - k.

    Args:
        df: Regularly sampled DataFrame.
        columns: Variables to lag.
        lags: Lag values in steps, e.g. (1, 5, 15).

    Returns:
        DataFrame with the original columns plus the lagged ones.
    """
    out = df.copy()
    for col in columns:
        for k in lags:
            out[f"{col}_lag{k}"] = df[col].shift(k)
    return out
    # TODO(Q4): build the lag columns with df[col].shift(k).
    raise NotImplementedError("Q4: implement lag features")


def make_target(df: pd.DataFrame, target: str, horizon_steps: int) -> pd.Series:
    """Return the value to predict: ``target`` at time t + horizon_steps.

    Args:
        df: Regularly sampled DataFrame.
        target: Column to forecast, e.g. "cpu".
        horizon_steps: Forecast horizon in steps.

    Returns:
        Series aligned with df's index, named ``f"{target}_t+{horizon_steps}"``.
    """
    return df[target].shift(-horizon_steps).rename(f"{target}_t+{horizon_steps}")
    # TODO(Q1/Q4): shift the target the other way (into the future).
    raise NotImplementedError("Q1/Q4: implement forecasting target")
