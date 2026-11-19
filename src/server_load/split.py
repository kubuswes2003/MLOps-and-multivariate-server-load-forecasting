"""Chronological train / validation / test split (Q6). Never shuffle."""

import pandas as pd


def chronological_split(
    df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: train < train_end <= val < val_end <= test.

    Args:
        df: DataFrame with a TimedeltaIndex (time since trace start).
        train_end: End of training period, e.g. "5D".
        val_end: End of validation period, e.g. "6D".

    Returns:
        (train, val, test) DataFrames, in chronological order, non-overlapping.

    Raises:
        ValueError: If train_end >= val_end.
    """
    # TODO(Q6): convert boundaries with pd.Timedelta, validate, slice df.
    # Report the exact time ranges of each part in the report.
    raise NotImplementedError("Q6: implement chronological split")
