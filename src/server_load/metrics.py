"""Regression metrics used to compare all models (Q9)."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_report(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R² and at least one normalised error.

    Rows where either value is NaN must be dropped before scoring.

    Args:
        y_true: Observed values.
        y_pred: Predicted values, same index as y_true.

    Returns:
        Dict like {"MAE": ..., "RMSE": ..., "R2": ..., "NRMSE": ...}.
    """
    pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    mask = y_true.notna() & pred.notna()
    if not mask.any():
        raise ValueError("No rows where both y_true and y_pred are defined")
    yt, yp = y_true[mask], pred[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": rmse,
        "R2": float(r2_score(yt, yp)),
        "NRMSE": rmse / float(yt.std()),
    }
