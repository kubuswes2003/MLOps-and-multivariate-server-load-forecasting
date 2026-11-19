"""Regression metrics used to compare all models (Q9)."""

import pandas as pd


def regression_report(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, R² and at least one normalised error.

    Rows where either value is NaN must be dropped before scoring.

    Args:
        y_true: Observed values.
        y_pred: Predicted values, same index as y_true.

    Returns:
        Dict like {"MAE": ..., "RMSE": ..., "R2": ..., "NRMSE": ...}.
    """
    # TODO(Q9): sklearn.metrics has MAE, MSE and R²; choose and justify a normalisation.
    raise NotImplementedError("Q9: metrics")
