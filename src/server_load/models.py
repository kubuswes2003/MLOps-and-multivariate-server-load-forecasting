"""Forecasting models: baseline (Q5), linear (Q5), nonlinear (Q7), iterative (Q8).

Every learned transformation (scaler, imputer) must live inside a sklearn Pipeline
so that it is fitted on the training data only — no leakage.
"""

import pandas as pd
from sklearn.pipeline import Pipeline


def persistence_forecast(current_target: pd.Series) -> pd.Series:
    """Naive baseline: the next value equals the current one (y_hat[t+1] = y[t]).

    Args:
        current_target: Target values at time t.

    Returns:
        Predictions for t + horizon.
    """
    # TODO(Q5)
    raise NotImplementedError("Q5: persistence baseline")


def build_ridge(alpha: float) -> Pipeline:
    """Imputer + StandardScaler + Ridge(alpha), as one Pipeline.

    Args:
        alpha: L2 penalty strength, chosen on the validation set.

    Returns:
        Unfitted Pipeline.
    """
    # TODO(Q5): also derive the Ridge objective in the report.
    raise NotImplementedError("Q5: Ridge pipeline")


def build_gradient_boosting(random_seed: int, **params: object) -> Pipeline:
    """Nonlinear model, e.g. HistGradientBoostingRegressor (handles NaN natively).

    Args:
        random_seed: Seed for reproducibility.
        **params: Hyperparameters tuned on validation (learning_rate, max_depth, ...).

    Returns:
        Unfitted Pipeline.
    """
    # TODO(Q7/Q8): HistGradientBoosting exposes per-iteration scores for learning curves.
    raise NotImplementedError("Q7: nonlinear model")
