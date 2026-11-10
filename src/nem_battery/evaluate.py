"""
Price forecasting evaluation metrics
"""

import pandas as pd
import numpy as np

def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: average size of the error in $/MWh."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: punishes large errors heavily."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def score_table(df: pd.DataFrame, target: str, pred_cols: list[str]) -> pd.DataFrame:
    """MAE and RMSE for several prediction columns, sorted by MAE."""
    rows = [{"model": c, "MAE": mae(df[target], df[c]), "RMSE": rmse(df[target], df[c])}
            for c in pred_cols]
    return pd.DataFrame(rows).sort_values("MAE").round(2).reset_index(drop=True)