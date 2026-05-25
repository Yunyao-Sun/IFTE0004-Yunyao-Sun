import numpy as np
import pandas as pd
from scipy.stats import norm


def newey_west_variance(x, lag=0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    t = len(x)
    if t <= 1:
        return np.nan
    x = x - x.mean()
    lrv = np.dot(x, x) / t
    for ell in range(1, lag + 1):
        gamma = np.dot(x[ell:], x[:-ell]) / t
        weight = 1.0 - ell / (lag + 1)
        lrv += 2.0 * weight * gamma
    return lrv


def dm_test_mse(y, pred_i, pred_j, horizon=1):
    """One-sided DM test. Small p-value means model j beats model i."""
    y = np.asarray(y)
    pred_i = np.asarray(pred_i)
    pred_j = np.asarray(pred_j)
    d = (y - pred_i) ** 2 - (y - pred_j) ** 2
    d = d[np.isfinite(d)]
    t = len(d)
    if t <= 5:
        return np.nan, np.nan
    lag = max(horizon - 1, 0)
    lrv = newey_west_variance(d, lag=lag)
    if not np.isfinite(lrv) or lrv <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(lrv / t)
    p_value = 1.0 - norm.cdf(stat)
    return stat, p_value


def build_pairwise_table(result, horizon=1):
    model_order = [
        "HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ",
        "RR", "LA", "EN", "A-LA", "P-LA",
        "BG", "RF", "GB",
        "NN11", "NN101", "NN12", "NN102", "NN13", "NN103", "NN14", "NN104",
    ]
    available = set(result["predictions"].keys())
    model_order = [m for m in model_order if m in available]
    raw = pd.DataFrame(index=model_order, columns=model_order, dtype=float)
    formatted = pd.DataFrame(index=model_order, columns=model_order, dtype=object)
    y = result["y_test"]
    for row_model in model_order:
        for col_model in model_order:
            if row_model == col_model:
                raw.loc[row_model, col_model] = 1.0
                formatted.loc[row_model, col_model] = "–"
                continue
            pred_i = result["predictions"][row_model]
            pred_j = result["predictions"][col_model]
            mse_i = np.mean((y - pred_i) ** 2)
            mse_j = np.mean((y - pred_j) ** 2)
            ratio = mse_j / mse_i
            raw.loc[row_model, col_model] = ratio
            _, p_value = dm_test_mse(y, pred_i, pred_j, horizon=horizon)
            text = f"{ratio:.3f}"
            if p_value < 0.01:
                text = f"[{text}]"
            elif p_value < 0.05:
                text = f"({text})"
            elif p_value < 0.10:
                text = f"{text}*"
            formatted.loc[row_model, col_model] = text
    return raw, formatted


def build_mse_summary(result):
    rows = []
    y = result["y_test"]
    for model_name, pred in result["predictions"].items():
        rows.append({"asset": result["asset"], "model": model_name, "mse": np.mean((y - pred) ** 2)})
    df = pd.DataFrame(rows)
    summary = df.sort_values("mse").reset_index(drop=True)
    return df, summary
