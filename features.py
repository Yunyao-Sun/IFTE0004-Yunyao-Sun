import numpy as np
import pandas as pd


def make_features(df, horizon=1, eps=1e-12):
    df = df.copy().sort_values(["asset", "date"])
    panels = []
    for asset, g in df.groupby("asset"):
        g = g.copy().sort_values("date")
        g["y"] = g["RV"].shift(-horizon)
        g["log_y"] = np.log(g["y"] + eps)
        g["RVD"] = g["RV"]
        g["RVW"] = g["RV"].rolling(5).mean()
        g["RVM"] = g["RV"].rolling(22).mean()
        g["M1W"] = 100.0 * g["ret"].rolling(5).sum()
        g["$VOL"] = 100.0 * np.log(g["dollar_volume"] + eps).diff()
        g["log_RVD"] = np.log(g["RVD"] + eps)
        g["log_RVW"] = np.log(g["RVW"] + eps)
        g["log_RVM"] = np.log(g["RVM"] + eps)
        g["rD"] = g["ret"]
        g["rW"] = g["ret"].rolling(5).mean()
        g["rM"] = g["ret"].rolling(22).mean()
        g["rD_neg"] = np.minimum(g["rD"], 0.0)
        g["rW_neg"] = np.minimum(g["rW"], 0.0)
        g["rM_neg"] = np.minimum(g["rM"], 0.0)
        g["RVposD"] = g["RV_pos"]
        g["RVnegD"] = g["RV_neg"]
        g["HARQ_interact"] = g["RVD"] * g["RQ"]
        panels.append(g)
    panel = pd.concat(panels, axis=0)
    return panel.replace([np.inf, -np.inf], np.nan)


def get_feature_sets(panel):
    feature_sets = {}
    feature_sets["HAR"] = ["RVD", "RVW", "RVM"]
    mall = ["RVD", "RVW", "RVM", "M1W", "$VOL", "IV", "EA", "VIX", "EPU", "US3M", "HSI", "ADS"]
    feature_sets["MALL"] = [c for c in mall if c in panel.columns]
    feature_sets["HAR-X"] = feature_sets["MALL"]
    feature_sets["LogHAR"] = ["log_RVD", "log_RVW", "log_RVM"]
    feature_sets["LevHAR"] = ["RVD", "RVW", "RVM", "rD_neg", "rW_neg", "rM_neg"]
    feature_sets["SHAR"] = ["RVnegD", "RVposD", "RVW", "RVM"]
    feature_sets["HARQ"] = ["RVD", "HARQ_interact", "RVW", "RVM"]
    clean = {}
    for name, cols in feature_sets.items():
        if all(c in panel.columns for c in cols) and len(cols) > 0:
            clean[name] = cols
    return clean


def make_table1_summary(panel):
    vars_order = ["RVD", "RVW", "RVM", "IV", "EA", "VIX", "EPU", "US3M", "HSI", "M1W", "$VOL", "ADS"]
    rows = []
    for i, var in enumerate(vars_order, 1):
        if var not in panel.columns:
            continue
        x = panel[var].dropna()
        if len(x) == 0:
            continue
        rows.append({
            "No.": i,
            "Acronym": var,
            "Mean": x.mean(),
            "Median": x.median(),
            "Maximum": x.max(),
            "Minimum": x.min(),
            "Standard deviation": x.std(),
            "Skewness": x.skew(),
            "Kurtosis": x.kurtosis(),
        })
    return pd.DataFrame(rows)
