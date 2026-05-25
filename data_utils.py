import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Load intraday stock data
# ============================================================

def load_intraday_txt(path, asset_name):
    cols = ["date", "time", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(path, header=None, names=cols)
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).copy()
    df["date"] = pd.to_datetime(df["datetime"].dt.date)
    df["asset"] = asset_name
    return df.sort_values(["asset", "datetime"]).reset_index(drop=True)


def load_all_assets(data_dir, asset_names):
    intraday_list = []
    for asset in asset_names:
        path = data_dir / f"{asset}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        print(f"Loading {asset}: {path}")
        intraday_list.append(load_intraday_txt(path, asset))
    return pd.concat(intraday_list, axis=0).reset_index(drop=True)


# ============================================================
# 2. Build equal-weight index
# ============================================================

def make_equal_weight_index_intraday(intraday, index_name="EW_INDEX"):
    df = intraday.copy().sort_values(["asset", "datetime"])
    df["log_close"] = np.log(df["close"])
    df["intraday_ret"] = df.groupby(["asset", "date"])["log_close"].diff()
    idx = (
        df.dropna(subset=["intraday_ret"])
        .groupby(["date", "time", "datetime"], as_index=False)
        .agg(intraday_ret=("intraday_ret", "mean"), volume=("volume", "sum"))
    )
    idx["cum_log_price"] = idx.groupby("date")["intraday_ret"].cumsum()
    idx["close"] = 100.0 * np.exp(idx["cum_log_price"])
    idx["open"] = idx["close"]
    idx["high"] = idx["close"]
    idx["low"] = idx["close"]
    idx["asset"] = index_name
    return idx[["date", "time", "open", "high", "low", "close", "volume", "datetime", "asset"]].copy()


# ============================================================
# 3. Construct daily realized measures
# ============================================================

def make_daily_from_intraday(intraday):
    df = intraday.copy().sort_values(["asset", "datetime"])
    df["log_close"] = np.log(df["close"])
    df["intraday_ret"] = df.groupby(["asset", "date"])["log_close"].diff()
    ret_df = df.dropna(subset=["intraday_ret"]).copy()
    ret_df["ret2"] = ret_df["intraday_ret"] ** 2
    ret_df["ret4"] = ret_df["intraday_ret"] ** 4
    ret_df["ret2_pos"] = np.where(ret_df["intraday_ret"] > 0, ret_df["ret2"], 0.0)
    ret_df["ret2_neg"] = np.where(ret_df["intraday_ret"] < 0, ret_df["ret2"], 0.0)
    daily = (
        ret_df.groupby(["asset", "date"])
        .agg(
            RV_raw=("ret2", "sum"),
            RV_pos_raw=("ret2_pos", "sum"),
            RV_neg_raw=("ret2_neg", "sum"),
            sum_ret4=("ret4", "sum"),
            n_intraday=("intraday_ret", "count"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    daily["RQ_raw"] = daily["n_intraday"] / 3.0 * daily["sum_ret4"]
    close_daily = (
        df.groupby(["asset", "date"])
        .agg(close=("close", "last"))
        .reset_index()
        .sort_values(["asset", "date"])
    )
    close_daily["ret"] = close_daily.groupby("asset")["close"].transform(
        lambda x: np.log(x).diff()
    )
    daily = daily.merge(
        close_daily[["asset", "date", "close", "ret"]],
        on=["asset", "date"],
        how="left",
    )
    daily = daily.drop(columns=["sum_ret4"])
    daily["dollar_volume"] = daily["close"] * daily["volume"]
    daily["RV"] = np.sqrt(daily["RV_raw"] * 252.0) * 100.0
    daily["RV_pos"] = np.sqrt(daily["RV_pos_raw"] * 252.0) * 100.0
    daily["RV_neg"] = np.sqrt(daily["RV_neg_raw"] * 252.0) * 100.0
    daily["RQ"] = np.sqrt(np.maximum(daily["RQ_raw"], 0.0)) * 10000.0
    return daily


# ============================================================
# 4. Helper: flatten MultiIndex columns from yfinance
# ============================================================

def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ============================================================
# 5. Download external macro data
# ============================================================

def download_vix(start_date, end_date):
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        vix = _flatten_columns(vix)
        vix = vix[["Close"]].rename(columns={"Close": "VIX"})
        vix.index = pd.to_datetime(vix.index)
        vix.index.name = "date"
        vix = vix.reset_index()
        vix["date"] = pd.to_datetime(vix["date"])
        print(f"Downloaded VIX: {len(vix)} rows")
        return vix
    except Exception as e:
        print(f"Warning: Could not download VIX ({e}). Skipping.")
        return None


def download_hsi(start_date, end_date):
    try:
        import yfinance as yf
        hsi = yf.download("^HSI", start=start_date, end=end_date, progress=False)
        hsi = _flatten_columns(hsi)
        hsi = hsi[["Close"]].copy()
        hsi["HSI"] = np.log(hsi["Close"]).diff() ** 2
        hsi = hsi[["HSI"]]
        hsi.index = pd.to_datetime(hsi.index)
        hsi.index.name = "date"
        hsi = hsi.reset_index().dropna()
        hsi["date"] = pd.to_datetime(hsi["date"])
        print(f"Downloaded HSI: {len(hsi)} rows")
        return hsi
    except Exception as e:
        print(f"Warning: Could not download HSI ({e}). Skipping.")
        return None


def download_us3m(start_date, end_date):
    try:
        import yfinance as yf
        tb = yf.download("^IRX", start=start_date, end=end_date, progress=False)
        tb = _flatten_columns(tb)
        tb = tb[["Close"]].rename(columns={"Close": "US3M_raw"})
        tb["US3M"] = tb["US3M_raw"].diff()
        tb = tb[["US3M"]]
        tb.index = pd.to_datetime(tb.index)
        tb.index.name = "date"
        tb = tb.reset_index().dropna()
        tb["date"] = pd.to_datetime(tb["date"])
        print(f"Downloaded US3M: {len(tb)} rows")
        return tb
    except Exception as e:
        print(f"Warning: Could not download US3M ({e}). Skipping.")
        return None


def download_ea(asset_names, start_date, end_date):
    try:
        import yfinance as yf
        all_dates = set()
        for asset in asset_names:
            try:
                ticker = yf.Ticker(asset)
                cal = ticker.calendar
                if cal is not None:
                    if isinstance(cal, dict) and "Earnings Date" in cal:
                        dates = pd.to_datetime(cal["Earnings Date"])
                        all_dates.update(dates)
                    elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                        dates = pd.to_datetime(cal["Earnings Date"].dropna())
                        all_dates.update(dates)
            except Exception:
                continue
        date_range = pd.date_range(start_date, end_date, freq="B")
        ea_df = pd.DataFrame({"date": date_range})
        ea_df["EA"] = ea_df["date"].isin(all_dates).astype(float)
        ea_df["date"] = pd.to_datetime(ea_df["date"])
        print(f"Downloaded EA: {len(ea_df)} rows, {int(ea_df['EA'].sum())} announcement days")
        return ea_df
    except Exception as e:
        print(f"Warning: Could not download EA ({e}). Skipping.")
        return None


def load_epu(epu_path=None):
    if epu_path is None or not Path(epu_path).exists():
        print("Warning: EPU file not found. Skipping.")
        return None
    try:
        epu = pd.read_csv(epu_path)
        epu.columns = [c.strip() for c in epu.columns]
        year_col = next((c for c in epu.columns if "year" in c.lower()), None)
        month_col = next((c for c in epu.columns if "month" in c.lower()), None)
        epu_col = next((c for c in epu.columns if "epu" in c.lower() or "uncertainty" in c.lower()), None)
        if year_col and month_col and epu_col:
            epu["date"] = pd.to_datetime(
                epu[year_col].astype(int).astype(str) + "-" +
                epu[month_col].astype(int).astype(str) + "-01"
            )
            epu = epu[["date", epu_col]].rename(columns={epu_col: "EPU_monthly"})
            epu = epu.sort_values("date")
            print(f"Loaded EPU: {len(epu)} monthly rows")
            return epu
        else:
            print(f"Warning: EPU columns not recognized: {epu.columns.tolist()}")
            return None
    except Exception as e:
        print(f"Warning: Could not load EPU ({e}). Skipping.")
        return None


def download_ads(start_date, end_date):
    try:
        import pandas_datareader as pdr
        ads = pdr.get_data_fred("ADSINDEX", start=start_date, end=end_date)
        ads = ads.rename(columns={"ADSINDEX": "ADS"})
        ads.index.name = "date"
        ads = ads.reset_index()
        ads["date"] = pd.to_datetime(ads["date"])
        ads["ADS"] = ads["ADS"].ffill()
        print(f"Downloaded ADS: {len(ads)} rows")
        return ads
    except Exception as e:
        print(f"Warning: Could not download ADS ({e}). Skipping.")
        return None


# ============================================================
# 6. Merge external data into daily panel
# ============================================================

def attach_macro_data(daily, data_dir=None, start_date=None, end_date=None):
    daily = daily.copy()
    daily = _flatten_columns(daily)
    daily["date"] = pd.to_datetime(daily["date"])

    if start_date is None:
        start_date = daily["date"].min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = (daily["date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    # VIX
    vix = download_vix(start_date, end_date)
    if vix is not None:
        daily = daily.merge(vix, on="date", how="left")

    # HSI
    hsi = download_hsi(start_date, end_date)
    if hsi is not None:
        daily = daily.merge(hsi, on="date", how="left")

    # US3M
    us3m = download_us3m(start_date, end_date)
    if us3m is not None:
        daily = daily.merge(us3m, on="date", how="left")

    # EPU
    epu_path = (data_dir / "epu.csv") if data_dir is not None else None
    epu_monthly = load_epu(epu_path)
    if epu_monthly is not None:
        epu_monthly["date"] = pd.to_datetime(epu_monthly["date"])
        all_dates = daily[["date"]].drop_duplicates().sort_values("date")
        all_dates = all_dates.merge(
            epu_monthly.rename(columns={"EPU_monthly": "EPU"}),
            on="date",
            how="left",
        )
        all_dates["EPU"] = all_dates["EPU"].ffill()
        daily = daily.merge(all_dates, on="date", how="left")

    # EA
    asset_names = list(daily["asset"].unique()) if "asset" in daily.columns else []
    ea = download_ea(asset_names, start_date, end_date)
    if ea is not None:
        daily = daily.merge(ea, on="date", how="left")

    # ADS
    ads = download_ads(start_date, end_date)
    if ads is not None:
        daily = daily.merge(ads, on="date", how="left")

    return daily
