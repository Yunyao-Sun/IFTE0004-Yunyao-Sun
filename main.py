import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from data_utils import load_all_assets, make_equal_weight_index_intraday, make_daily_from_intraday, attach_macro_data
from features import make_features, get_feature_sets, make_table1_summary
from models import run_model_zoo
from evaluation import build_pairwise_table, build_mse_summary
from plot_figures import plot_model_performance, plot_pairwise_dm_heatmap, plot_cumulative_csfe_difference


def main():
    np.random.seed(config.RANDOM_STATE)
    tf.keras.utils.set_random_seed(config.RANDOM_STATE)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Volatility forecasting for an equal-weight index")
    print("=" * 80)
    print(f"Data folder: {config.DATA_DIR}")
    print(f"Assets: {config.ASSET_NAMES}")
    print(f"Index name: {config.INDEX_NAME}")
    print(f"NN seeds: {config.NN_SEEDS}")

    intraday_stocks = load_all_assets(config.DATA_DIR, config.ASSET_NAMES)
    intraday_index = make_equal_weight_index_intraday(intraday_stocks, index_name=config.INDEX_NAME)
    print(f"Built equal-weight index: {config.INDEX_NAME}")
    print(f"Index intraday rows before resampling: {len(intraday_index)}")

    # Resample to 5-minute frequency to mitigate microstructure noise
    print("Resampling to 5-minute frequency...")
    intraday_index = intraday_index.set_index("datetime")
    intraday_index = (
        intraday_index
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum", "asset": "last"})
        .dropna(subset=["close"])
        .reset_index()
    )
    intraday_index["date"] = pd.to_datetime(intraday_index["datetime"].dt.date)
    print(f"Index intraday rows after resampling: {len(intraday_index)}")

    daily = make_daily_from_intraday(intraday_index)
    daily = attach_macro_data(daily, data_dir=config.DATA_DIR)
    daily.to_csv(config.OUTPUT_DIR / "daily_realized_measures_index.csv", index=False)
    table1 = make_table1_summary(make_features(daily, horizon=1))
    table1.to_csv(config.OUTPUT_DIR / "table1_summary_statistics.csv", index=False)
    print("\nTable 1 style summary:")
    print(table1)

    for horizon in config.HORIZONS:
        print(f"\n{'='*60}")
        print(f"Running horizon: {horizon} days")

        panel = make_features(daily, horizon=horizon)
        feature_sets = get_feature_sets(panel)

        print("\nFeature sets:")
        for name, cols in feature_sets.items():
            print(f"  {name}: {cols}")

        print(f"\nRunning model zoo (horizon={horizon})...")
        result = run_model_zoo(panel, feature_sets, config)

        raw_table, formatted_table = build_pairwise_table(result, horizon=horizon)
        mse_df, mse_summary = build_mse_summary(result)

        panel.to_csv(config.OUTPUT_DIR / f"daily_model_data_h{horizon}.csv", index=False)
        raw_table.to_csv(config.OUTPUT_DIR / f"pairwise_mse_raw_h{horizon}.csv")
        formatted_table.to_csv(config.OUTPUT_DIR / f"pairwise_mse_table_h{horizon}.csv")

        with open(config.OUTPUT_DIR / f"pairwise_mse_pretty_h{horizon}.txt", "w", encoding="utf-8") as f:
            f.write(f"Pairwise relative MSE table (horizon={horizon}):\n\n")
            f.write(formatted_table.to_string())
            f.write("\n")

        mse_df.to_csv(config.OUTPUT_DIR / f"model_mse_h{horizon}.csv", index=False)
        mse_summary.to_csv(config.OUTPUT_DIR / f"model_mse_summary_h{horizon}.csv", index=False)

        with open(config.OUTPUT_DIR / f"model_mse_summary_pretty_h{horizon}.txt", "w", encoding="utf-8") as f:
            f.write(f"Model MSE summary (horizon={horizon}):\n\n")
            f.write(mse_summary.to_string(index=False))
            f.write("\n")

        print(f"\nMSE summary (horizon={horizon}):")
        print(mse_summary)
        print(f"\nPairwise MSE table (horizon={horizon}):")
        print(formatted_table)

        print(f"\n[Visualization] Generating figures for Horizon={horizon}...")

        plot_model_performance(
            config.OUTPUT_DIR / f"model_mse_summary_h{horizon}.csv",
            config.OUTPUT_DIR,
            horizon=horizon
        )

        plot_pairwise_dm_heatmap(
            config.OUTPUT_DIR / f"pairwise_mse_raw_h{horizon}.csv",
            config.OUTPUT_DIR / f"pairwise_mse_table_h{horizon}.csv",
            config.OUTPUT_DIR,
            horizon=horizon
        )

        if "y_test" in result and "predictions" in result:
            plot_cumulative_csfe_difference(
                y_test=result["y_test"],
                predictions=result["predictions"],
                output_dir=config.OUTPUT_DIR,
                horizon=horizon
            )

    print("\nDone.")
    print(f"Outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
