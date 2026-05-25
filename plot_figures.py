import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})


def plot_model_performance(summary_csv_path, output_dir, horizon=1):
    """
    Figure 1: Bar chart of out-of-sample MSE for all models.
    """
    if not Path(summary_csv_path).exists():
        print(f"Warning: File not found: {summary_csv_path}. Skipping Figure 1.")
        return

    df = pd.read_csv(summary_csv_path)
    if 'Model' in df.columns:
        df = df.set_index('Model')
    elif 'model' in df.columns:
        df = df.set_index('model')
    else:
        df = df.set_index(df.columns[0])

    mse_col = [c for c in df.columns if 'MSE' in c.upper() or 'mse' in c.lower()][0]
    plot_data = df[mse_col].sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4' if 'HAR' in idx else '#ff7f0e' for idx in plot_data.index]
    bars = plt.barh(plot_data.index, plot_data.values, color=colors, edgecolor='none', height=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2,
                 f'{width:.5f}',
                 va='center', ha='left', fontsize=9, color='#333333')

    plt.title(f"Model Out-of-Sample Performance (OOS MSE) - Horizon: {horizon}", pad=20)
    plt.xlabel("Mean Squared Error (MSE)")
    plt.ylabel("Models")
    plt.tight_layout()

    save_path = Path(output_dir) / f"figure1_model_mse_h{horizon}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to: {save_path}")


def plot_pairwise_dm_heatmap(raw_csv_path, table_csv_path, output_dir, horizon=1):
    """
    Figure 2: Heatmap of pairwise relative MSE with DM test significance.
    """
    if not (Path(raw_csv_path).exists() and Path(table_csv_path).exists()):
        print(f"Warning: Pairwise CSV files not found. Skipping Figure 2.")
        return

    raw_df = pd.read_csv(raw_csv_path, index_col=0)
    table_df = pd.read_csv(table_csv_path, index_col=0)

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        raw_df,
        annot=table_df.astype(str),
        fmt="",
        cmap="RdBu_r",
        center=1.0,
        cbar_kws={'label': 'MSE Ratio (Column MSE / Row MSE)'},
        linewidths=0.5,
        annot_kws={"size": 9}
    )

    plt.title(
        f"Pairwise Relative MSE & DM Test Significance Heatmap (Horizon: {horizon})\n"
        "Note: Ratio < 1 implies Column Model outperforms Row Model.\n"
        "Significance: [p<0.01], (p<0.05), p*<0.10",
        pad=20
    )
    plt.xlabel("Column Model (j)")
    plt.ylabel("Row Model (i)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = Path(output_dir) / f"figure2_pairwise_dm_heatmap_h{horizon}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to: {save_path}")


def plot_cumulative_csfe_difference(y_test, predictions, output_dir, horizon=1, dates=None):
    """
    Figure 3: Cumulative sum of squared forecast error (CSFE) differences
    relative to HAR benchmark.
    """
    if "HAR" not in predictions:
        print("Note: 'HAR' not found in predictions. Cannot compute CSFE differences.")
        return

    y_test = np.asarray(y_test)
    har_error_sq = (y_test - np.asarray(predictions["HAR"])) ** 2

    plt.figure(figsize=(12, 6))

    target_models = ["LA", "A-LA", "RF", "GB", "NN13", "NN104"]
    available_targets = [m for m in target_models if m in predictions]

    if not available_targets:
        available_targets = [m for m in predictions.keys() if m != "HAR"][:5]

    x_axis = np.arange(len(y_test)) if dates is None else dates

    for model_name in available_targets:
        model_error_sq = (y_test - np.asarray(predictions[model_name])) ** 2
        csfe_diff = np.cumsum(har_error_sq - model_error_sq)
        plt.plot(x_axis, csfe_diff, label=f"{model_name} vs HAR", linewidth=2)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.title(f"Cumulative Sum of Squared Forecast Error Differences (Horizon: {horizon})", pad=15)
    plt.xlabel("Test Set Timeline" if dates is None else "Date")
    plt.ylabel("Cumulative Error Reduction (sum of HAR - ML squared errors)")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    save_path = Path(output_dir) / f"figure3_cumulative_csfe_h{horizon}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to: {save_path}")


if __name__ == "__main__":
    import config
    output_path = Path(config.OUTPUT_DIR)

    print("=" * 60)
    print("Generating figures from saved output files...")
    print("=" * 60)

    horizons = [1, 5, 22]

    for h in horizons:
        summary_csv = output_path / f"model_mse_summary_h{h}.csv"
        raw_csv = output_path / f"pairwise_mse_raw_h{h}.csv"
        table_csv = output_path / f"pairwise_mse_table_h{h}.csv"

        if summary_csv.exists() or raw_csv.exists():
            print(f"\nProcessing horizon = {h}...")
            plot_model_performance(summary_csv, output_path, horizon=h)
            plot_pairwise_dm_heatmap(raw_csv, table_csv, output_path, horizon=h)
        else:
            print(f"\nNo output data found for horizon = {h}. Skipping.")

    print("\nDone. Note: Figure 3 (CSFE) requires running via main.py.")
