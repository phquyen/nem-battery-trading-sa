"""Regenerate the four README charts from the saved project data.

Run from the project root, with the .venv active:
    python scripts/make_figures.py

Charts are saved to reports/figures/.
"""
import matplotlib

matplotlib.use("Agg")  # save files only, no pop-up windows

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

from nem_battery.battery import BatterySpec, simulate
from nem_battery.features import FEATURES, TARGET, split_by_time
from nem_battery.load import INTERIM_DIR, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, random_state=42, verbose=-1)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_negative_share(df5: pd.DataFrame) -> None:
    """Share of 5-min intervals with a negative price, by hour of day."""
    hour = df5["interval_start"].dt.hour
    share = (df5["rrp"] < 0).groupby(hour).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    share.plot(kind="bar", ax=ax)
    ax.set_title("Share of 5-min intervals with a negative price, by hour (SA1, Oct 2021 – Sep 2026)")
    ax.set_xlabel("Hour of day (interval start, market time UTC+10)")
    ax.set_ylabel("% of intervals below $0")
    save(fig, "q3_negative_share_by_hour.png")


def fig_heatmap(df5: pd.DataFrame) -> None:
    """Median price by year and month."""
    ts = df5["interval_start"]
    table = (df5.groupby([ts.dt.year.rename("year"), ts.dt.month.rename("month")])["rrp"]
                .median()
                .unstack()
                .reindex(columns=range(1, 13))
                .round(0))

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(table.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Median RRP ($/MWh)")
    ax.set_title("SA1 median price by year and month")
    save(fig, "q4b_median_price_heatmap.png")


def fig_importance(model: lgb.LGBMRegressor) -> None:
    """Gain-based feature importance of the final LightGBM model."""
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=FEATURES)
    fig, ax = plt.subplots(figsize=(8, 5))
    (imp / imp.sum()).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Final LightGBM (MAE objective): feature importance, share of gain")
    ax.set_xlabel("Share of total gain")
    save(fig, "b3_lgbm_feature_importance.png")


def fig_test_profit(test: pd.DataFrame, trainval: pd.DataFrame, model: lgb.LGBMRegressor) -> None:
    """Cumulative profit of each strategy on the held-out test year."""
    test = test.copy()
    test["forecast"] = model.predict(test[FEATURES])
    test["forecast_timing"] = (test.sort_values("interval_start")
                                   .groupby("date")["forecast"]
                                   .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))
    test["typical"] = test["slot"].map(trainval.groupby("slot")[TARGET].median())

    spec = BatterySpec()
    runs = {
        "perfect foresight (upper bound)": simulate(test, None, spec, spec),
        "timing-robust ML": simulate(test, "forecast_timing", spec, spec),
        "ML forecast": simulate(test, "forecast", spec, spec),
        "typical-day rule": simulate(test, "typical", spec, spec),
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, r in runs.items():
        ax.plot(r["date"], r["profit"].cumsum(), label=f"{name}: ${r['profit'].sum():,.0f}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Cumulative profit, simulated 1 MW / 2 MWh battery, test year (Oct 2025 – Sep 2026)")
    ax.set_ylabel("Cumulative profit ($)")
    ax.legend()
    save(fig, "o2_test_cumulative_profit.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df5 = pd.read_parquet(INTERIM_DIR / "price_demand_sa1.parquet")
    fig_negative_share(df5)
    fig_heatmap(df5)

    model_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features_sa1_30min.parquet")
    train, valid, test = split_by_time(model_df)
    trainval = pd.concat([train, valid])

    final_model = lgb.LGBMRegressor(objective="l1", **PARAMS).fit(trainval[FEATURES], trainval[TARGET])
    fig_importance(final_model)

    print("Simulating the test year (about a minute)...")
    fig_test_profit(test, trainval, final_model)
    print("Done. Charts are in", FIG_DIR)


if __name__ == "__main__":
    main()