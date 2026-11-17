"""Precompute everything the Streamlit dashboard shows.

Run from the project root, with the .venv active:
    python scripts/build_dashboard_data.py

This re-creates the FINAL model exactly as in the final test evaluation
(LightGBM, MAE objective, trained on train + validation, fixed settings and seed)
and scores it once on the test year. Nothing is tuned here.

Outputs (small files, committed so the app can run in the cloud):
    data/dashboard/prices.parquet     test-year half-hours: actual price + every forecast
    data/dashboard/schedules.parquet  half-hourly battery schedule per strategy
    data/dashboard/daily.parquet      daily profit per strategy
"""
import lightgbm as lgb
import pandas as pd

from nem_battery.battery import BatterySpec, optimise_day, realised_profit
from nem_battery.evaluate import mae
from nem_battery.features import FEATURES, TARGET, split_by_time
from nem_battery.load import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "dashboard"

# Same settings as the final test evaluation (see report/figures.py)
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, random_state=42, verbose=-1)

# Strategy name -> column of prices used to PLAN the day (None = actual prices)
STRATEGIES = {
    "Perfect foresight": None,
    "Timing-robust ML": "forecast_timing",
    "LightGBM forecast": "forecast",
    "Typical-day rule": "typical",
}


def build_prices(test: pd.DataFrame, trainval: pd.DataFrame) -> pd.DataFrame:
    """Actual test-year prices plus every forecast / planning curve."""
    model = lgb.LGBMRegressor(objective="l1", **PARAMS).fit(trainval[FEATURES], trainval[TARGET])

    p = test[["interval_start", "date", "slot", TARGET, "total_demand_mw"]].copy()
    p["forecast"] = model.predict(test[FEATURES])
    # Timing-robust plan: smooth the forecast over +/- 30 minutes within each day
    p["forecast_timing"] = (p.sort_values("interval_start")
                             .groupby("date")["forecast"]
                             .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))
    # Typical-day rule: median price of each half-hour slot in train + validation
    p["typical"] = p["slot"].map(trainval.groupby("slot")[TARGET].median())
    # Baselines for the accuracy table
    p["slot_median_7d"] = test["rrp_slot_median_7d"]
    p["global_median"] = trainval[TARGET].median()
    return p.sort_values("interval_start").reset_index(drop=True)


def build_schedules(prices: pd.DataFrame, spec: BatterySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plan each day per strategy, pay at ACTUAL prices (same logic as battery.simulate)."""
    sched_parts, daily_rows = [], []
    for date, day in prices.groupby("date"):
        day = day.sort_values("slot")
        actual = day[TARGET].to_numpy()
        for name, col in STRATEGIES.items():
            plan = actual if col is None else day[col].to_numpy()
            s = optimise_day(plan, spec)
            profit = realised_profit(s, actual, spec)

            s.insert(0, "strategy", name)
            s.insert(0, "slot", day["slot"].to_numpy())
            s.insert(0, "interval_start", day["interval_start"].to_numpy())
            s.insert(0, "date", date)
            s["plan_price"] = plan
            s["actual_price"] = actual
            # Cash earned in each half-hour, at actual prices, after degradation
            s["revenue"] = (actual * (s["discharge_mw"] - s["charge_mw"])
                            - spec.degradation_cost * s["discharge_mw"]) * spec.dt_h
            sched_parts.append(s)
            daily_rows.append({"date": date, "strategy": name, "profit": profit,
                               "mwh_out": s["discharge_mw"].sum() * spec.dt_h})

    schedules = pd.concat(sched_parts, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    # Daily price context, for the "interesting days" picker
    ctx = prices.groupby("date")[TARGET].agg(day_max="max", day_min="min", day_mean="mean")
    daily = daily.merge(ctx, left_on="date", right_index=True)
    return schedules, daily


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features_sa1_30min.parquet")
    train, valid, test = split_by_time(df)
    trainval = pd.concat([train, valid])

    prices = build_prices(test, trainval)
    print("Forecast MAE on test year ($/MWh):")
    for col in ["forecast", "slot_median_7d", "global_median"]:
        print(f"  {col:<16} {mae(prices[TARGET], prices[col]):.2f}")

    print("Simulating the test year (about a minute)...")
    spec = BatterySpec()
    schedules, daily = build_schedules(prices, spec)

    totals = daily.groupby("strategy")["profit"].sum()
    perfect = totals["Perfect foresight"]
    print("Battery profit on test year:")
    for name in STRATEGIES:
        print(f"  {name:<18} ${totals[name]:>10,.0f}  ({totals[name] / perfect:.1%} of perfect)")

    prices.to_parquet(OUT_DIR / "prices.parquet", index=False)
    schedules.to_parquet(OUT_DIR / "schedules.parquet", index=False)
    daily.to_parquet(OUT_DIR / "daily.parquet", index=False)
    for f in sorted(OUT_DIR.glob("*.parquet")):
        print(f"saved {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
