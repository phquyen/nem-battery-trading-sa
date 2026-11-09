"""Resampling and feature engineering for day-ahead price forecasting.

Forecast setup (Option B): at midnight (market time), predict all 48
half-hourly prices of the next day. Every feature must be known at midnight.
"""
import pandas as pd

SLOTS_PER_DAY = 48


def to_half_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """5-min data -> 30-min data (mean of six intervals). Labelled by START time."""
    hh = (
        df.set_index("interval_start")
          .resample("30min")
          .agg({"rrp": "mean", "total_demand_mw": "mean"})
          .reset_index()
    )
    return hh


def add_features(hh: pd.DataFrame) -> pd.DataFrame:
    """Add features that are all known at midnight before the target day."""
    hh = hh.sort_values("interval_start").reset_index(drop=True)
    ts = hh["interval_start"]

    # Calendar features (always known in advance)
    hh["date"] = ts.dt.normalize()
    hh["slot"] = ts.dt.hour * 2 + ts.dt.minute // 30      # total 47 slot per day, 0 = 00:00-00:30, 47 = 23:30-00:00 -> the model easy to learn the daily pattern
    hh["dayofweek"] = ts.dt.dayofweek                      # 0 = Monday
    hh["month"] = ts.dt.month
    hh["is_weekend"] = (hh["dayofweek"] >= 5).astype(int)

    # Same-slot lags (safe because the data has no gaps)
    hh["rrp_lag_1d"] = hh["rrp"].shift(SLOTS_PER_DAY)              # same half-hour yesterday
    hh["rrp_lag_7d"] = hh["rrp"].shift(7 * SLOTS_PER_DAY)          # same half-hour last week
    hh["demand_lag_1d"] = hh["total_demand_mw"].shift(SLOTS_PER_DAY)

    # Same-slot median over the previous 7 days (shift(1) excludes today)
    hh["rrp_slot_median_7d"] = (
        hh.groupby("slot")["rrp"]
          .transform(lambda s: s.shift(1).rolling(7).median())
    )

    # Previous-day summary stats: computed per day, then moved forward 1 day
    daily = hh.groupby("date")["rrp"].agg(
        prev_day_mean="mean",
        prev_day_max="max",
        prev_day_neg_share=lambda s: (s < 0).mean(),
    )
    daily.index = daily.index + pd.Timedelta(days=1)
    hh = hh.merge(daily, left_on="date", right_index=True, how="left")

    return hh


FEATURES = [
    "slot", "dayofweek", "month", "is_weekend",
    "rrp_lag_1d", "rrp_lag_7d", "demand_lag_1d",
    "rrp_slot_median_7d",
    "prev_day_mean", "prev_day_max", "prev_day_neg_share",
]
TARGET = "rrp"

# Chronological split boundaries (inclusive starts)
VALID_START = pd.Timestamp("2024-10-01")
TEST_START = pd.Timestamp("2025-10-01")


def split_by_time(df: pd.DataFrame):
    """Chronological train / validation / test split by target date."""
    train = df[df["date"] < VALID_START]
    valid = df[(df["date"] >= VALID_START) & (df["date"] < TEST_START)]
    test = df[df["date"] >= TEST_START]
    return train, valid, test