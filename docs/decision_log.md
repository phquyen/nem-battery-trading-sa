# Decision Log

Every methodological choice in this project, with the reason and the alternatives considered.

| Date | Decision | Reason | Alternatives considered |
|---|---|---|---|
| 25/10/2026 | Battery trading on SA1 is the sole flagship project | Covers the full data science pipeline (prediction, uncertainty, optimisation, evaluation) with Australian data, within 8 weeks | Continuing earlier projects (NYC trip demand, Adelaide crime) |
| 25/10/2026 | Region: South Australia (`SA1`) | High price volatility and very high renewable share; relevant to Adelaide employers | NSW1, VIC1 |
| 25/10/2026 | Data source: AEMO aggregated price and demand CSVs, downloaded by `download.py` | Simple, official, one file per region per month | NEMOSIS package; raw NEMweb/MMSDM files |
| 25/10/2026 | Data range: Oct 2021 – Sep 2026 | All intervals under 5-minute settlement (from 1 Oct 2021); 5 years covers several seasons and extreme events | Including pre-Oct 2021 data (mixed 30-min and 5-min granularity) |
| 25/10/2026 | Keep timestamps in market time (UTC+10, no daylight saving) | Avoids misalignment; every day has exactly 48 half-hours | Converting to Adelaide local time |
| 25/10/2026 | pandas + Parquet; no DuckDB for now | Personal preference; Parquet keeps data types, is compressed and fast | DuckDB, SQLite (SQL tool to be revisited) |
| 26/10/2026 | Keep all extreme values: negative prices, the −$1,000 floor, the $20,300 cap, negative demand | They are real market events, not errors, and matter most for a battery | Removing them as outliers |
| 27/10/2026 | Package project code with `pyproject.toml` and `pip install -e .` | Notebooks, scripts and tests import the same code without path workarounds | `sys.path` edits in each notebook |
| 09/11/2026 | Forecast setup: at midnight (market time), predict all 48 half-hourly prices of the next day | Gives the battery a full-day schedule; realistic and feasible in 8 weeks | 5-min-ahead (too short for planning); hourly rolling forecasts (too complex) |
| 09/11/2026 | Resample to 30 minutes using the mean of six 5-min prices | Matches the forecast setup; smooths noise; matches AEMO's post-5MS trading price derivation | Keep 5-min resolution |
| 09/11/2026 | Features: calendar (slot, weekday, month, weekend), lags (1 day, 7 days), 7-day same-slot median, previous-day mean/max/negative share, previous-day demand | All known at midnight before the target day; verified with a manual leakage check (12 Feb 2025, 18:30) | Same-time demand and intraday lags (leakage) |
| 09/11/2026 | Chronological split: train Oct 2021–Sep 2024, validation Oct 2024–Sep 2025, test Oct 2025–Sep 2026 | Full-year validation and test sets balance seasonality; the test set stays untouched until final evaluation | Random split (leakage); shorter test periods (season bias) |
| 09/11/2026 | Keep the 2022 energy crisis in training | A real regime; the model should know extremes can happen | Excluding 2022 (hides a real risk) |

## Open questions

- [ ] Spike threshold: $300, $1,000 or the 99th percentile? (Q5 EDA)
- [ ] Should 2022 be evaluated as a separate regime, or down-weighted in training?
- [ ] Add weather (e.g. Bureau of Meteorology temperature) as a feature? Idea from the 12 Feb 2025 peak.
- [ ] Are night-time negative prices driven by wind? Would need generation data (optional).
- [ ] SQL tool for the portfolio: SQLite or DuckDB?
