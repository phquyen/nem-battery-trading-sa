"""Streamlit dashboard: Intelligent Battery Trading in South Australia.

Reads ONLY the precomputed files in data/dashboard/ (made by
scripts/build_dashboard_data.py). No model is trained here.

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dashboard"

# One fixed colour per strategy, used on every chart
COLORS = {
    "Timing-robust ML": "#2a78d6",
    "LightGBM forecast": "#eb6834",
    "Typical-day rule": "#1baf7a",
    "Perfect foresight": "#8a8985",
}
STRATEGY_ORDER = list(COLORS)
TRADEABLE = ["Timing-robust ML", "LightGBM forecast", "Typical-day rule"]
NEUTRAL = "#8a8985"
INK = "#52514e"
# Single-day charts use "hours since midnight" (a plain number, 0-24) instead of
# datetimes, so the browser can never shift them into the viewer's time zone.
HOUR_X = dict(title="Time of day (NEM market time, UTC+10)", scale=alt.Scale(domain=[0, 24], nice=False),
              axis=alt.Axis(values=list(range(0, 25, 3)),
                            labelExpr="(datum.value < 10 ? '0' : '') + datum.value + ':00'"))

# Facts not stored in the dashboard files were checked against the 5-minute
# AEMO data (data/interim) when the dashboard was built.
INTERESTING_DAYS = {
    "26 Jan 2026 - price cap, ML's best day": (
        "2026-01-26",
        "The 5-minute price hit the $20,300/MWh market cap from 20:35 to 20:45. "
        "This is the day the timing-robust ML strategy gained most over the typical-day rule.",
    ),
    "22 Jun 2026 - morning price cap, every forecast missed": (
        "2026-06-22",
        "The 5-minute price sat at the $20,300/MWh cap from 07:15 to 08:00 on a winter morning. "
        "No strategy based on a forecast or a typical day planned for it, so all earned far less than perfect foresight.",
    ),
    "6 Jan 2026 - ML close to perfect": (
        "2026-01-06",
        "An evening spike that the forecast placed well: the timing-robust ML strategy came close to the perfect-foresight profit.",
    ),
    "25 Dec 2025 - negative prices all day": (
        "2025-12-25",
        "Every half-hour price was below $0. South Australia's grid demand fell to -311 MW at 13:30 "
        "(5-minute data), because rooftop solar exceeded all consumption.",
    ),
    "18 May 2026 - ML's worst day": (
        "2026-05-18",
        "A calm day with a narrow price range. The small gap between cheap and expensive hours did not cover "
        "efficiency losses and degradation, so a slightly wrong forecast turned the trade into a loss.",
    ),
}


def money(x: float) -> str:
    """$1,234 or −$241 (minus sign before the dollar sign)."""
    return f"−${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


# ---------- data ----------
@st.cache_data
def load():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    return prices, schedules, daily


def summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("strategy")["profit"]
    t = pd.DataFrame({
        "Profit": g.sum(),
        "Worst day": g.min(),
        "Losing days": g.apply(lambda s: (s < 0).mean()),
    }).reindex(STRATEGY_ORDER)
    t.insert(1, "% of perfect", t["Profit"] / t.loc["Perfect foresight", "Profit"])
    t.index.name = "Strategy"
    return t


def mae_table(prices: pd.DataFrame) -> pd.DataFrame:
    cols = {"LightGBM": "forecast", "7-day same-slot median (best baseline)": "slot_median_7d",
            "Global median": "global_median"}
    return pd.DataFrame({
        "MAE ($/MWh)": {name: np.mean(np.abs(prices["rrp"] - prices[c])) for name, c in cols.items()}
    })


# ---------- charts ----------
def color_scale(names):
    return alt.Scale(domain=list(names), range=[COLORS[n] for n in names])


def hover_lines(df, x, y, series, x_title, y_title, title, names, dash=None, x_format=None):
    """Line chart with a legend under the plot and a nearest-point hover tooltip."""
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    x_enc = alt.X(f"{x}:T", title=x_title, axis=alt.Axis(format=x_format) if x_format else alt.Axis())
    base = alt.Chart(df).encode(
        x=x_enc,
        y=alt.Y(f"{y}:Q", title=y_title),
        color=alt.Color(f"{series}:N", scale=color_scale(names), sort=list(names),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=0)),
    )
    line_kw = {"strokeWidth": 2}
    lines = base.mark_line(**line_kw)
    if dash is not None:
        lines = lines.encode(strokeDash=alt.StrokeDash(f"{series}:N", scale=dash, legend=None))
    points = base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:T", title=x_title, format=x_format or "%d %b %Y"),
                 alt.Tooltip(f"{series}:N", title="Series"),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=",.0f")],
    ).add_params(nearest)
    rule = alt.Chart(df).mark_rule(color=NEUTRAL, strokeWidth=1).encode(x=f"{x}:T").transform_filter(nearest)
    return (lines + points + rule).properties(title=title, height=340)


# ---------- page ----------
st.set_page_config(page_title="SA Battery Trading", page_icon="🔋", layout="wide")
prices, schedules, daily = load()
summary = summary_table(daily)
ml_total = summary.loc["Timing-robust ML", "Profit"]
rule_total = summary.loc["Typical-day rule", "Profit"]

st.title("Intelligent Battery Trading in South Australia")
st.caption("Day-ahead electricity price forecasting and battery dispatch on real AEMO market data "
           "(region SA1). Results are for a held-out test year, Oct 2025 – Sep 2026.")

st.subheader("About this project")
st.markdown(
    "South Australia's electricity prices swing wildly: often **below zero at midday** when solar floods the grid, "
    "and sometimes **thousands of dollars per MWh** in the evening. A battery can make money by charging when power "
    "is cheap and selling when it is expensive, but it has to decide *before* it knows the prices. "
    "This project forecasts tomorrow's prices with machine learning, then turns the forecast into a "
    "charge/discharge plan for a simulated 1 MW / 2 MWh battery."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Timing-robust ML profit", f"${ml_total:,.0f}")
c2.metric("vs typical-day rule", f"${rule_total:,.0f}", f"+{ml_total / rule_total - 1:.1%} for ML")
c3.metric("Share of perfect foresight", f"{summary.loc['Timing-robust ML', '% of perfect']:.1%}")
c4.metric("Forecast error (MAE), LightGBM", f"${mae_table(prices).iloc[0, 0]:.2f}/MWh")

with st.expander("How to read this dashboard"):
    st.markdown(
        """
- **Prices** are in **$/MWh** (dollars per megawatt-hour), the wholesale price of electricity in each half-hour.
- **The battery** can charge or discharge up to **1 MW** at a time and store **2 MWh**, about two hours of full output.
  It starts and ends every day empty.
- **Each strategy plans the whole next day at midnight**, then is paid at the prices that actually happened:
  - **Timing-robust ML**: plans with a machine-learning (LightGBM) forecast, smoothed over ±30 minutes so the plan does not bet everything on one exact half-hour.
  - **LightGBM forecast**: plans with the raw machine-learning forecast.
  - **Typical-day rule**: plans with the median price of each half-hour from past years (a simple baseline).
  - **Perfect foresight** (grey dashed line): plans with the *actual* prices. This is impossible in practice and shows the most that could have been earned.
- **% of perfect** (share of perfect foresight) is how much of that maximum each strategy captured.
- **MAE** (mean absolute error) is the average size of a forecast's mistake in $/MWh. Lower is better.
- Hover over any chart to see exact values. Use **Explore a single day** to see what the battery did, half-hour by half-hour.
        """
    )

# ---- Results ----
st.header("Results for the test year")
left, right = st.columns([3, 2])

with left:
    cum = daily.sort_values("date").copy()
    cum["cumulative"] = cum.groupby("strategy")["profit"].cumsum()
    dash = alt.Scale(domain=STRATEGY_ORDER, range=[[1, 0], [1, 0], [1, 0], [6, 4]])
    st.altair_chart(
        hover_lines(cum, "date", "cumulative", "strategy", "Date", "Cumulative profit ($)",
                    "Cumulative profit by strategy, Oct 2025 – Sep 2026", STRATEGY_ORDER, dash=dash),
        width="stretch",
    )

with right:
    st.markdown("**Battery profit, test year**")
    st.dataframe(
        summary.style.format({"Profit": money, "% of perfect": "{:.1%}",
                              "Worst day": money, "Losing days": "{:.1%}"}),
        width="stretch",
    )
    st.markdown("**Forecast accuracy, test year**")
    st.dataframe(mae_table(prices).style.format("{:.2f}"), width="stretch")
    st.caption("Losing days = share of days where the plan lost money at actual prices.")

# Monthly advantage
monthly = (daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
                .pivot_table(index="month", columns="strategy", values="profit", aggfunc="sum"))
monthly = (monthly["Timing-robust ML"] - monthly["Typical-day rule"]).rename("advantage").reset_index()
monthly["Who earned more"] = np.where(monthly["advantage"] >= 0, "Timing-robust ML", "Typical-day rule")
n_win = int((monthly["advantage"] > 0).sum())
monthly["label"] = monthly["month"].dt.strftime("%b %Y")  # text labels: one per bar, no time-zone shifts
st.altair_chart(
    alt.Chart(monthly).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("label:O", title="Month", sort=list(monthly["label"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("advantage:Q", title="Difference ($)"),
        color=alt.Color("Who earned more:N",
                        scale=alt.Scale(domain=["Timing-robust ML", "Typical-day rule"],
                                        range=[COLORS["Timing-robust ML"], COLORS["Typical-day rule"]]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("label:O", title="Month"),
                 alt.Tooltip("advantage:Q", title="ML minus rule ($)", format=",.0f")],
    ).properties(title=f"Monthly profit difference: timing-robust ML minus typical-day rule "
                       f"(ML ahead in {n_win} of {len(monthly)} months)", height=320),
    width="stretch",
)
st.caption("Much of the year's gap comes from January 2026, a month with price-cap events. "
           "Use the day explorer below to see those days.")

# ---- Single day ----
st.header("Explore a single day")
pick_col, strat_col = st.columns([3, 2])
with pick_col:
    choice = st.selectbox("Interesting days (or pick any date below)",
                          ["Choose your own date"] + list(INTERESTING_DAYS))
    dates = sorted(daily["date"].dt.date.unique())
    if choice == "Choose your own date":
        day = st.date_input("Date", value=dates[0], min_value=dates[0], max_value=dates[-1])
        note = None
    else:
        day = pd.Timestamp(INTERESTING_DAYS[choice][0]).date()
        note = INTERESTING_DAYS[choice][1]
with strat_col:
    strategy = st.radio("Strategy to show", STRATEGY_ORDER, index=0)

day_ts = pd.Timestamp(day)
if note:
    st.info(note)

d_daily = daily[daily["date"] == day_ts].set_index("strategy").reindex(STRATEGY_ORDER)
m = st.columns(5)
for col, name in zip(m[:4], STRATEGY_ORDER):
    col.metric(name, money(d_daily.loc[name, "profit"]))
m[4].metric("Highest half-hour price", f"{money(d_daily['day_max'].iloc[0])}/MWh")

d = schedules[(schedules["date"] == day_ts) & (schedules["strategy"] == strategy)].copy()
d["hour"] = d["slot"] * 0.5               # slot 27 -> 13.5 hours after midnight
d["hour_end"] = d["hour"] + 0.5
d["time"] = d["interval_start"].dt.strftime("%H:%M")  # text label for tooltips

# Chart 1: actual price vs the prices the plan was built on
cols = ["hour", "time"]
if strategy == "Perfect foresight":
    plot = d[cols + ["actual_price"]].rename(columns={"actual_price": "price"}).assign(series="Actual price")
    names, rng, dash_rng = ["Actual price"], [INK], [[1, 0]]
else:
    plan_name = f"Price used for planning ({strategy})"
    plot = pd.concat([
        d[cols + ["actual_price"]].rename(columns={"actual_price": "price"}).assign(series="Actual price"),
        d[cols + ["plan_price"]].rename(columns={"plan_price": "price"}).assign(series=plan_name),
    ])
    names, rng, dash_rng = ["Actual price", plan_name], [INK, COLORS[strategy]], [[1, 0], [6, 4]]

nearest = alt.selection_point(nearest=True, on="pointerover", fields=["hour"], empty=False)
base = alt.Chart(plot).encode(
    x=alt.X("hour:Q", **HOUR_X),
    y=alt.Y("price:Q", title="Price ($/MWh)"),
    color=alt.Color("series:N", scale=alt.Scale(domain=names, range=rng), legend=alt.Legend(orient="bottom", title=None, labelLimit=0)),
)
price_chart = (
    base.mark_line(strokeWidth=2, interpolate="step-after").encode(
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=dash_rng), legend=None))
    + base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("time:N", title="Half-hour starting"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("price:Q", title="$/MWh", format=",.2f")]).add_params(nearest)
    + alt.Chart(plot).mark_rule(color=NEUTRAL).encode(x="hour:Q").transform_filter(nearest)
).properties(title=f"Half-hourly price on {day_ts:%d %b %Y}: actual vs the prices the plan used", height=380)
st.altair_chart(price_chart, width="stretch")

# Chart 2: what the battery did
d["power_mw"] = d["discharge_mw"] - d["charge_mw"]
d["Action"] = np.select([d["power_mw"] > 1e-6, d["power_mw"] < -1e-6],
                        ["Discharging (selling)", "Charging (buying)"], "Idle")
d = d[d["Action"] != "Idle"]
if d.empty:
    st.write(f"The **{strategy}** plan did not trade on this day.")
else:
    st.altair_chart(
        alt.Chart(d).mark_bar(cornerRadiusEnd=4, stroke=None).encode(
            x=alt.X("hour:Q", **HOUR_X),
            x2="hour_end:Q",  # each bar spans exactly its half-hour
            y=alt.Y("power_mw:Q", title="Power (MW)", scale=alt.Scale(domain=[-1.05, 1.05])),
            y2=alt.datum(0),  # bars grow from 0 MW
            color=alt.Color("Action:N",
                            scale=alt.Scale(domain=["Charging (buying)", "Discharging (selling)"],
                                            range=[NEUTRAL, COLORS[strategy] if strategy != "Perfect foresight" else INK]),
                            legend=alt.Legend(orient="bottom", title=None, labelLimit=0)),
            tooltip=[alt.Tooltip("time:N", title="Half-hour starting"),
                     alt.Tooltip("Action:N"),
                     alt.Tooltip("power_mw:Q", title="MW", format=".2f"),
                     alt.Tooltip("soc_mwh:Q", title="Stored after (MWh)", format=".2f"),
                     alt.Tooltip("actual_price:Q", title="Actual price ($/MWh)", format=",.2f"),
                     alt.Tooltip("revenue:Q", title="Cash this half-hour ($)", format=",.0f")],
        ).properties(title=f"What the battery did ({strategy} plan): bars above 0 = selling, below 0 = buying", height=320),
        width="stretch",
    )
    st.caption("Each bar is one half-hour (1 MW for 30 minutes = 0.5 MWh). Profit is paid at the actual price.")

# ---- Limitations ----
st.header("Assumptions and limitations")
st.markdown(
    """
- **Simulated, not real trading.** Results come from replaying historical prices; they are not real trading performance.
- **Simplified battery:** 1 MW / 2 MWh, 90% round-trip efficiency, $10/MWh degradation cost, planned once per day,
  starting and ending each day empty, no other cycle limits.
- **Price-taker:** the battery's trades are assumed not to move market prices.
- **Day-ahead only:** real batteries re-plan continuously during the day and also earn from frequency-control (FCAS) markets, which are not modelled.
- **No network charges, bidding rules or regulatory detail.**
- **Forecast uncertainty is not well calibrated yet:** the raw 80% prediction intervals from quantile LightGBM covered only
  **68.4%** of actual prices on the validation year, so they were overconfident. They are not used in this dashboard.
- **Extreme spikes are essentially unpredictable** a day ahead, as 22 Jun 2026 shows.
- The test year was used **once**, for final evaluation only. No model or strategy was tuned on it.
    """
)
st.caption("Data © AEMO, used under the AEMO Copyright Permissions Notice. Built with Python, LightGBM, SciPy and Streamlit.")
