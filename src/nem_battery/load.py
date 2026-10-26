"""Load and combine raw AEMO price and demand files."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def load_raw(region: str = "SA1") -> pd.DataFrame:
    """Read all monthly CSVs for a region into one DataFrame."""
    files = sorted(RAW_DIR.glob(f"PRICE_AND_DEMAND_*_{region}.csv"))
    print(f"Found {len(files)} files")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    df = df.rename(columns={
        "REGION": "region",
        "SETTLEMENTDATE": "settlement_date",
        "TOTALDEMAND": "total_demand_mw",
        "RRP": "rrp",
        "PERIODTYPE": "period_type",
    })

    df["settlement_date"] = pd.to_datetime(df["settlement_date"], format="%Y/%m/%d %H:%M:%S")
    df["interval_start"] = df["settlement_date"] - pd.Timedelta(minutes=5)

    return df.sort_values("settlement_date").reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw("SA1")
    print(df.head())
    print(df.dtypes)
    print(len(df))