"""
Download AEMO aggregated price and demand CSVs for one NEM region.
Time range is specified as YYYY-MM, e.g. 2021-10 to 2026-09.
"""

import time
from pathlib import Path

import pandas as pd
import requests

"""
Source: https://www.aemo.com.au/aemo/data/nem/priceanddemand
"""
BASE_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"   # <- CHANGE 1: added www.
HEADERS = {"User-Agent": "Mozilla/5.0"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def build_url(year_month: str, region: str) -> str:
    return f"{BASE_URL}/PRICE_AND_DEMAND_{year_month}_{region}.csv"


def month_range(start: str, end: str) -> list[str]:
    """'2021-10', '2021-12' -> ['202110', '202111', '202112']"""
    return list(pd.period_range(start, end, freq="M").strftime("%Y%m"))


def download_month(year_month: str, region: str = "SA1") -> str:
    """Download one file. Return 'skipped', 'ok' or 'failed'."""
    out_path = RAW_DIR / f"PRICE_AND_DEMAND_{year_month}_{region}.csv"

    if out_path.exists():
        return "skipped"

    url = build_url(year_month, region)
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        if not response.text.startswith("REGION"):
            print(f"FAILED {year_month}: response is not a CSV")
            return "failed"
        out_path.write_bytes(response.content)
        return "ok"
    except requests.RequestException as e:
        print(f"FAILED {year_month}: {e}")
        return "failed"


def download_range(start: str, end: str, region: str = "SA1") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for ym in month_range(start, end):
        results[ym] = download_month(ym, region)
        print(f"{ym}: {results[ym]}")
        if results[ym] == "ok":
            time.sleep(1)

    statuses = list(results.values())
    print(f"\nok={statuses.count('ok')}  skipped={statuses.count('skipped')}  failed={statuses.count('failed')}")
    failed = [ym for ym, s in results.items() if s == "failed"]
    if failed:
        print("Failed months:", failed)

if __name__ == "__main__":
    download_range("2021-10", "2026-09")