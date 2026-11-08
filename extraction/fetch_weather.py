from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

LAT, LON = 44.4268, 26.1025
TZ = "Europe/Bucharest"
FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CAMS_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
    "boundary_layer_height",
]
CAMS_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide"]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "weather"


def fetch(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            print(f"  network error: {exc}")
        time.sleep(2**attempt * 3)
    raise RuntimeError(f"failed: {url} {params}")


def year_chunks(start: date, end: date):
    for year in range(start.year, end.year + 1):
        a = max(start, date(year, 1, 1))
        b = min(end, date(year, 12, 31))
        if a <= b:
            yield year, a, b


def download(kind: str, url: str, variables: list[str], start: date, end: date, refresh: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year, a, b in year_chunks(start, end):
        path = OUT_DIR / f"{kind}_{year}.json"
        complete = b < date.today() - timedelta(days=7) and b == date(year, 12, 31)
        if path.exists() and complete and not refresh:
            print(f"{path.name} cached")
            continue
        data = fetch(url, {
            "latitude": LAT, "longitude": LON, "timezone": TZ,
            "start_date": a.isoformat(), "end_date": b.isoformat(),
            "hourly": ",".join(variables),
        })
        path.write_text(json.dumps(data))
        hourly = data.get("hourly", {})
        n = len(hourly.get("time", []))
        nulls = {v: sum(x is None for x in hourly.get(v, [])) for v in variables}
        print(f"{path.name}: {n} hours, nulls: {nulls}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="date_from", default="2020-03-01")
    ap.add_argument("--to", dest="date_to", default=date.today().isoformat())
    ap.add_argument("--cams", action="store_true", help="also fetch CAMS air-quality forecasts")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.date_from), date.fromisoformat(args.date_to)
    download("historical_forecast", FORECAST_URL, WEATHER_VARS, start, end, args.refresh)
    if args.cams:
        download("cams", CAMS_URL, CAMS_VARS, start, end, args.refresh)


if __name__ == "__main__":
    main()
