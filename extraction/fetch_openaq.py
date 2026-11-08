from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://api.openaq.org/v3"
BUCHAREST = (44.4268, 26.1025)
RADIUS_M = 25_000
PAGE_LIMIT = 1000
PM_PARAMS = {"pm25", "pm10"}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "openaq"
STATIONS_CSV = RAW_DIR / "stations.csv"


class OpenAQClient:
    PER_MINUTE = 58
    PER_HOUR = 1950

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "Accept": "application/json"})
        self.calls: deque[float] = deque()

    def _throttle(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()
        if len(self.calls) >= self.PER_HOUR:
            wait = 3600 - (now - self.calls[0]) + 1
            print(f"hourly limit reached, sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
        last_minute = [t for t in self.calls if now - t < 60]
        if len(last_minute) >= self.PER_MINUTE:
            time.sleep(60 - (now - last_minute[0]) + 0.5)
        self.calls.append(time.monotonic())

    def get(self, path: str, params: dict | None = None, retries: int = 6) -> dict:
        url = f"{BASE_URL}{path}"
        for attempt in range(retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:
                wait = 2**attempt
                print(f"network error ({exc}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (408, 429, 500, 502, 503, 504):
                reset = r.headers.get("Retry-After") or r.headers.get("x-ratelimit-reset")
                wait = int(reset) if reset and reset.isdigit() else 2**attempt * 5
                print(f"  HTTP {r.status_code}; retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code == 401:
                sys.exit("HTTP 401: invalid or missing OPENAQ_API_KEY")
            r.raise_for_status()
        raise RuntimeError(f"giving up on {url} after {retries} attempts")

    def get_all(self, path: str, params: dict | None = None) -> list[dict]:
        params = dict(params or {}, limit=PAGE_LIMIT)
        results: list[dict] = []
        page = 1
        while True:
            data = self.get(path, {**params, "page": page})
            batch = data.get("results", [])
            results.extend(batch)
            if len(batch) < PAGE_LIMIT:
                return results
            page += 1


def make_client() -> OpenAQClient:
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not key:
        sys.exit("Set OPENAQ_API_KEY (environment variable or a local .env file).")
    return OpenAQClient(key)

STATION_FIELDS = [
    "location_id", "location_name", "locality", "provider", "owner", "is_monitor",
    "is_mobile", "lat", "lon", "sensor_id", "parameter", "units",
    "datetime_first", "datetime_last", "license",
]


def cmd_stations(args: argparse.Namespace) -> None:
    client = make_client()
    lat, lon = BUCHAREST
    locations = client.get_all(
        "/locations", {"coordinates": f"{lat},{lon}", "radius": RADIUS_M}
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "locations.json").write_text(json.dumps(locations, indent=1, ensure_ascii=False))

    rows = []
    for loc in locations:
        coords = loc.get("coordinates") or {}
        licenses = ", ".join(lic.get("name", "") for lic in loc.get("licenses") or [])
        for s in loc.get("sensors") or []:
            p = s.get("parameter") or {}
            rows.append({
                "location_id": loc.get("id"),
                "location_name": loc.get("name"),
                "locality": loc.get("locality"),
                "provider": (loc.get("provider") or {}).get("name"),
                "owner": (loc.get("owner") or {}).get("name"),
                "is_monitor": loc.get("isMonitor"),
                "is_mobile": loc.get("isMobile"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "sensor_id": s.get("id"),
                "parameter": p.get("name"),
                "units": p.get("units"),
                "datetime_first": ((loc.get("datetimeFirst") or {}).get("utc")),
                "datetime_last": ((loc.get("datetimeLast") or {}).get("utc")),
                "license": licenses,
            })
    with STATIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATION_FIELDS)
        w.writeheader()
        w.writerows(rows)

    pm = [r for r in rows if r["parameter"] in PM_PARAMS]
    monitors = {r["location_id"] for r in rows if r["is_monitor"]}
    print(f"{len(locations)} locations, {len(rows)} sensors -> {STATIONS_CSV}")
    print(f"reference monitors (isMonitor=True): {len(monitors)} locations")
    print(f"PM2.5/PM10 sensors: {len(pm)}\n")
    print(f"{'loc':>8} {'mon':>3} {'param':>5} {'first':>10} {'last':>10}  name / provider")
    for r in sorted(pm, key=lambda r: (not r["is_monitor"], r["location_name"] or "")):
        first = (r["datetime_first"] or "")[:10]
        last = (r["datetime_last"] or "")[:10]
        mon = "Y" if r["is_monitor"] else "-"
        print(f"{r['location_id']:>8} {mon:>3} {r['parameter']:>5} {first:>10} {last:>10}  "
              f"{r['location_name']} / {r['provider']}")

def load_sensors(args: argparse.Namespace) -> list[dict]:
    if not STATIONS_CSV.exists():
        sys.exit("Run `fetch_openaq.py stations` first.")
    with STATIONS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    params = set(args.params.split(","))
    sel = [r for r in rows if r["parameter"] in params]
    if args.monitors_only:
        sel = [r for r in sel if r["is_monitor"] == "True"]
    if args.sensors:
        wanted = set(args.sensors.split(","))
        sel = [r for r in sel if r["sensor_id"] in wanted]
    if args.active_since:
        sel = [r for r in sel if (r["datetime_last"] or "") >= args.active_since]
    if args.min_days:
        def span(r: dict) -> int:
            try:
                a = datetime.fromisoformat(r["datetime_first"].replace("Z", "+00:00"))
                b = datetime.fromisoformat(r["datetime_last"].replace("Z", "+00:00"))
                return (b - a).days
            except (AttributeError, ValueError):
                return 0
        sel = [r for r in sel if span(r) >= args.min_days]
    print(f"{len(sel)} sensors selected")
    return sel


def cmd_days(args: argparse.Namespace) -> None:
    client = make_client()
    out_dir = RAW_DIR / "days"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(load_sensors(args), 1):
        path = out_dir / f"{s['sensor_id']}.json"
        if path.exists() and not args.refresh:
            print(f"[{i}] {s['sensor_id']} cached")
            continue
        res = client.get_all(
            f"/sensors/{s['sensor_id']}/days",
            {"datetime_from": args.date_from, "datetime_to": args.date_to},
        )
        path.write_text(json.dumps({"sensor": s, "results": res}, ensure_ascii=False))
        print(f"[{i}] {s['sensor_id']} {s['parameter']} {s['location_name']}: {len(res)} days")


def cmd_hours(args: argparse.Namespace) -> None:
    client = make_client()
    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)
    months = []
    m = date(start.year, start.month, 1)
    while m < end:
        nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        months.append((m, nxt))
        m = nxt
    sensors = load_sensors(args)
    for i, s in enumerate(sensors, 1):
        out_dir = RAW_DIR / "hours" / s["sensor_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        fetched = 0
        first = (s.get("datetime_first") or "")[:7]
        last = (s.get("datetime_last") or "9999-12")[:7]
        for a, b in months:
            if first and f"{a:%Y-%m}" < first or f"{a:%Y-%m}" > last:
                continue
            path = out_dir / f"{a:%Y-%m}.json"
            final = b <= date.today() - timedelta(days=3)
            if path.exists() and final and not args.refresh:
                continue
            a2, b2 = max(start, a), min(end, b)
            try:
                res = client.get_all(
                    f"/sensors/{s['sensor_id']}/hours",
                    {"datetime_from": a2.isoformat(), "datetime_to": b2.isoformat()},
                )
            except (requests.HTTPError, RuntimeError) as exc:
                print(f"  WARNING {s['sensor_id']} {a:%Y-%m} skipped: {exc}", file=sys.stderr)
                continue
            path.write_text(json.dumps({"sensor": s, "results": res}, ensure_ascii=False))
            fetched += len(res)
        print(f"[{i}/{len(sensors)}] {s['sensor_id']} {s['parameter']} {s['location_name']}: "
              f"{fetched} new hours")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stations", help="list locations around Bucharest").set_defaults(func=cmd_stations)

    for name, func in (("days", cmd_days), ("hours", cmd_hours)):
        p = sub.add_parser(name, help=f"{name} per sensor")
        p.add_argument("--from", dest="date_from", default="2023-01-01")
        p.add_argument("--to", dest="date_to", default=date.today().isoformat())
        p.add_argument("--params", default="pm25,pm10")
        p.add_argument("--sensors", help="comma-separated sensor ids (default: all)")
        p.add_argument("--monitors-only", action="store_true", help="only isMonitor=True locations")
        p.add_argument("--min-days", type=int, default=0, help="minimum history length in days")
        p.add_argument("--active-since", help="only locations reporting after this date (YYYY-MM-DD)")
        p.add_argument("--refresh", action="store_true", help="ignore cache")
        p.set_defaults(func=func)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
