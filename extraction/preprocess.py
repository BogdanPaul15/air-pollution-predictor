from __future__ import annotations

import argparse
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

try:
    from . import eaqi
except ImportError:
    import eaqi

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Bucharest"
MIN_HOURS = 18
MAX_VALID = 1000.0
MIN_CITY_STATIONS = 2              
MAX_GAP_HOURS = 3          
FFILL_DAYS = 3              


def _read_results(path: Path) -> tuple[dict, list[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("sensor", {}), obj.get("results", [])


def canonical_ids(openaq_dir: Path) -> dict[int, int]:
    s = pd.read_csv(openaq_dir / "stations.csv").drop_duplicates("location_id")
    s["canon"] = s.groupby(["provider", "location_name"])["location_id"].transform("min")
    return dict(zip(s["location_id"], s["canon"]))


def load_hourly(openaq_dir: Path) -> pd.DataFrame:
    canon = canonical_ids(openaq_dir)
    rows = []
    for path in sorted((openaq_dir / "hours").glob("*/*.json")):
        sensor, results = _read_results(path)
        if sensor.get("parameter") not in eaqi.POLLUTANTS:
            continue
        loc = int(sensor["location_id"])
        for r in results:
            ts = ((r.get("period") or {}).get("datetimeFrom") or {}).get("utc")
            rows.append((canon.get(loc, loc), sensor["parameter"], ts, r.get("value")))
    if not rows:
        return pd.DataFrame()
    h = pd.DataFrame(rows, columns=["location_id", "parameter", "ts", "value"])
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    h = h[(h["value"] >= 0) & (h["value"] <= MAX_VALID)]
    h = h.groupby(["location_id", "parameter", "ts"], as_index=False)["value"].mean()
    h = fill_short_gaps(h)
    h["ts"] = h["ts"].dt.tz_convert(TZ)
    h["date"] = h["ts"].dt.tz_localize(None).dt.normalize()
    h["evening"] = h["ts"].dt.hour >= 21

    g = h.groupby(["location_id", "parameter", "date"])
    daily = g["value"].agg(mean="mean", max="max", n_hours="count")
    daily["n_measured"] = g["measured"].sum()
    daily["evening"] = h[h["evening"]].groupby(["location_id", "parameter", "date"])["value"].mean()
    return daily.reset_index()


def fill_short_gaps(h: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (loc, par), g in h.groupby(["location_id", "parameter"], sort=False):
        s = g.set_index("ts")["value"].sort_index()
        full = pd.date_range(s.index.min(), s.index.max(), freq="h")
        v = s.reindex(full).to_numpy(dtype=float)
        day = full.tz_convert(TZ).tz_localize(None).normalize().to_numpy()
        nan = np.isnan(v)
        filled = v.copy()
        if nan.any():
            edges = np.diff(np.concatenate(([0], nan.astype(int), [0])))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            for a, b in zip(starts, ends):
                if b - a > MAX_GAP_HOURS or a == 0 or b == len(v):
                    continue
                if day[a - 1] != day[b]:
                    continue
                filled[a:b] = np.interp(np.arange(a, b), [a - 1, b], [v[a - 1], v[b]])
        keep = ~np.isnan(filled)
        out.append(pd.DataFrame({"location_id": loc, "parameter": par, "ts": full[keep],
                                 "value": filled[keep], "measured": ~nan[keep]}))
    return pd.concat(out, ignore_index=True)


def load_daily(openaq_dir: Path) -> pd.DataFrame:
    canon = canonical_ids(openaq_dir)
    rows = []
    for path in sorted((openaq_dir / "days").glob("*.json")):
        sensor, results = _read_results(path)
        for r in results:
            local = ((r.get("period") or {}).get("datetimeFrom") or {}).get("local")
            loc = int(sensor["location_id"])
            rows.append({
                "location_id": canon.get(loc, loc), "parameter": sensor["parameter"],
                "date": local[:10] if local else None, "mean": r.get("value"),
                "max": (r.get("summary") or {}).get("max"),
                "n_hours": (r.get("coverage") or {}).get("observedCount"), "evening": np.nan,
            })
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).dropna(subset=["date"])
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["mean"] >= 0) & (d["mean"] <= MAX_VALID)]
    d["n_measured"] = d["n_hours"]
    return d.groupby(["location_id", "parameter", "date"], as_index=False).agg(
        {"mean": "mean", "max": "max", "n_hours": "max", "n_measured": "max", "evening": "mean"})


def pollution_wide(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily[daily["parameter"].isin(eaqi.POLLUTANTS)].copy()
    invalid = daily["n_hours"] < MIN_HOURS
    daily.loc[invalid, ["mean", "max", "evening"]] = np.nan
    wide = daily.pivot_table(index=["location_id", "date"], columns="parameter",
                             values=[c for c in ("mean", "max", "evening", "n_hours", "n_measured") if c in daily],
                             aggfunc="first")
    wide.columns = [f"{p}_{stat}" for stat, p in wide.columns]
    return wide.reset_index()


def station_meta(openaq_dir: Path) -> pd.DataFrame:
    s = pd.read_csv(openaq_dir / "stations.csv")
    s["location_id"] = s["location_id"].map(canonical_ids(openaq_dir))
    meta = s.groupby("location_id").agg(
        station_name=("location_name", "first"), provider=("provider", "first"),
        is_monitor=("is_monitor", "first"), lat=("lat", "first"), lon=("lon", "first")).reset_index()
    meta["station_type"] = np.where(meta["is_monitor"].astype(str) == "True", "reference", "low-cost")
    return meta


MEAN_COLS = {p: f"{p}_mean" for p in eaqi.POLLUTANTS}


def stations_daily(openaq_dir: Path) -> pd.DataFrame:
    daily = load_hourly(openaq_dir)
    if daily.empty:
        daily = load_daily(openaq_dir)
    if daily.empty:
        raise SystemExit("No OpenAQ data found; run fetch_openaq.py hours (or days) first.")
    st = pollution_wide(daily).merge(station_meta(openaq_dir), on="location_id", how="left")
    st = st[st["station_type"] == "reference"].copy()
    for col in MEAN_COLS.values():
        if col not in st:
            st[col] = np.nan
    st["eaqi_value"], st["eaqi_driver"] = eaqi.overall(st, MEAN_COLS)
    st["eaqi_category"] = eaqi.category(st["eaqi_value"]).to_numpy()
    keep = ["location_id", "station_name", "lat", "lon", "date"]
    keep += [c for p in eaqi.POLLUTANTS for c in (f"{p}_mean", f"{p}_max", f"{p}_n_hours", f"{p}_n_measured") if c in st]
    keep += ["pm25_evening", "eaqi_value", "eaqi_driver", "eaqi_category"]
    return st[[c for c in keep if c in st]].sort_values(["date", "location_id"]).reset_index(drop=True)


def _load_hourly_json(weather_dir: Path, prefix: str) -> pd.DataFrame:
    frames = [pd.DataFrame(json.loads(p.read_text())["hourly"])
              for p in sorted(weather_dir.glob(f"{prefix}_*.json"))]
    if not frames:
        return pd.DataFrame()
    w = pd.concat(frames, ignore_index=True)
    w["time"] = pd.to_datetime(w["time"])
    w["date"] = w["time"].dt.normalize()
    return w.drop_duplicates("time")


def weather_daily(weather_dir: Path) -> pd.DataFrame:
    w = _load_hourly_json(weather_dir, "historical_forecast")
    rad = np.deg2rad(w["wind_direction_10m"])
    w["u"] = w["wind_speed_10m"] * np.sin(rad)
    w["v"] = w["wind_speed_10m"] * np.cos(rad)
    g = w.groupby("date")
    d = pd.DataFrame({
        "blh_mean": g["boundary_layer_height"].mean(),
        "blh_min": g["boundary_layer_height"].min(),
        "wind_speed": g["wind_speed_10m"].mean(),
        "temp": g["temperature_2m"].mean(),
        "temp_min": g["temperature_2m"].min(),
        "humidity": g["relative_humidity_2m"].mean(),
        "precip": g["precipitation"].sum(),
        "pressure": g["surface_pressure"].mean(),
    })
    ang = np.arctan2(g["u"].mean(), g["v"].mean())
    d["wind_dir_sin"], d["wind_dir_cos"] = np.sin(ang), np.cos(ang)
    return d.add_prefix("fc_").reset_index()


def cams_daily(weather_dir: Path) -> pd.DataFrame:
    c = _load_hourly_json(weather_dir, "cams")
    if c.empty:
        return pd.DataFrame(columns=["date", "bench_cams_eaqi"])
    src = {"pm25": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide", "o3": "ozone", "so2": "sulphur_dioxide"}
    d = c.groupby("date")[[v for v in src.values() if v in c]].mean()
    d["bench_cams_eaqi"], _ = eaqi.overall(d, {p: v for p, v in src.items() if v in d})
    return d[["bench_cams_eaqi"]].reset_index()


def heating_season(d: pd.Series) -> pd.Series:
    md = d.dt.month * 100 + d.dt.day
    return ((md >= 1015) | (md <= 415)).astype(int)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    target = df["date"] + pd.Timedelta(days=1)
    ro = holidays.Romania(years=range(target.dt.year.min(), target.dt.year.max() + 1))
    df["target_date"] = target
    df["dow"] = target.dt.day_of_week
    df["is_weekend_or_holiday"] = ((target.dt.day_of_week >= 5) | target.dt.date.map(lambda x: x in ro)).astype(int)
    df["month"] = target.dt.month
    df["heating_season"] = heating_season(target)
    return df


FEATURES = [
    "eaqi_today", "eaqi_yesterday", "eaqi_mean3", "eaqi_mean7",
    "pm25_today", "pm25_yesterday", "pm25_mean3", "pm25_mean7", "pm25_max_today", "pm25_evening_today",
    "pm10_today", "no2_today", "o3_today", "so2_today", "ratio_pm25_pm10",
    "fc_blh_mean", "fc_blh_min", "fc_wind_speed", "fc_wind_dir_sin", "fc_wind_dir_cos",
    "fc_temp", "fc_temp_min", "fc_humidity", "fc_precip", "fc_pressure",
    "dow", "is_weekend_or_holiday", "month", "heating_season",
]
TARGET = "y_eaqi"
INFO = ["n_stations_pm25", "today_filled", "eaqi_driver_today", "y_eaqi_category", "y_eaqi_driver",
        "y_strict", "bench_cams_eaqi"]


def build_city(st: pd.DataFrame, weather: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    g = st.groupby("date")
    city = pd.DataFrame({p: g[f"{p}_mean"].median() for p in eaqi.POLLUTANTS})
    city["pm25_max"] = g["pm25_max"].median()
    city["pm25_evening"] = g["pm25_evening"].median()
    city["n_stations_pm25"] = g["pm25_mean"].count()
    strict = st["pm25_mean"].notna() & (st["pm25_n_measured"] >= MIN_HOURS)
    city["n_stations_pm25_strict"] = strict.groupby(st["date"]).sum()
    city = city.asfreq("D")
    city.index.name = "date"
    counts = ["n_stations_pm25", "n_stations_pm25_strict"]
    too_few = city["n_stations_pm25"].fillna(0) < MIN_CITY_STATIONS
    city.loc[too_few, [c for c in city.columns if c not in counts]] = np.nan
    city["strict_day"] = (city["n_stations_pm25_strict"].fillna(0) >= MIN_CITY_STATIONS).astype(int)

    e_raw, driver_raw = eaqi.overall(city, {p: p for p in eaqi.POLLUTANTS})

    city[TARGET] = e_raw.shift(-1)
    city["y_eaqi_driver"] = driver_raw.shift(-1)
    city["y_eaqi_category"] = eaqi.category(city[TARGET]).to_numpy()
    city["y_strict"] = city["strict_day"].shift(-1)

    city["today_filled"] = (e_raw.isna() & e_raw.ffill(limit=FFILL_DAYS).notna()).astype(int)
    ff = lambda x: x.ffill(limit=FFILL_DAYS)
    e, pm = ff(e_raw), ff(city["pm25"])
    city["eaqi_today"], city["eaqi_driver_today"] = e, driver_raw.ffill(limit=FFILL_DAYS)
    city["eaqi_yesterday"] = e.shift(1)
    city["eaqi_mean3"] = e_raw.rolling(3, min_periods=2).mean()
    city["eaqi_mean7"] = e_raw.rolling(7, min_periods=4).mean()
    city["pm25_today"], city["pm25_yesterday"] = pm, pm.shift(1)
    city["pm25_mean3"] = city["pm25"].rolling(3, min_periods=2).mean()
    city["pm25_mean7"] = city["pm25"].rolling(7, min_periods=4).mean()
    city["pm25_max_today"], city["pm25_evening_today"] = ff(city["pm25_max"]), ff(city["pm25_evening"])
    for p in ("pm10", "no2", "o3", "so2"):
        city[f"{p}_today"] = ff(city[p])
    city["ratio_pm25_pm10"] = (city["pm25_today"] / city["pm10_today"]).where(city["pm10_today"] > 0)

    city = add_calendar(city.reset_index())
    city = city.merge(weather.rename(columns={"date": "target_date"}), on="target_date", how="left")
    city = city.merge(cams.rename(columns={"date": "target_date"}), on="target_date", how="left")
    city = city[city["eaqi_today"].notna() & city[TARGET].notna()]
    cols = ["date", "target_date"] + FEATURES + [TARGET] + INFO
    return city[cols].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openaq-dir", type=Path, default=ROOT / "data" / "raw" / "openaq")
    ap.add_argument("--weather-dir", type=Path, default=ROOT / "data" / "raw" / "weather")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    st = stations_daily(args.openaq_dir)
    city = build_city(st, weather_daily(args.weather_dir), cams_daily(args.weather_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st.to_csv(args.out_dir / "stations_daily.csv", index=False)
    city.to_csv(args.out_dir / "dataset.csv", index=False)
    print(f"\n-> {args.out_dir / 'dataset.csv'}\n-> {args.out_dir / 'stations_daily.csv'}")


if __name__ == "__main__":
    main()
