from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ["Good", "Fair", "Moderate", "Poor", "Very poor", "Extremely poor"]
LEVELS_RO = ["Bună", "Acceptabilă", "Moderată", "Slabă", "Foarte slabă", "Extrem de slabă"]
POINTS_PER_BAND = 20

# upper bounds of categories 1..5 (category 6 = above the last bound)
BANDS = {
    "pm25": [5, 15, 50, 90, 140],
    "pm10": [15, 45, 120, 195, 270],
    "no2": [10, 25, 60, 100, 150],
    "o3": [60, 100, 120, 160, 180],
    "so2": [20, 40, 125, 190, 275],
}
POLLUTANTS = list(BANDS)


def sub_value(values: pd.Series, pollutant: str) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").clip(lower=0)
    bounds = np.array([0.0] + BANDS[pollutant], dtype=float)
    out = np.full(len(v), np.nan)
    arr = v.to_numpy(dtype=float)
    ok = ~np.isnan(arr)
    # band index 0..5 (upper bound inclusive)
    band = np.digitize(arr[ok], bounds[1:], right=True)
    lo = np.where(band < 5, bounds[np.minimum(band, 5)], bounds[5])
    hi = np.where(band < 5, bounds[np.minimum(band + 1, 5)], 2 * bounds[5] - bounds[4])
    out[ok] = POINTS_PER_BAND * band + POINTS_PER_BAND * (arr[ok] - lo) / (hi - lo)
    return pd.Series(out, index=values.index)


def category(value: pd.Series | np.ndarray) -> pd.Series:
    v = pd.Series(value, dtype=float)
    cat = np.clip(np.ceil(v / POINTS_PER_BAND), 1, 6)
    return cat.where(v.notna())


def overall(frame: pd.DataFrame, columns: dict[str, str], require: str = "pm25") -> tuple[pd.Series, pd.Series]:
    subs = pd.DataFrame({p: sub_value(frame[c], p) for p, c in columns.items() if c in frame})
    value = subs.max(axis=1, skipna=True)
    driver = subs.fillna(-1).idxmax(axis=1).where(value.notna())
    if require in subs:
        value = value.where(subs[require].notna())
        driver = driver.where(subs[require].notna())
    return value, driver
