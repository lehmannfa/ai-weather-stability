"""
Metric 1 — Time of blow-up.

Detects exponential growth in the spatial min/max time series of each
variable. Uses a rolling log-linear regression; growth is declared when
R² exceeds a threshold and the explosive behaviour is sustained.

Operates on the physical zarr rollout (not the spectra).

Usage
-----
    python metrics/blowup_time.py \\
        --zarr_path  /path/to/rollout.zarr \\
        --model_name aurora \\
        --output_dir data/metrics

Output
------
    data/metrics/<model_name>/blowup.csv

    Columns: variable, blowup_days
    NaN means no blow-up was detected within the rollout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from utils import DEFAULT_VARIABLES, VARS_LATEX


def detect_exponential_onset(
    y: np.ndarray,
    window_size: int = 120,
    r2_threshold: float = 0.90,
    slope_min: float = 0.001,
) -> tuple[np.ndarray, int | None]:
    """
    Detect when exponential growth begins in a 1-D time series.

    Returns (log_y, onset_index) where onset_index is the index into the
    original (NaN-removed) array, or None if no growth was detected.
    """
    y = np.array(y, dtype=float)

    if np.nanmin(y) < 0:
        y = y - 1.1 * np.nanmin(y)

    valid = ~np.isnan(y)
    orig_idx = np.where(valid)[0]
    y = y[valid]

    log_y = np.log(np.abs(y) + 1e-10)

    for i in range(window_size, len(y)):
        window = log_y[i - window_size : i]
        X = np.arange(window_size).reshape(-1, 1)
        reg = LinearRegression().fit(X, window)
        r2 = reg.score(X, window)
        if np.abs(reg.coef_[0]) > slope_min and r2 > r2_threshold:
            return log_y, orig_idx[i - window_size]

    return log_y, None


def compute_blowup_days(da: xr.DataArray, window_size: int = 120) -> float | None:
    """
    Return the blow-up time in days for a single variable DataArray,
    or None if no blow-up is detected.
    """
    smooth = dict(time=16, center=True)
    p_min = da.min(dim=["latitude", "longitude"]).rolling(**smooth).mean()
    p_max = da.max(dim=["latitude", "longitude"]).rolling(**smooth).mean()

    _, it_min = detect_exponential_onset(p_min.values, window_size=window_size)
    _, it_max = detect_exponential_onset(p_max.values, window_size=window_size)

    candidates = [it for it in [it_min, it_max] if it is not None]
    if not candidates:
        return None
    return min(candidates) / 4.0  # 6-hourly steps → days


def main(nc_path=None, model_name=None, variables=None, output_dir=None):
    if nc_path is None:  # called from CLI
        parser = argparse.ArgumentParser(
            description="Compute blow-up time metric for a model rollout."
        )
        parser.add_argument(
            "--nc_path", required=True,
            help="Path to the pred NetCDF file for that model (time × lat × lon, level already dropped).",
        )
        parser.add_argument("--model_name",  required=True)
        parser.add_argument("--variables",   nargs="+", default=DEFAULT_VARIABLES)
        parser.add_argument("--output_dir",  default="data/metrics")
        args = parser.parse_args()
        nc_path    = args.nc_path
        model_name = args.model_name
        variables  = args.variables
        output_dir = args.output_dir

    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {nc_path} …")
    ds = xr.open_dataset(nc_path)
    n_days_total = ds.time.size // 4

    rows = []
    for var in tqdm(variables, desc="Blow-up"):
        if var not in ds:
            print(f"  {var}: not found in dataset, skipping.")
            rows.append({"variable": var, "blowup_days": np.nan})
            continue
        da = ds[var].compute()

        days = compute_blowup_days(da)
        label = f"{days:.1f}" if days is not None else f"> {n_days_total}"
        print(f"  {VARS_LATEX.get(var, var)}: {label} days")
        rows.append({"variable": var, "blowup_days": days if days is not None else np.nan})

    out_path = out_dir / "blowup.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
