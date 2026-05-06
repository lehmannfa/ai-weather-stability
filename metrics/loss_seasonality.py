"""
Metric 2 — Loss of seasonality.

Measures how long a rollout's large-scale Fourier spectrum remains within
the climatological envelope (mean ± range over dayofyear × hour groups).
Failure is declared when the spectrum exits the envelope for more than
`threshold` consecutive 6-hourly steps (default 180 = 45 days).

Operates on the pre-computed model and ERA5 spectra (NetCDF).

Usage
-----
    python metrics/loss_seasonality.py \\
        --model_name aurora \\
        --spectra_dir data/spectra \\
        --era5_dir    data/era5 \\
        --output_dir  data/metrics

Output
------
    data/metrics/<model_name>/loss_seasonality.csv

    Columns: variable, seasonality_days
    NaN means seasonality was preserved for the full rollout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from utils import DEFAULT_VARIABLES, VARS_LATEX, LARGE_BANDS


def _build_envelope(
    spectrum_clim: xr.Dataset,
    spectrum_target: xr.Dataset,
    var: str,
    bands: list[int],
    smoothing: int = 16,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the climatological min/max envelope aligned to the target time axis.

    The climatology is smoothed with a rolling mean, then grouped by
    (dayofyear, hour) to get a mean and range. These are looked up using the
    dayofyear and hour of each target time step.
    """
    wn_lo = 1.0 / bands[1]
    wn_hi = 1.0 / bands[0]

    clim_smooth = spectrum_clim.rolling(time=smoothing, center=True).mean()
    clim_mean   = clim_smooth.groupby(["time.dayofyear", "time.hour"]).mean()
    clim_range  = (
        clim_smooth.groupby(["time.dayofyear", "time.hour"]).max()
        - clim_smooth.groupby(["time.dayofyear", "time.hour"]).min()
    )

    doy  = spectrum_target.time.dt.dayofyear
    hour = spectrum_target.time.dt.hour

    min_env = (
        (clim_mean - clim_range)
        .sel(dayofyear=doy, hour=hour, wavenumber=slice(wn_lo, wn_hi))
        .mean(dim="wavenumber")[var]
    )
    max_env = (
        (clim_mean + clim_range)
        .sel(dayofyear=doy, hour=hour, wavenumber=slice(wn_lo, wn_hi))
        .mean(dim="wavenumber")[var]
    )
    return min_env, max_env


def duration_preserved_seasonality(
    spectrum_pred: xr.Dataset,
    spectrum_target: xr.Dataset,
    var: str,
    spectrum_clim: xr.Dataset,
    large_bands: list[int] = LARGE_BANDS,
    threshold: int = 4 * 45,
) -> float | None:
    """
    Return the number of days seasonality is preserved, or None if it holds
    for the full rollout.
    """
    wn_lo = 1.0 / large_bands[1]
    wn_hi = 1.0 / large_bands[0]

    min_env, max_env = _build_envelope(
        spectrum_clim, spectrum_target, var, large_bands
    )

    pred = (
        spectrum_pred
        .sel(wavenumber=slice(wn_lo, wn_hi))
        .mean(dim="wavenumber")[var]
    )

    out_of_bounds = (pred > max_env) | (pred < min_env)
    cumsum = out_of_bounds.cumsum()
    exceeded = (cumsum > threshold).values
    onset = int(np.argmax(exceeded))

    if onset == 0 and not exceeded[0]:
        return None  # threshold never reached

    return onset / 4.0  # 6-hourly steps → days


def main(model_name=None, path_spectrum_pred=None, path_spectrum_target=None, 
         path_spectrum_clim=None, output_dir=None, variables=None):
    if model_name is None:  # called from CLI
        parser = argparse.ArgumentParser(
            description="Compute loss-of-seasonality metric."
        )
        parser.add_argument("--model_name",  required=True)
        parser.add_argument("--path_spectrum_pred", default="data/spectra/spectrum_aurora.nc")
        parser.add_argument("--path_spectrum_target",    default="data/era5/spectrum_target.nc")
        parser.add_argument("--path_spectrum_clim",    default="data/era5/spectrum_clim.nc")
        parser.add_argument("--output_dir",  default="data/metrics")
        parser.add_argument("--variables",   nargs="+", default=DEFAULT_VARIABLES)
        args = parser.parse_args()

        model_name = args.model_name
        path_spectrum_pred = args.path_spectrum_pred
        path_spectrum_target = args.path_spectrum_target
        path_spectrum_clim = args.path_spectrum_clim
        output_dir = args.output_dir
        variables = args.variables

    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ERA5 spectra …")
    spectrum_target = xr.open_dataset(Path(path_spectrum_target))
    spectrum_clim   = xr.open_dataset(Path(path_spectrum_clim))

    print(f"Loading model spectrum for {model_name} …")
    spectrum_pred = xr.open_dataset(Path(path_spectrum_pred))

    rows = []
    for var in tqdm(variables, desc="Seasonality"):
        if var not in spectrum_pred or var not in spectrum_target or var not in spectrum_clim:
            print(f"  {var}: missing from one of the spectrum files, skipping.")
            rows.append({"variable": var, "seasonality_days": np.nan})
            continue

        days = duration_preserved_seasonality(
            spectrum_pred, spectrum_target, var, spectrum_clim,
            threshold=45*4,
        )
        n_days_total = spectrum_pred.time.size // 4
        label = f"{int(days)}" if days is not None else f"> {n_days_total}"
        print(f"  {VARS_LATEX.get(var, var)}: {label} days")
        rows.append({"variable": var, "seasonality_days": days if days is not None else np.nan})

    out_path = out_dir / "loss_seasonality.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
