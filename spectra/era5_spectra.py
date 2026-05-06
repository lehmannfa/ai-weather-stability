"""
Compute ERA5 Fourier spectra for two periods:

  target     : 2021-01-01 – 2023-01-01  (comparison against model rollouts)
  climatology: 1990-01-01 – 2019-01-01  (envelope for seasonality / small-scale metrics)

ERA5 data may be spread across multiple zarr/NetCDF files (e.g. one per year
or one per variable set). Pass all of them with --era5_paths; they are opened
and concatenated along the time dimension.

Usage
-----
    python spectra/era5_spectra.py \\
        --era5_paths /path/to/era5_2000.zarr /path/to/era5_2001.zarr ... \\
        --variables  2m_temperature geopotential_500 temperature_500 \\
        --output_dir data/era5

Output
------
    data/era5/spectrum_target.nc        (time × wavenumber)
    data/era5/spectrum_climatology.nc   (time × wavenumber, 2000–2004)

Both files contain one DataArray per variable.
The climatology file is used by the metrics scripts to compute the
(dayofyear, hour) mean and ±range envelope.
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

from rollout_stability import DEFAULT_VARIABLES
from spectra.compute_spectra import calculate_energy_spectrum

# Evaluation periods
T_TARGET_START = "2021-01-01T00"
T_TARGET_END   = "2023-01-01T00"
T_CLIM_START   = "1990-01-01T00"
T_CLIM_END     = "2019-01-01T00"


def compute_spectra_for_period(
        paths: list[str],
        t_vec: pd.DatetimeIndex,
        variables: list[str],
        desc: str,
    ) -> xr.Dataset:
    datasets = [xr.open_zarr(p, consolidated=False) if p.endswith('.zarr') or p[-3:] != '.nc'
                else xr.open_dataset(p) for p in paths]

    spectra = []
    for var in tqdm(variables, desc=desc):
        parts = var.rsplit('_', 1)
        try:
            if len(parts) == 2 and parts[1].isdigit():
                lev       = int(parts[1])
                var_short = parts[0]
                chunks = [
                    ds[var_short].sel(
                        time=np.intersect1d(ds.time.values, t_vec.values),
                        level=lev,
                    ).drop_vars('level')
                    for ds in datasets
                ]
            else:
                chunks = [
                    ds[var].sel(time=np.intersect1d(ds.time.values, t_vec.values))
                    for ds in datasets
                ]
            tmp = xr.concat(chunks, dim='time')
            tmp.name = var
        except (KeyError, ValueError) as exc:
            print(f'  {var}: not found ({exc}), skipping.')
            continue

        spect = calculate_energy_spectrum(tmp.compute())
        spect.name = var
        spect.attrs = {}
        spectra.append(spect)

    ds_out = xr.merge(spectra)
    ds_out.attrs = {
        'time_start': t_vec[0].strftime('%Y-%m-%dT%H'),
        'time_end':   t_vec[-1].strftime('%Y-%m-%dT%H'),
        'time_step':  f"{int((t_vec[1] - t_vec[0]) / pd.Timedelta('1h'))}h",
    }
    return ds_out


def main():
    parser = argparse.ArgumentParser(
        description="Compute ERA5 target and climatology Fourier spectra."
    )
    parser.add_argument(
        "--era5_paths", required=True, nargs="+",
        help=(
            "One or more paths to ERA5 zarr or NetCDF files. "
            "Files are opened and concatenated along the time dimension. "
        ),
    )
    parser.add_argument("--variables",  nargs="+", default=DEFAULT_VARIABLES,
                        help="Variables to process (ERA5 long names).")
    parser.add_argument("--output_dir", default="data/era5",
                        help="Directory where output NetCDF files are written.")
    parser.add_argument(
        "--target_start", default=T_TARGET_START,
        help=f"Start of the target period (default: {T_TARGET_START})."
    )
    parser.add_argument(
        "--target_end", default=T_TARGET_END,
        help=f"End of the target period (default: {T_TARGET_END})."
    )
    parser.add_argument(
        "--clim_start", default=T_CLIM_START,
        help=f"Start of the climatology period (default: {T_CLIM_START})."
    )
    parser.add_argument(
        "--clim_end", default=T_CLIM_END,
        help=f"End of the climatology period (default: {T_CLIM_END})."
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_target = pd.date_range(start=args.target_start, end=args.target_end, freq="6h")
    t_clim   = pd.date_range(start=args.clim_start,   end=args.clim_end,   freq="6h")

    print("\nComputing target-period spectra …")
    spectrum_target = compute_spectra_for_period(
        args.era5_paths, t_target, args.variables, desc="Target"
    )
    spectrum_target.to_netcdf(out_dir / "spectrum_target.nc")
    print(f"Saved → {out_dir}/spectrum_target.nc")

    print("\nComputing climatology-period spectra …")
    spectrum_clim = compute_spectra_for_period(
        args.era5_paths, t_clim, args.variables, desc="Climatology"
    )
    spectrum_clim.to_netcdf(out_dir / "spectrum_climatology.nc")
    print(f"Saved → {out_dir}/spectrum_climatology.nc")


if __name__ == "__main__":
    main()