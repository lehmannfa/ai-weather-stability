"""
Compute the zonal Fourier power spectrum for a model rollout.

Reads a pre-saved all_pred NetCDF file (dimensions: time × latitude × longitude,
one DataArray per variable, level dimension already dropped — e.g. "temperature_500"
is a surface-like field at that level).

Usage
-----
    python spectra/compute_spectra.py \\
        --nc_path    /path/to/all_pred_aurora.nc \\
        --model_name aurora \\
        --output_dir data/spectra

Output
------
    data/spectra/<model_name>.nc

    Dataset with one DataArray per variable,
    dimensions (time, wavenumber), coordinate wavelength (km).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm
from xrscipy.fft import rfft

from utils import DEFAULT_VARIABLES

EARTH_RADIUS_KM = 6371.0


def calculate_energy_spectrum(da: xr.DataArray) -> xr.DataArray:
    """
    Latitude-weighted zonal power spectrum of a (time, latitude, longitude) field.

    Parameters
    ----------
    da : xr.DataArray
        Must have dimensions (time, latitude, longitude).

    Returns
    -------
    xr.DataArray
        Dimensions (time, wavenumber).
        Non-index coordinate `wavelength` (km) is attached.
    """
    da_fft = rfft(da, "longitude").rename({"longitude": "wavenumber"})

    n_lon = da.longitude.size
    dx = 2 * np.pi * EARTH_RADIUS_KM / n_lon
    da_fft["wavenumber"] = np.fft.rfftfreq(n_lon, d=dx)

    # Drop DC component (wavenumber = 0)
    da_fft = da_fft.isel(wavenumber=range(1, da_fft.wavenumber.size))
    da_fft["wavelength"] = 1.0 / da_fft["wavenumber"]

    da_power = np.abs(da_fft) ** 2

    cosw = np.cos(np.deg2rad(da.latitude))
    cosw = np.clip(cosw, 1e-6, None)
    return da_power.weighted(cosw).mean(dim="latitude")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Fourier power spectra from an all_pred NetCDF file."
    )
    parser.add_argument(
        "--nc_path", required=True,
        help="Path to the pred NetCDF file (time × lat × lon, one var per DataArray).",
    )
    parser.add_argument("--model_name",  required=True,
                        help="Short model identifier used for the output filename.")
    parser.add_argument("--variables",   nargs="+", default=DEFAULT_VARIABLES,
                        help="Variables to process. Must match DataArray names in the file.")
    parser.add_argument("--output_dir",  default="data/spectra")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}.nc"

    print(f"Opening {args.nc_path} …")
    ds = xr.open_dataset(args.nc_path)
    print(ds)

    spectra = []
    for var in tqdm(args.variables, desc="Spectra"):
        if var not in ds:
            print(f"  {var}: not found in dataset, skipping.")
            continue
        spect = calculate_energy_spectrum(ds[var].compute())
        spect.name = var
        spect.attrs = {}
        spectra.append(spect)

    ds_out = xr.merge(spectra)
    ds_out.attrs = {
        "model":      args.model_name,
        "source":     args.nc_path,
        "time_start": str(ds_out.time.values[0]),
        "time_end":   str(ds_out.time.values[-1]),
    }
    ds_out.to_netcdf(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
