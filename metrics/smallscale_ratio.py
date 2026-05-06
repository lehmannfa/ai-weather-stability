"""
Metric 3 — Ratio of small scales.

Quantifies the evolution of small-scale spectral power relative to:
  (a) the climatological envelope at the end of the rollout  → ratio_ref
  (b) the rollout's own initial small-scale power            → ratio_init

ratio_ref > 1  →  excess small-scale noise (spectral blow-up)
ratio_ref < 1  →  over-smoothed / damped small scales
ratio_ref = 1  →  within the climatological envelope

The spectrum is restricted to the pre-blow-up period (read from
blowup.csv if present) so that the ratio is not contaminated by
the exponential growth phase.

Operates on pre-computed model spectra (NetCDF) and the ERA5 spectra.

Usage
-----
    python metrics/smallscale_ratio.py \\
        --model_name aurora \\
        --spectra_dir data/spectra \\
        --era5_dir    data/era5 \\
        --output_dir  data/metrics

Output
------
    data/metrics/<model_name>/smallscale_ratio.csv

    Columns: variable, ratio_ref, ratio_init
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

from utils import DEFAULT_VARIABLES, VARS_LATEX, SMALL_BANDS, format_ratio
from metrics.loss_seasonality import _build_envelope


def ratio_small_scales(
    spectrum_pred: xr.Dataset,
    spectrum_target: xr.Dataset,
    var: str,
    spectrum_clim: xr.Dataset,
    small_bands: list[int] = SMALL_BANDS,
    threshold_init: int = 4 * 2,
    threshold_end:  int = 4 * 30,
) -> tuple[float, float]:
    """
    Compute the small-scale spectral ratio relative to climatology and rollout start.

    Parameters
    ----------
    spectrum_pred : already truncated to the pre-blowup period if applicable.
    threshold_init : steps at the start used to estimate initial small-scale power.
    threshold_end  : steps at the end used to estimate final small-scale power.

    Returns
    -------
    ratio_ref : float
        fft_end / climatological bound (exactly 1.0 if within the envelope).
    ratio_init : float
        fft_end / fft_init.
    """
    wn_lo = 1.0 / small_bands[1]
    wn_hi = 1.0 / small_bands[0]

    min_env, max_env = _build_envelope(
        spectrum_clim, spectrum_target, var, small_bands
    )

    pred = (
        spectrum_pred
        .sel(wavenumber=slice(wn_lo, wn_hi))
        .mean(dim="wavenumber")[var]
    )
    pred = pred.where(pred != 0, drop=True)

    fft_init = float(pred.values[:threshold_init].mean())
    fft_end  = float(pred.values[-threshold_end:].mean())

    min_ref = float(min_env.sel(time=pred.time[-threshold_end:]).mean())
    max_ref = float(max_env.sel(time=pred.time[-threshold_end:]).mean())

    if fft_end < min_ref:
        ratio_ref = fft_end / min_ref
    elif fft_end > max_ref:
        ratio_ref = fft_end / max_ref
    else:
        ratio_ref = 1.0

    ratio_init = fft_end / fft_init if fft_init != 0 else np.nan
    return ratio_ref, ratio_init


def main(model_name=None, path_spectrum_pred=None, path_spectrum_target=None, 
         path_spectrum_clim=None, output_dir=None, variables=None):
    if model_name is None:
        parser = argparse.ArgumentParser(
            description="Compute small-scale spectral ratio metric."
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

    # Load blow-up times to restrict the spectrum (optional)
    blowup_path = out_dir / "blowup.csv"
    blowup_dict: dict[str, float] = {}
    if blowup_path.exists():
        df_blowup = pd.read_csv(blowup_path).set_index("variable")
        blowup_dict = df_blowup["blowup_days"].to_dict()
        print(f"Loaded blow-up times from {blowup_path}")
    else:
        print("No blowup.csv found — spectra will not be truncated.")

    rows = []
    for var in tqdm(variables, desc="Small-scale ratio"):
        if var not in spectrum_pred or var not in spectrum_target or var not in spectrum_clim:
            print(f"  {var}: missing from one of the spectrum files, skipping.")
            rows.append({"variable": var, "ratio_ref": np.nan, "ratio_init": np.nan})
            continue

        # Restrict to pre-blowup period
        pred = spectrum_pred
        blowup_days = blowup_dict.get(var, np.nan)
        if not np.isnan(blowup_days):
            max_date = pred.time.values[0] + np.timedelta64(int(blowup_days) * 24, "h")
            pred = pred.sel(time=(pred.time <= max_date))
            print(f"  {var}: restricted to {blowup_days:.0f} days pre-blowup")

        ratio_ref, ratio_init = ratio_small_scales(
            pred,
            spectrum_target.sel(time=pred.time),
            var,
            spectrum_clim,
        )
        print(
            f"  {VARS_LATEX.get(var, var)}: "
            f"ratio_ref={format_ratio(ratio_ref)}  ratio_init={format_ratio(ratio_init)}"
        )
        rows.append({"variable": var, "ratio_ref": ratio_ref, "ratio_init": ratio_init})

    out_path = out_dir / "smallscale_ratio.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
