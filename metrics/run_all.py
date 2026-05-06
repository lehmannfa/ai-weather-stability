"""
Run all three stability metrics for one or several models.

Usage — single model
--------------------
    python metrics/run_all.py \\
        --model_name aurora \\
        --nc_path              /path/to/all_pred_aurora.nc \\
        --path_spectrum_pred   data/spectra/spectrum_aurora.nc \\
        --path_spectrum_target data/era5/spectrum_target.nc \\
        --path_spectrum_clim   data/era5/spectrum_clim.nc \\
        --output_dir           data/metrics

Usage — multiple models
-----------------------
    python metrics/run_all.py \\
        --model_names    aurora graphcast pangu \\
        --nc_paths       /path/aurora.nc /path/graphcast.nc /path/pangu.nc \\
        --spectra_paths  data/spectra/spectrum_aurora.nc data/spectra/spectrum_graphcast.nc data/spectra/spectrum_pangu.nc \\
        --path_spectrum_target data/era5/spectrum_target.nc \\
        --path_spectrum_clim   data/era5/spectrum_clim.nc \\
        --output_dir           data/metrics

When running multiple models, --model_names, --nc_paths, and --spectra_paths
must have the same number of entries.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from utils import DEFAULT_VARIABLES
from metrics.blowup_time      import main as run_blowup
from metrics.loss_seasonality import main as run_seasonality
from metrics.smallscale_ratio import main as run_smallscale


def run_one(model_name, nc_path, path_spectrum_pred,
            path_spectrum_target, path_spectrum_clim,
            output_dir, variables):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    run_blowup(
        nc_path    = nc_path,
        model_name = model_name,
        variables  = variables,
        output_dir = output_dir,
    )
    run_seasonality(
        model_name           = model_name,
        path_spectrum_pred   = path_spectrum_pred,
        path_spectrum_target = path_spectrum_target,
        path_spectrum_clim   = path_spectrum_clim,
        output_dir           = output_dir,
        variables            = variables,
    )
    run_smallscale(
        model_name           = model_name,
        path_spectrum_pred   = path_spectrum_pred,
        path_spectrum_target = path_spectrum_target,
        path_spectrum_clim   = path_spectrum_clim,
        output_dir           = output_dir,
        variables            = variables,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run all three stability metrics for one or several models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared ERA5 paths (always required)
    parser.add_argument("--path_spectrum_target", required=True,
                        help="Path to ERA5 target-period spectrum NetCDF.")
    parser.add_argument("--path_spectrum_clim",   required=True,
                        help="Path to ERA5 climatology spectrum NetCDF.")
    parser.add_argument("--output_dir", default="data/metrics")
    parser.add_argument("--variables",  nargs="+", default=DEFAULT_VARIABLES)

    # Single-model mode
    single = parser.add_argument_group("single model")
    single.add_argument("--model_name",         default=None)
    single.add_argument("--nc_path",            default=None,
                        help="Path to all_pred NetCDF for the model.")
    single.add_argument("--path_spectrum_pred", default=None,
                        help="Path to pre-computed model spectrum NetCDF.")

    # Multi-model mode
    multi = parser.add_argument_group("multiple models")
    multi.add_argument("--model_names",   nargs="+", default=None,
                       help="List of model names.")
    multi.add_argument("--nc_paths",      nargs="+", default=None,
                       help="List of all_pred NetCDF paths, one per model.")
    multi.add_argument("--spectra_paths", nargs="+", default=None,
                       help="List of model spectrum NetCDF paths, one per model.")

    args = parser.parse_args()

    # Build list of (model_name, nc_path, spectrum_pred_path) tuples
    if args.model_names is not None:
        if not (args.nc_paths and args.spectra_paths):
            parser.error("--model_names requires --nc_paths and --spectra_paths.")
        if not (len(args.model_names) == len(args.nc_paths) == len(args.spectra_paths)):
            parser.error("--model_names, --nc_paths, and --spectra_paths must have the same length.")
        models = list(zip(args.model_names, args.nc_paths, args.spectra_paths))
    elif args.model_name is not None:
        if not (args.nc_path and args.path_spectrum_pred):
            parser.error("--model_name requires --nc_path and --path_spectrum_pred.")
        models = [(args.model_name, args.nc_path, args.path_spectrum_pred)]
    else:
        parser.error("Provide either --model_name (single) or --model_names (multiple).")

    for model_name, nc_path, path_spectrum_pred in models:
        run_one(
            model_name           = model_name,
            nc_path              = nc_path,
            path_spectrum_pred   = path_spectrum_pred,
            path_spectrum_target = args.path_spectrum_target,
            path_spectrum_clim   = args.path_spectrum_clim,
            output_dir           = args.output_dir,
            variables            = args.variables,
        )

    print(f"\nDone. Results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
