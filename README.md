# Rollout Stability Metrics for AI Weather Models

Code for the paper "Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts"


This repository proposes three metrics to characterise the long-term stability
of autoregressive AI weather model rollouts over multi-year timescales.

| Metric | What it measures | Input |
|---|---|---|
| **Time of blow-up** | Onset of exponential growth in the physical field | zarr rollout |
| **Loss of seasonality** | When large-scale spectral power exits the climatological envelope | pre-computed spectra |
| **Ratio of small scales** | Small-scale power gained or lost over the rollout | pre-computed spectra |

---

## Structure

```
rollout-stability/
├── utils.py       # Shared constants and helpers
├── spectra/
│   ├── compute_spectra.py   # Model rollout → spectrum NetCDF
│   └── era5_spectra.py      # ERA5 target + climatology spectra
├── metrics/
│   ├── blowup_time.py
│   ├── loss_seasonality.py
│   ├── smallscale_ratio.py
│   └── run_all.py           # Runs all three in order
├── notebooks/
│   ├── 01_spectra.ipynb
│   ├── 02_blowup.ipynb
│   ├── 03_seasonality.ipynb
│   └── 04_smallscale.ipynb
├── data/                    # git-ignored — populate via your own inference pipeline
    ├── spectra/             # <model_name>.nc
    ├── metrics/             # <model_name>/{blowup,loss_seasonality,smallscale_ratio}.csv
    └── era5/                # spectrum_target.nc, spectrum_climatology.nc

```

---

## Dependencies

```bash
pip install -r requirements.txt
```

---

## Computing the metrics

### Step 1 — ERA5 reference spectra (once)

Compute the Fourier spectra for the ERA5 forecast period (2021–2024) and
climatological period (2000–2004). Both are written to a single output
directory and are shared across all models.

```bash
python spectra/era5_spectra.py \
    --era5_path  /path/to/era5.zarr \
    --output_dir data/era5
```

The ERA5 dataset must cover both periods and have dimensions
`(time, level, latitude, longitude)` with ERA5 long variable names.

### Step 2 — Model spectrum

Compute the Fourier spectrum for a model rollout saved as a NetCDF file.
The rollout must have dimensions `(time, latitude, longitude)` with one
DataArray per variable. Pressure-level variables must have the level
encoded in the name (e.g. `temperature_500`); there is no `level` dimension.

```bash
python spectra/compute_spectra.py \
    --nc_path    /path/to/all_pred_aurora.nc \
    --model_name Aurora \
    --output_dir data/spectra
```

### Step 3 — All three metrics

```bash
python metrics/run_all.py \
    --model_name           Aurora \
    --nc_path              /path/to/all_pred_aurora.nc \
    --path_spectrum_pred   data/spectra/spectrum_Aurora.nc \
    --path_spectrum_target data/era5/spectrum_target.nc \
    --path_spectrum_clim   data/era5/spectrum_climatology.nc \
    --output_dir           outputs
```

Results are written to `outputs/Aurora/`.

To run several models in one call:

```bash
python metrics/run_all.py \
    --model_names    Aurora Pangu GraphCast \
    --nc_paths       /path/aurora.nc /path/pangu.nc /path/graphcast.nc \
    --spectra_paths  data/spectra/spectrum_Aurora.nc data/spectra/spectrum_Pangu.nc data/spectra/spectrum_GraphCast.nc \
    --path_spectrum_target data/era5/spectrum_target.nc \
    --path_spectrum_clim   data/era5/spectrum_climatology.nc \
    --output_dir           outputs
```

By default the nine variables listed in `utils.py` are evaluated. Pass
`--variables` to any script to use a different subset.

---

## Notebooks

All notebooks read from `data/spectra/` and `data/era5/` for spectral data,
and directly from model rollout NetCDF files for physical-field plots.
Update the path variables in the parameters cell of each notebook before
running.

**`01_spectra_loglog.ipynb`** — Log-log plot of the zonal power spectrum at a
chosen time step, comparing the model against ERA5. Can either load a
pre-computed spectrum from `data/spectra/` or compute it on the fly from a
raw rollout file.

**`02_spectra_bands.ipynb`** — Three-panel time series of band-averaged
spectral energy (large / medium / small scales) over the full rollout, with
the ERA5 climatological envelope shown as grey shading. This is the main
diagnostic for the loss-of-seasonality and small-scale ratio metrics.

**`03_regional_timeseries.ipynb`** — Spatial min / mean / max of the physical
field over predefined regions (global, tropics, Europe, North America, etc.),
plotted as a function of rollout time and compared against ERA5. Useful for
identifying where and when a rollout starts to diverge physically.

**`04_blowup_diagnostic.ipynb`** — Visualises the blow-up detection algorithm:
plots the rolling spatial min and max time series and highlights the window
where exponential growth was detected by the log-linear regression.

---

## Acknowledgments

The scientific content of this repository — the metric definitions, algorithmic
choices, and experimental results — was developed entirely by the authors.
The code was refactored and documented with the assistance of
[Claude](https://claude.ai) (Anthropic) to improve readability and
reproducibility. All scientific decisions, parameter choices, and
interpretations remain the authors' own.

---

## License

See `LICENSE`.

