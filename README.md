# Rollout Stability Metrics for AI Weather Models

Code for the paper "Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts"


Three metrics to characterise the long-term stability of autoregressive
AI weather model rollouts:

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

