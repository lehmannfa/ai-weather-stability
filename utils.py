"""
Shared constants and helpers used across spectra and metrics scripts.
"""

# ── Spatial scale bands (km) ─────────────────────────────────────────────────
SMALL_BANDS        = [10,    250]
MEDIUM_BANDS       = [250,  1000]
MEDIUM_LARGE_BANDS = [1000, 5000]
LARGE_BANDS        = [5000, 50000]

# ── Variable display names ────────────────────────────────────────────────────
VARS_LATEX = {
    "2m_temperature":           "T2m",
    "10m_u_component_of_wind":  "U10m",
    "10m_v_component_of_wind":  "V10m",
    "mean_sea_level_pressure":  "MSLP",
    "temperature_850":          "T850",
    "u_component_of_wind_850":  "U850",
    "v_component_of_wind_850":  "V850",
    "geopotential_850":         "Z850",
    "temperature_500":          "T500",
    "specific_humidity_500":    "Q500",
    "geopotential_500":         "Z500",
    "temperature_300":          "T300",
    "u_component_of_wind_300":  "U300",
    "v_component_of_wind_300":  "V300",
    "temperature_100":          "T100",
    "specific_humidity_100":    "Q100",
}

DEFAULT_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_500",
    "temperature_500",
    "specific_humidity_500",
    "u_component_of_wind_300",
    "temperature_100",
    "specific_humidity_100",
]


def format_ratio(x: float) -> str:
    """Human-readable formatting for small-scale ratio values."""
    if x >= 100:
        return f"{x:.0e}".replace("+0", "")
    elif x >= 10:
        return f"{x:.0f}"
    elif x < 0.05:
        return f"{x:.2f}"
    else:
        return f"{x:.1f}"
