"""Shared constants, helpers, and output-header builders for the analysis pipeline.

Imported by every script in this directory so that predictor/outcome names,
the standardisation convention, the file-version header format, and the
figure palette are defined in exactly one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# Inferential variable sets — match the manuscript Methods exactly.
PREDICTORS = ["Rel_Peak_Ext_Torque", "HQ_Ratio", "RSI_DJ30"]
OUTCOMES = ["Sprint_20m_Best", "CODD_Rel"]

# Output directory (created on import for any script that writes results).
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Figure colour palette — kept here so reliability/descriptives/figures all
# use the same visual language without hard-coding hexes.
PALETTE = {
    "primary": "#1b4d72",     # deep blue — primary predictor / sprint
    "secondary": "#c0392b",   # red — CODD / paradox accent
    "accent": "#27ae60",      # green — H:Q ratio
    "neutral": "#5d6d7e",     # slate — supporting elements
    "highlight": "#f39c12",   # amber — descriptive supplementary
    "background": "#fafafa",
    "grid": "#bdc3c7",
}


def standardize(s: pd.Series) -> pd.Series:
    """z-score with population SD (ddof=0) — matches PyMC default conventions."""
    return (s - s.mean()) / s.std(ddof=0)


def build_file_header(extras: dict[str, str] | None = None) -> str:
    """Return a deterministic header for results/*.txt files.

    Contains language + library versions + seed but no wall-clock timestamp,
    so re-running a script produces byte-identical output. Optional `extras`
    is merged after the canonical keys.
    """
    import scipy
    import statsmodels

    base = {
        "Python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "Seed": str(SEED),
    }
    if extras:
        base.update(extras)
    return "\n".join(f"{k}: {v}" for k, v in base.items()) + "\n"
