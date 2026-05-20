"""Descriptive statistics for the field (n=43) and isokinetic (n=22) samples.

Computes mean, SD, t-based 95% CI, and Shapiro–Wilk normality test for each
variable in the manuscript's Tables 2 (field) and 3 (isokinetic).

Outputs (idempotent):
    results/descriptives_field.txt
    results/descriptives_iso.txt
    results/descriptives_field.csv
    results/descriptives_iso.csv

Run from notebooks/:
    ../.venv/bin/python descriptives.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import RESULTS_DIR, build_file_header  # noqa: E402
from data_prep import get_field_sample, get_iso_sample, load_data  # noqa: E402


FIELD_VARS = [
    "Age", "Body_Mass", "Height", "BMI",
    "SJ_Max", "CMJ_Max", "CMJ_Free_Max", "Horiz_Jump_Max",
    "Sprint_20m_Best", "ZigZag_Best", "CODD_Rel", "RSI_DJ30",
]

ISO_VARS = [
    "Rel_Peak_Ext_Torque", "HQ_Ratio",
    "Sprint_20m_Best", "CODD_Rel",
]


def descriptives(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Per-variable mean, SD, t-based 95% CI, Shapiro–Wilk W and p."""
    rows = []
    for var in variables:
        x = df[var].dropna()
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1)

        # t-based 95% CI (appropriate for small samples)
        t_crit = stats.t.ppf(0.975, n - 1)
        se = sd / np.sqrt(n)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se

        # Shapiro–Wilk normality (valid for 3 <= n <= 5000)
        w_stat, p_val = stats.shapiro(x)

        rows.append({
            "Variable": var,
            "n": n,
            "Mean": round(mean, 2),
            "SD": round(sd, 2),
            "CI_lower": round(ci_lo, 2),
            "CI_upper": round(ci_hi, 2),
            "Shapiro_W": round(w_stat, 4),
            "Shapiro_p": round(p_val, 4),
        })

    return pd.DataFrame(rows).set_index("Variable")


def write_outputs(desc_field: pd.DataFrame, desc_iso: pd.DataFrame,
                   n_field: int, n_iso: int) -> None:
    header = build_file_header().rstrip()

    field_lines = ["=" * 80,
                   f"FIELD SAMPLE (n = {n_field})",
                   "=" * 80,
                   "",
                   header,
                   "",
                   desc_field.to_string()]
    field_path = RESULTS_DIR / "descriptives_field.txt"
    field_path.write_text("\n".join(field_lines))
    print(f"Wrote: {field_path}")

    iso_lines = ["=" * 80,
                 f"ISOKINETIC SUBSAMPLE (n = {n_iso})",
                 "=" * 80,
                 "",
                 header,
                 "",
                 desc_iso.to_string()]
    iso_path = RESULTS_DIR / "descriptives_iso.txt"
    iso_path.write_text("\n".join(iso_lines))
    print(f"Wrote: {iso_path}")

    desc_field.to_csv(RESULTS_DIR / "descriptives_field.csv")
    desc_iso.to_csv(RESULTS_DIR / "descriptives_iso.csv")
    print(f"Wrote: {RESULTS_DIR / 'descriptives_field.csv'}")
    print(f"Wrote: {RESULTS_DIR / 'descriptives_iso.csv'}")


if __name__ == "__main__":
    df = load_data()
    field = get_field_sample(df)
    iso = get_iso_sample(df)

    print(f"Field sample: n = {len(field)}")
    print(f"Isokinetic subsample: n = {len(iso)}\n")

    desc_field = descriptives(field, FIELD_VARS)
    desc_iso = descriptives(iso, ISO_VARS)

    print("=" * 80)
    print(f"FIELD SAMPLE (n = {len(field)})")
    print("=" * 80)
    print(desc_field.to_string())

    print("\n" + "=" * 80)
    print(f"ISOKINETIC SUBSAMPLE (n = {len(iso)})")
    print("=" * 80)
    print(desc_iso.to_string())
    print()

    write_outputs(desc_field, desc_iso, n_field=len(field), n_iso=len(iso))
