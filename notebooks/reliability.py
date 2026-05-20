"""Test–retest reliability of field performance measures (Table 1).

Computes ICC(3,1) (two-way mixed, consistency), CV% (mean of within-subject
coefficients of variation), and SEM (SD_pooled · sqrt(1 − ICC)) for every
trial pair declared in `data_prep.RELIABILITY_PAIRS`.

Outputs (idempotent):
    results/reliability_summary.txt
    results/reliability_table.csv

Run from notebooks/:
    ../.venv/bin/python reliability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import RESULTS_DIR, build_file_header  # noqa: E402
from data_prep import RELIABILITY_PAIRS, get_field_sample, load_data  # noqa: E402


def reliability(data: pd.DataFrame, col1: str, col2: str) -> dict:
    """Compute ICC(3,1), CV%, and SEM for two repeated trials.

    ICC is obtained from pingouin (two-way mixed, consistency, single rater).
    CV% is the mean of per-subject CVs (within-subject SD divided by mean).
    SEM is the pooled SD times sqrt(1 − ICC).
    """
    tmp = data[[col1, col2]].dropna()
    n = len(tmp)
    if n < 3:
        return {"N": n, "ICC": np.nan, "CI95": "", "CV (%)": np.nan, "SEM": np.nan}

    long = pd.DataFrame({
        "Subject": np.tile(np.arange(n), 2),
        "Trial":   np.repeat(["T1", "T2"], n),
        "Value":   np.concatenate([tmp[col1].values, tmp[col2].values]),
    })
    icc_table = pg.intraclass_corr(
        data=long, targets="Subject", raters="Trial", ratings="Value"
    )
    # Two-way mixed effects, consistency, single rater.
    # pingouin <0.5 labelled this row "ICC3"; pingouin >=0.5 uses "ICC(C,1)".
    type_col = icc_table["Type"].astype(str)
    row = icc_table[type_col.isin(["ICC3", "ICC(C,1)"])]
    icc_val = row["ICC"].values[0]
    ci_lo = row["CI95"].values[0][0]
    ci_hi = row["CI95"].values[0][1]

    subj_mean = tmp.mean(axis=1)
    subj_sd = tmp.std(axis=1, ddof=1)
    cv_pct = (subj_sd / subj_mean * 100).mean()

    sd_pooled = np.std(tmp.values.flatten(), ddof=1)
    sem = sd_pooled * np.sqrt(1 - icc_val) if icc_val < 1 else 0.0

    return {
        "N":      n,
        "ICC":    round(icc_val, 3),
        "CI95":   f"[{ci_lo:.3f}, {ci_hi:.3f}]",
        "CV (%)": round(cv_pct, 2),
        "SEM":    round(sem, 3),
    }


def run() -> pd.DataFrame:
    df = load_data()
    df_field = get_field_sample(df)

    rows = []
    for label, c1, c2 in RELIABILITY_PAIRS:
        res = reliability(df_field, c1, c2)
        res["Variable"] = label
        rows.append(res)

    results = pd.DataFrame(rows)[["Variable", "N", "ICC", "CI95", "CV (%)", "SEM"]]
    results.columns = ["Variable", "N", "ICC", "95% CI", "CV (%)", "SEM"]
    return results


def write_outputs(results: pd.DataFrame, n_field: int) -> None:
    lines = ["=" * 70]
    lines.append("Test–Retest Reliability — Field Sample")
    lines.append("=" * 70)
    lines.append("")
    lines.append(build_file_header({"pingouin": pg.__version__}).rstrip())
    lines.append(f"Sample: n = {n_field} (field sample)")
    lines.append("")
    lines.append(results.to_string(index=False))

    summary_path = RESULTS_DIR / "reliability_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Wrote: {summary_path}")

    table_path = RESULTS_DIR / "reliability_table.csv"
    results.to_csv(table_path, index=False)
    print(f"Wrote: {table_path}")


if __name__ == "__main__":
    df = load_data()
    df_field = get_field_sample(df)
    print(f"Field sample: n = {len(df_field)}")

    results = run()
    print()
    print("=== Test–Retest Reliability (Field Sample) ===")
    print(results.to_string(index=False))
    print()

    write_outputs(results, n_field=len(df_field))
