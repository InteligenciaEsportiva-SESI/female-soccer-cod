"""Inferential analyses for the isokinetic subsample (n=22).

Generates the numbers that populate the manuscript's Tables 4, 5, 6 and 7:

* IQR outlier screening (flagged, not removed) — diagnostic only.
* Pearson correlations between the three mechanical predictors and the
  three locomotor outcomes (Table 4).
* Two continuous multiple linear regression models with all three
  mechanical predictors standardised to z-scores (no stepwise selection,
  no dichotomisation), one for 20-m sprint time (Table 5) and one for
  relative CODD (Table 6), with residual diagnostics and VIF.
* Descriptive supplementary median-split comparison by relative peak
  extensor torque (Supplementary Table S1 / legacy Table 7), Welch's
  t-test plus Cohen's d.

The Bayesian sensitivity arm and the menstrual-status sensitivity arm are
estimated in `robustness_check.py` and are not duplicated here.

Outputs (idempotent):
    results/inferential_summary.txt
    results/inferential_correlations.csv
    results/inferential_regression_sprint.csv
    results/inferential_regression_codd.csv
    results/inferential_median_split.csv

Run from notebooks/:
    ../.venv/bin/python inferential.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import OUTCOMES, PREDICTORS, RESULTS_DIR, build_file_header, standardize  # noqa: E402
from data_prep import get_iso_sample, load_data  # noqa: E402

CORRELATION_OUTCOMES = ["Sprint_20m_Best", "ZigZag_Best", "CODD_Rel"]
MEDIAN_SPLIT_VARS = ["Sprint_20m_Best", "HQ_Ratio", "CODD_Rel"]


def iqr_outliers(df: pd.DataFrame, variables: list[str]) -> list[dict]:
    """Flag (do not remove) observations beyond the 1.5·IQR fences."""
    out = []
    for var in variables:
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = df[(df[var] < lower) | (df[var] > upper)]
        out.append({
            "Variable": var,
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "Fence_lower": lower, "Fence_upper": upper,
            "n_flagged": len(flagged),
            "flagged_ids": ", ".join(map(str, flagged["Id"].tolist())) if "Id" in df.columns else "",
        })
    return out


def pearson_matrix(df: pd.DataFrame, predictors: list[str], outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for pred in predictors:
        for out in outcomes:
            valid = df[[pred, out]].dropna()
            r, p = stats.pearsonr(valid[pred], valid[out])
            rows.append({"predictor": pred, "outcome": out, "r": r, "p": p, "n": len(valid)})
    return pd.DataFrame(rows)


def continuous_ols(df: pd.DataFrame, outcome: str, predictors: list[str]) -> dict:
    """Multiple OLS regression with all predictors retained and standardised."""
    X = pd.DataFrame({p: standardize(df[p]) for p in predictors})
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.OLS(y, X).fit()
    resid = model.resid

    sw_stat, sw_p = stats.shapiro(resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X)
    dw = durbin_watson(resid)
    cooks = model.get_influence().cooks_distance[0]

    # VIF on the standardised design (excluding constant column)
    X_raw = pd.DataFrame({p: standardize(df[p]) for p in predictors})
    X_with_const = sm.add_constant(X_raw)
    vif = {
        col: float(variance_inflation_factor(X_with_const.values, i + 1))
        for i, col in enumerate(predictors)
    }

    coefs = []
    for name in X.columns:
        coefs.append({
            "param": name,
            "estimate": float(model.params[name]),
            "se": float(model.bse[name]),
            "lower_95": float(model.conf_int().loc[name, 0]),
            "upper_95": float(model.conf_int().loc[name, 1]),
            "p_value": float(model.pvalues[name]),
        })

    return {
        "outcome": outcome,
        "predictors": predictors,
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "coefs": coefs,
        "shapiro_w": float(sw_stat),
        "shapiro_p": float(sw_p),
        "bp_lm": float(bp_lm),
        "bp_p": float(bp_p),
        "durbin_watson": float(dw),
        "cooks_max": float(cooks.max()),
        "vif": vif,
        "summary_text": str(model.summary()),
    }


def cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    n1, n2 = len(g1), len(g2)
    pooled_sd = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def median_split(df: pd.DataFrame, split_var: str, compare_vars: list[str]) -> tuple[pd.DataFrame, float, int, int]:
    median = df[split_var].median()
    high = df[df[split_var] >= median]
    low = df[df[split_var] < median]

    rows = []
    for var in compare_vars:
        h = high[var].dropna()
        l = low[var].dropna()
        t_stat, p_val = stats.ttest_ind(h, l, equal_var=False)
        d = cohens_d(h, l)
        rows.append({
            "Variable": var,
            "High_Mean": h.mean(),
            "Low_Mean": l.mean(),
            "Difference": h.mean() - l.mean(),
            "t": t_stat,
            "p": p_val,
            "Cohen_d": d,
        })
    return pd.DataFrame(rows), median, len(high), len(low)


def write_summary(iso, outliers, corr, ols_sprint, ols_codd, median_df, median_val, n_high, n_low) -> None:
    lines = ["=" * 78]
    lines.append("Inferential Analyses — Isokinetic Subsample (n = 22)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(build_file_header().rstrip())
    lines.append("")

    lines.append("=" * 78)
    lines.append("IQR outlier screening (flagged, not removed)")
    lines.append("=" * 78)
    for o in outliers:
        lines.append(f"  {o['Variable']}: Q1={o['Q1']:.3f} Q3={o['Q3']:.3f} IQR={o['IQR']:.3f} "
                     f"fences=[{o['Fence_lower']:.3f}, {o['Fence_upper']:.3f}]  "
                     f"n_flagged={o['n_flagged']}{(' ids=' + o['flagged_ids']) if o['flagged_ids'] else ''}")
    lines.append("")

    lines.append("=" * 78)
    lines.append("Table 4 — Pearson Correlations (predictors × outcomes)")
    lines.append("=" * 78)
    pivot_r = corr.pivot(index="predictor", columns="outcome", values="r")
    pivot_p = corr.pivot(index="predictor", columns="outcome", values="p")
    for pred in PREDICTORS:
        lines.append(f"  {pred}")
        for out in CORRELATION_OUTCOMES:
            r = pivot_r.loc[pred, out]
            p = pivot_p.loc[pred, out]
            sig = " *" if p < 0.05 else ""
            lines.append(f"    vs {out:20s}  r = {r:+.3f}  p = {p:.4f}{sig}")
        lines.append("")

    for name, res in [("Table 5 — Continuous OLS for 20-m Sprint", ols_sprint),
                       ("Table 6 — Continuous OLS for Relative CODD", ols_codd)]:
        lines.append("=" * 78)
        lines.append(name)
        lines.append("=" * 78)
        lines.append(f"  n = {res['n']}")
        lines.append(f"  R²       = {res['r_squared']:.3f}")
        lines.append(f"  Adj. R²  = {res['r_squared_adj']:.3f}")
        lines.append(f"  F({res['df_model']},{res['df_resid']}) = {res['f_statistic']:.3f}, "
                     f"p = {res['f_pvalue']:.4f}")
        lines.append("")
        lines.append("  Coefficients (predictors standardised to z; outcome in original units):")
        lines.append(f"    {'param':<25s} {'estimate':>10s} {'SE':>8s} "
                     f"{'95% CI lower':>14s} {'95% CI upper':>14s} {'p':>8s}")
        for c in res["coefs"]:
            lines.append(f"    {c['param']:<25s} {c['estimate']:>+10.3f} {c['se']:>8.3f} "
                         f"{c['lower_95']:>+14.3f} {c['upper_95']:>+14.3f} {c['p_value']:>8.4f}")
        lines.append("")
        lines.append(f"  Diagnostics:")
        lines.append(f"    Shapiro–Wilk W = {res['shapiro_w']:.4f}, p = {res['shapiro_p']:.4f}")
        lines.append(f"    Breusch–Pagan LM = {res['bp_lm']:.4f}, p = {res['bp_p']:.4f}")
        lines.append(f"    Durbin–Watson    = {res['durbin_watson']:.4f}")
        lines.append(f"    Max Cook's D     = {res['cooks_max']:.4f}")
        lines.append(f"    VIF:")
        for p, v in res["vif"].items():
            lines.append(f"      {p:<25s} {v:.3f}")
        lines.append("")

    lines.append("=" * 78)
    lines.append("Supplementary Table S1 — Median-Split Descriptive Comparison")
    lines.append("=" * 78)
    lines.append(f"  Split variable: Rel_Peak_Ext_Torque  median = {median_val:.3f}")
    lines.append(f"  High-Torque group n = {n_high}; Low-Torque group n = {n_low}")
    lines.append("")
    lines.append(f"    {'Variable':<25s} {'High mean':>10s} {'Low mean':>10s} "
                 f"{'diff':>8s} {'t':>8s} {'p':>8s} {'Cohen d':>10s}")
    for _, row in median_df.iterrows():
        sig = " *" if row["p"] < 0.05 else ""
        lines.append(f"    {row['Variable']:<25s} {row['High_Mean']:>10.3f} {row['Low_Mean']:>10.3f} "
                     f"{row['Difference']:>+8.3f} {row['t']:>+8.3f} {row['p']:>8.4f} "
                     f"{row['Cohen_d']:>+10.3f}{sig}")
    lines.append("")
    lines.append("  Reported as descriptive supplementary context only "
                 "(see main text Methods — Statistical Analysis); the continuous "
                 "regression in Tables 5 and 6 is the inferential anchor.")

    summary_path = RESULTS_DIR / "inferential_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"\nWrote: {summary_path}")

    corr.to_csv(RESULTS_DIR / "inferential_correlations.csv", index=False)
    pd.DataFrame(ols_sprint["coefs"]).to_csv(
        RESULTS_DIR / "inferential_regression_sprint.csv", index=False
    )
    pd.DataFrame(ols_codd["coefs"]).to_csv(
        RESULTS_DIR / "inferential_regression_codd.csv", index=False
    )
    median_df.to_csv(RESULTS_DIR / "inferential_median_split.csv", index=False)
    for p in ["inferential_correlations.csv",
              "inferential_regression_sprint.csv",
              "inferential_regression_codd.csv",
              "inferential_median_split.csv"]:
        print(f"Wrote: {RESULTS_DIR / p}")


if __name__ == "__main__":
    df = load_data()
    iso = get_iso_sample(df)
    print(f"Isokinetic subsample: n = {len(iso)}\n")

    # Drop any rows missing a predictor or outcome before fitting
    keep = OUTCOMES + PREDICTORS + ["ZigZag_Best", "HQ_Ratio"]
    iso_clean = iso.dropna(subset=keep).copy()

    outliers = iqr_outliers(iso_clean, PREDICTORS + OUTCOMES)
    corr = pearson_matrix(iso_clean, PREDICTORS, CORRELATION_OUTCOMES)
    ols_sprint = continuous_ols(iso_clean, "Sprint_20m_Best", PREDICTORS)
    ols_codd = continuous_ols(iso_clean, "CODD_Rel", PREDICTORS)
    median_df, median_val, n_high, n_low = median_split(
        iso_clean, "Rel_Peak_Ext_Torque", MEDIAN_SPLIT_VARS
    )

    print("=== Table 4 — Pearson Correlations ===")
    print(corr.to_string(index=False))
    print()
    print("=== Table 5 — Continuous OLS Sprint ===")
    print(ols_sprint["summary_text"])
    print()
    print("=== Table 6 — Continuous OLS CODD ===")
    print(ols_codd["summary_text"])
    print()
    print("=== Supplementary — Median-Split Descriptive ===")
    print(median_df.to_string(index=False))

    write_summary(iso_clean, outliers, corr, ols_sprint, ols_codd,
                  median_df, median_val, n_high, n_low)
