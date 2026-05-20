"""Robustness checks for the primary continuous regressions.

Re-estimates the two primary outcomes (20-m sprint time and relative CODD) on
the isokinetic subsample (n = 22) under three lenses and compares them
side-by-side:

* OLS continuous multiple regression with three standardised mechanical
  predictors retained.
* Weakly informative Bayesian regression (PyMC, NUTS) on the same
  specification, reporting the posterior median, the 95 % highest density
  interval (HDI) and the probability of direction (pd).
* A 4-predictor sensitivity arm that adds menstrual status at testing as a
  raw 0/1 covariate (menstruating = 1; not bleeding, i.e. pre-menarcheal or
  luteal-phase response = 0) to the same OLS and Bayesian specifications.

Outputs (idempotent across re-runs):
    results/robustness_check_summary.txt
    results/robustness_check_coefs.csv
    results/robustness_check_posterior_summary.csv
    results/robustness_sensitivity_menstrual_summary.txt
    results/robustness_sensitivity_menstrual_coefs.csv

Run from notebooks/:
    ../.venv/bin/python robustness_check.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import (  # noqa: E402
    OUTCOMES,
    PREDICTORS,
    RESULTS_DIR,
    SEED,
    build_file_header,
    standardize,
)
from data_prep import get_iso_sample, load_data  # noqa: E402

np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402

FILE_HEADER = build_file_header({"PyMC": pm.__version__, "ArviZ": az.__version__})


# Menstrual covariate (sensitivity arm). The cross-reference produces a
# `phase` column with three values; the binary recoding pools pre-menarcheal
# athletes and the single luteal-phase response into the "not bleeding"
# bucket (0), because luteal-phase by definition is post-menarcheal but not
# actively bleeding and the form did not graduate cycle phase finely.
MENSTRUAL_CSV = RESULTS_DIR / "cycle_data_iso_match_anon.csv"
MENSTRUAL_BINARY_COL = "menstrual_binary"
PHASE_TO_BINARY = {
    "🩸 Menstruação": 1,
    "🧒 Não menstrua ainda": 0,
    "🌙 Fase lútea": 0,
}


def load_iso_sample_clean() -> pd.DataFrame:
    df = load_data()
    iso = get_iso_sample(df)
    keep = OUTCOMES + PREDICTORS
    return iso[keep].dropna().copy()


def load_menstrual_binary() -> pd.Series:
    """Load the menstrual_binary (0/1) series keyed by iso_id (1..22)."""
    df = pd.read_csv(MENSTRUAL_CSV)
    df[MENSTRUAL_BINARY_COL] = df["phase"].map(PHASE_TO_BINARY)
    if df[MENSTRUAL_BINARY_COL].isna().any():
        unmapped = df[df[MENSTRUAL_BINARY_COL].isna()]["phase"].unique().tolist()
        raise ValueError(f"Unmapped phase values in menstrual CSV: {unmapped}")
    return df.set_index("iso_id")[MENSTRUAL_BINARY_COL].astype(int)


def load_iso_sample_with_menstrual() -> pd.DataFrame:
    """Iso subsample with the menstrual_binary column attached by position.

    The cross-reference assigns ``iso_id`` to the 1-indexed row of
    ``get_iso_sample(df).reset_index(drop=True)``. ``get_iso_sample`` only
    drops NaN on the four iso outcomes, while this script additionally
    requires RSI_DJ30 to be non-null; if RSI_DJ30 had any NaN, the
    position-based join below would silently misalign with the iso_id
    labels. The explicit guard turns that latent error into a hard fail.
    """
    df = load_data()
    iso = get_iso_sample(df).reset_index(drop=True)
    if iso["RSI_DJ30"].isna().any():
        raise ValueError(
            "RSI_DJ30 has NaN in iso subsample — position-based menstrual_binary "
            "join would misalign with iso_id from cycle_data_cross_reference. "
            "Switch to a name-keyed join before re-running."
        )
    keep = OUTCOMES + PREDICTORS
    iso_clean = iso[keep].dropna().reset_index(drop=True).copy()
    if len(iso_clean) != 22:
        raise ValueError(f"Expected 22 iso athletes; got {len(iso_clean)}")
    menstrual = load_menstrual_binary()
    if len(menstrual) != 22:
        raise ValueError(f"Expected 22 menstrual entries; got {len(menstrual)}")
    iso_clean[MENSTRUAL_BINARY_COL] = menstrual.reindex(range(1, 23)).values
    return iso_clean


def bayes_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    binary_predictors: list[str] | None = None,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    seed: int = SEED,
) -> dict:
    """Weakly informative Bayesian linear regression in PyMC.

    Priors:
        β0 ~ Normal(0, 10)
        β_i ~ Normal(0, 1)   (weakly informative on standardised predictors)
        σ ~ HalfNormal(1)

    Predictors listed in ``binary_predictors`` are passed through as raw 0/1
    instead of being z-standardised; β for such a predictor is the
    mean-difference between groups on the outcome's original scale.

    Returns posterior summary (with 95 % HDI), Bayesian R² (median + 95 %
    HDI), probability-of-direction per coefficient, and convergence
    diagnostics.
    """
    binary_set = set(binary_predictors or [])
    cols = []
    for p in predictors:
        if p in binary_set:
            cols.append(df[p].astype(float).values)
        else:
            cols.append(standardize(df[p]).values)
    X = np.column_stack(cols)
    y = df[outcome].values
    n_pred = X.shape[1]

    with pm.Model() as model:  # noqa: F841
        β0 = pm.Normal("β0", mu=0, sigma=10)
        β = pm.Normal("β", mu=0, sigma=1, shape=n_pred)
        σ = pm.HalfNormal("σ", sigma=1)

        μ = pm.Deterministic("μ", β0 + pm.math.dot(X, β))
        y_obs = pm.Normal("y_obs", mu=μ, sigma=σ, observed=y)  # noqa: F841

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )

    summary = az.summary(
        idata, var_names=["β0", "β", "σ"], ci_prob=0.95, ci_kind="hdi"
    )
    r_hat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    n_divergent = int(idata.sample_stats.diverging.values.sum())

    pd_per_coef = []
    posterior = idata.posterior["β"].stack(sample=("chain", "draw"))
    for i, p_name in enumerate(predictors):
        samples = posterior.isel({"β_dim_0": i}).values
        median = float(np.median(samples))
        pd_val = float((samples > 0).mean() if median > 0 else (samples < 0).mean())
        pd_per_coef.append({"predictor": p_name, "median": median, "pd": pd_val})

    # Bayesian R² via the variance-decomposition definition R² = Var(μ) /
    # (Var(μ) + σ²) (Gelman et al. 2019, Am. Stat.). ArviZ ≥ 1.x exposes
    # this as `bayesian_r2`; earlier `r2_score` is deprecated.
    r2_result = az.bayesian_r2(idata, pred_mean="μ", scale="σ", summary=False)
    r2_samples = np.asarray(r2_result).flatten()
    r2_median = float(np.median(r2_samples))
    r2_hdi_arr = np.asarray(az.hdi(r2_samples, prob=0.95)).flatten()
    r2_hdi_lower = float(r2_hdi_arr[0])
    r2_hdi_upper = float(r2_hdi_arr[-1])

    return {
        "outcome": outcome,
        "predictors": predictors,
        "n": len(y),
        "summary_df": summary,
        "r_hat_max": r_hat_max,
        "ess_min": ess_min,
        "n_divergent": n_divergent,
        "pd_per_coef": pd_per_coef,
        "r2_median": r2_median,
        "r2_hdi_lower": r2_hdi_lower,
        "r2_hdi_upper": r2_hdi_upper,
        "idata": idata,
    }


def _strip_summary_timestamps(text: str) -> str:
    """Replace statsmodels' wall-clock Date/Time fields with fixed placeholders
    so the rendered summary is byte-identical across runs."""
    text = re.sub(
        r"Date:\s+\w+,\s+\d+\s+\w+\s+\d+",
        "Date:                Mon, 01 Jan 1900",
        text,
    )
    text = re.sub(
        r"Time:\s+\d+:\d+:\d+",
        "Time:                        00:00:00",
        text,
    )
    return text


def ols_continuous(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    binary_predictors: list[str] | None = None,
) -> dict:
    """Continuous OLS multiple regression on standardised (or raw 0/1) predictors.

    Predictors listed in ``binary_predictors`` are passed through as raw 0/1
    instead of being z-standardised; this preserves the mean-difference
    interpretation for binary covariates.
    """
    from scipy.stats import shapiro
    from statsmodels.stats.diagnostic import het_breuschpagan

    binary_set = set(binary_predictors or [])
    X_cols = {}
    for p in predictors:
        if p in binary_set:
            X_cols[p] = df[p].astype(float)
        else:
            X_cols[p] = standardize(df[p])
    X = pd.DataFrame(X_cols)
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.OLS(y, X).fit()

    sw_stat, sw_p = shapiro(model.resid)
    bp_lm, bp_lm_p, _, _ = het_breuschpagan(model.resid, X)

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
        "coefs": coefs,
        "shapiro_p": float(sw_p),
        "breusch_pagan_p": float(bp_lm_p),
        "summary_text": str(model.summary()),
    }


def write_main_summary(
    ols_sprint, ols_codd, bayes_sprint, bayes_codd, iso
):
    """Write the headline OLS + Bayes comparison plus two CSVs."""
    lines = ["=" * 70]
    lines.append("Robustness Check — OLS continuous vs Bayesian sensitivity")
    lines.append("=" * 70)
    lines.append("")
    lines.append(FILE_HEADER)
    lines.append(f"Sample: n = {len(iso)} (isokinetic subsample)")
    lines.append(f"Outcomes: {', '.join(OUTCOMES)}")
    lines.append(f"Predictors (z-standardised): {', '.join(PREDICTORS)}")
    lines.append("")

    for label, ols_res, bayes_res in [
        ("Sprint_20m_Best", ols_sprint, bayes_sprint),
        ("CODD_Rel", ols_codd, bayes_codd),
    ]:
        lines.append("=" * 70)
        lines.append(f"Outcome: {label}")
        lines.append("=" * 70)
        lines.append(f"OLS R²       = {ols_res['r_squared']:.3f}")
        lines.append(f"OLS Adj. R²  = {ols_res['r_squared_adj']:.3f}")
        lines.append(f"Bayes R² med = {bayes_res['r2_median']:.3f}  "
                     f"95% HDI [{bayes_res['r2_hdi_lower']:.3f}, "
                     f"{bayes_res['r2_hdi_upper']:.3f}]")
        lines.append("")
        lines.append("Bayes probability of direction by coefficient:")
        for pd_item in bayes_res["pd_per_coef"]:
            lines.append(
                f"  {pd_item['predictor']:25s}  "
                f"median = {pd_item['median']:+.4f}   pd = {pd_item['pd']:.4f}"
            )
        lines.append("")
        lines.append("Full OLS summary:")
        lines.append(_strip_summary_timestamps(ols_res["summary_text"]))
        lines.append(f"Shapiro–Wilk p: {ols_res['shapiro_p']:.4f}")
        lines.append(f"Breusch–Pagan p: {ols_res['breusch_pagan_p']:.4f}")
        lines.append("")
        lines.append("Full Bayes posterior summary:")
        lines.append(bayes_res["summary_df"].to_string())
        lines.append(
            f"\nDiagnostics: r̂_max = {bayes_res['r_hat_max']:.4f}, "
            f"ESS_bulk_min = {bayes_res['ess_min']:.0f}, "
            f"divergent = {bayes_res['n_divergent']}"
        )
        lines.append("")

    summary_path = RESULTS_DIR / "robustness_check_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"\nWrote: {summary_path}")

    rows = []
    for res in [ols_sprint, ols_codd]:
        for c in res["coefs"]:
            rows.append({
                "outcome": res["outcome"], "model": "OLS", "param": c["param"],
                "estimate": c["estimate"], "lower_95": c["lower_95"],
                "upper_95": c["upper_95"], "scale": "95% CI",
                "pd_or_p": c["p_value"],
            })
    for bayes_res in [bayes_sprint, bayes_codd]:
        sdf = bayes_res["summary_df"]
        for idx in sdf.index:
            lower = sdf.loc[idx, "hdi95_lb"] if "hdi95_lb" in sdf.columns else np.nan
            upper = sdf.loc[idx, "hdi95_ub"] if "hdi95_ub" in sdf.columns else np.nan
            rows.append({
                "outcome": bayes_res["outcome"], "model": "Bayes", "param": idx,
                "estimate": sdf.loc[idx, "mean"],
                "lower_95": lower, "upper_95": upper,
                "scale": "95% HDI", "pd_or_p": "",
            })
    coefs_df = pd.DataFrame(rows)
    coefs_path = RESULTS_DIR / "robustness_check_coefs.csv"
    coefs_df.to_csv(coefs_path, index=False)
    print(f"Wrote: {coefs_path}")

    posteriors = []
    for bayes_res in [bayes_sprint, bayes_codd]:
        sdf = bayes_res["summary_df"].reset_index()
        sdf["outcome"] = bayes_res["outcome"]
        posteriors.append(sdf)
    post_df = pd.concat(posteriors, ignore_index=True)
    post_path = RESULTS_DIR / "robustness_check_posterior_summary.csv"
    post_df.to_csv(post_path, index=False)
    print(f"Wrote: {post_path}")


def run_sensitivity_arm(iso_with_menstrual: pd.DataFrame) -> dict:
    """Re-estimate both outcomes with menstrual_binary as a 4th covariate."""
    sens_predictors = PREDICTORS + [MENSTRUAL_BINARY_COL]
    binary_preds = [MENSTRUAL_BINARY_COL]

    print("\n" + "=" * 70)
    print("Sensitivity arm — menstrual_binary as 4th covariate (n = 22)")
    print("=" * 70)
    print(f"Predictors (4): {', '.join(sens_predictors)}")
    print(f"Binary (not standardised): {', '.join(binary_preds)}")
    print(f"menstrual_binary distribution: "
          f"{int(iso_with_menstrual[MENSTRUAL_BINARY_COL].sum())} menstruating, "
          f"{int((1 - iso_with_menstrual[MENSTRUAL_BINARY_COL]).sum())} not")
    print()

    return {
        "ols_sprint": ols_continuous(
            iso_with_menstrual, "Sprint_20m_Best", sens_predictors,
            binary_predictors=binary_preds,
        ),
        "ols_codd": ols_continuous(
            iso_with_menstrual, "CODD_Rel", sens_predictors,
            binary_predictors=binary_preds,
        ),
        "bayes_sprint": bayes_regression(
            iso_with_menstrual, "Sprint_20m_Best", sens_predictors,
            binary_predictors=binary_preds,
        ),
        "bayes_codd": bayes_regression(
            iso_with_menstrual, "CODD_Rel", sens_predictors,
            binary_predictors=binary_preds,
        ),
    }


def write_sensitivity_summary(
    main: dict, sens: dict, n_menstruating: int, n_not_menstruating: int
):
    """Side-by-side comparison of the main 3-predictor vs the 4-predictor arm."""
    lines = ["=" * 78]
    lines.append("Sensitivity Arm — Menstrual Status as a 4th Covariate")
    lines.append("=" * 78)
    lines.append("")
    lines.append(FILE_HEADER.rstrip())
    lines.append("")
    lines.append(f"Sample: n = 22 (iso subsample; 100% coverage)")
    lines.append(f"  Menstruating at test:     {n_menstruating}")
    lines.append(f"  Not menstruating at test: {n_not_menstruating}  "
                 f"(pre-menarcheal + 1 post-menarcheal luteal-phase entry pooled here)")
    lines.append("")
    lines.append("Continuous predictors (z-standardised): "
                 + ", ".join(PREDICTORS))
    lines.append(f"Binary covariate (0/1, not standardised): {MENSTRUAL_BINARY_COL}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  β for continuous predictors: change in outcome per +1 SD of predictor.")
    lines.append("  β for menstrual_binary: mean-difference between menstruating (= 1)")
    lines.append("  and not-bleeding (= 0) athletes, holding mechanical predictors at mean.")
    lines.append("")
    lines.append("Inferential framing:")
    lines.append("  This arm is a ROBUSTNESS CHECK on the direction-stability of the main")
    lines.append("  3-predictor mechanical paradox — not an inferential test of menstrual")
    lines.append("  status. With n = 22 and 4 predictors (5.5 obs per predictor, below")
    lines.append("  Harrell's 10:1 rule), the 4-predictor model is intentionally")
    lines.append("  over-parameterised; the headline read is whether the three mechanical")
    lines.append("  β estimates preserve sign and magnitude under adjustment (see the")
    lines.append("  'Direction held?' column).")
    lines.append("")
    lines.append("Prior calibration:")
    lines.append("  β ~ Normal(0, 1) is weakly informative for SD-standardised continuous")
    lines.append("  predictors. For the raw 0/1 binary covariate the prior is scale-dependent")
    lines.append("  on the outcome's residual SD:")
    lines.append("    Sprint (residual SD ≈ 0.09 s): prior wide; minimal shrinkage.")
    lines.append("    CODD   (residual SD ≈ 4.7 %): prior SD = 1 < OLS SE ≈ 2.8; posterior")
    lines.append("                                  median is moderately shrunk toward 0.")
    lines.append("  Implication for CODD menstrual_binary: OLS β ≈ −1.9 vs Bayes median ≈ 0")
    lines.append("  reflects prior-driven shrinkage, not contradiction. Both arms agree")
    lines.append("  the binary covariate is not credibly distinct from zero (OLS p > 0.5,")
    lines.append("  Bayes pd ≈ 0.5).")
    lines.append("")

    for outcome_label, main_ols, main_bayes, sens_ols, sens_bayes in [
        ("Sprint_20m_Best", main["ols_sprint"], main["bayes_sprint"],
         sens["ols_sprint"], sens["bayes_sprint"]),
        ("CODD_Rel", main["ols_codd"], main["bayes_codd"],
         sens["ols_codd"], sens["bayes_codd"]),
    ]:
        lines.append("=" * 78)
        lines.append(f"Outcome: {outcome_label}")
        lines.append("=" * 78)
        lines.append("")
        lines.append(f"OLS R²:        main 3-pred = {main_ols['r_squared']:.3f}  "
                     f"|  sensitivity 4-pred = {sens_ols['r_squared']:.3f}")
        lines.append(f"OLS R²_adj:    main 3-pred = {main_ols['r_squared_adj']:.3f}  "
                     f"|  sensitivity 4-pred = {sens_ols['r_squared_adj']:.3f}")
        lines.append(f"Bayes R² med:  main 3-pred = {main_bayes['r2_median']:.3f}  "
                     f"|  sensitivity 4-pred = {sens_bayes['r2_median']:.3f}")
        lines.append("")
        lines.append("OLS coefficient comparison (β [95% CI], p):")
        lines.append(f"  {'Predictor':<28} {'Main (3-pred)':<32} {'Sensitivity (4-pred)':<32}")
        sens_coefs = {c["param"]: c for c in sens_ols["coefs"]}
        for c in main_ols["coefs"]:
            name = c["param"]
            sens_c = sens_coefs.get(name)
            if sens_c is None:
                continue
            main_str = (f"{c['estimate']:+.3f} [{c['lower_95']:+.3f}, "
                        f"{c['upper_95']:+.3f}] p={c['p_value']:.3f}")
            sens_str = (f"{sens_c['estimate']:+.3f} [{sens_c['lower_95']:+.3f}, "
                        f"{sens_c['upper_95']:+.3f}] p={sens_c['p_value']:.3f}")
            lines.append(f"  {name:<28} {main_str:<32} {sens_str:<32}")
        for c in sens_ols["coefs"]:
            if c["param"] == MENSTRUAL_BINARY_COL:
                sens_str = (f"{c['estimate']:+.3f} [{c['lower_95']:+.3f}, "
                            f"{c['upper_95']:+.3f}] p={c['p_value']:.3f}")
                lines.append(f"  {c['param']:<28} {'(not in main)':<32} {sens_str:<32}")
        lines.append("")
        lines.append("Bayes probability of direction (pd) comparison:")
        lines.append(f"  {'Predictor':<28} {'Main pd':<14} {'Sensitivity pd':<18} "
                     f"{'Direction held?':<18}")
        main_pd = {p["predictor"]: p for p in main_bayes["pd_per_coef"]}
        for p_name in PREDICTORS:
            main_p = main_pd[p_name]
            sens_p = next(
                p for p in sens_bayes["pd_per_coef"] if p["predictor"] == p_name
            )
            same_dir = "yes" if (main_p["median"] * sens_p["median"]) > 0 else "NO"
            lines.append(
                f"  {p_name:<28} "
                f"{main_p['pd']:.4f} ({main_p['median']:+.3f})   "
                f"{sens_p['pd']:.4f} ({sens_p['median']:+.3f})    "
                f"{same_dir}"
            )
        m_p = next(
            p for p in sens_bayes["pd_per_coef"]
            if p["predictor"] == MENSTRUAL_BINARY_COL
        )
        lines.append(
            f"  {MENSTRUAL_BINARY_COL:<28} "
            f"{'(not in main)':<14} "
            f"{m_p['pd']:.4f} ({m_p['median']:+.3f})"
        )
        lines.append("")
        lines.append("Sensitivity Bayes diagnostics:")
        lines.append(f"  r̂ max:     {sens_bayes['r_hat_max']:.4f}  (target ≤ 1.01)")
        lines.append(f"  ESS min:    {sens_bayes['ess_min']:.0f}    (target ≥ 400)")
        lines.append(f"  Divergent:  {sens_bayes['n_divergent']}     (target = 0)")
        lines.append("")
        lines.append("Full sensitivity OLS summary:")
        lines.append(_strip_summary_timestamps(sens_ols["summary_text"]))
        lines.append(f"Shapiro–Wilk p: {sens_ols['shapiro_p']:.4f}")
        lines.append(f"Breusch–Pagan p: {sens_ols['breusch_pagan_p']:.4f}")
        lines.append("")
        lines.append("Full sensitivity Bayes summary:")
        lines.append(sens_bayes["summary_df"].to_string())
        lines.append("")

    summary_path = RESULTS_DIR / "robustness_sensitivity_menstrual_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"\nWrote: {summary_path}")

    rows = []
    for outcome_label, main_ols, sens_ols, main_bayes, sens_bayes in [
        ("Sprint_20m_Best", main["ols_sprint"], sens["ols_sprint"],
         main["bayes_sprint"], sens["bayes_sprint"]),
        ("CODD_Rel", main["ols_codd"], sens["ols_codd"],
         main["bayes_codd"], sens["bayes_codd"]),
    ]:
        for c in main_ols["coefs"]:
            rows.append({
                "outcome": outcome_label, "arm": "main",
                "model": "OLS", "param": c["param"],
                "estimate": c["estimate"],
                "lower_95": c["lower_95"], "upper_95": c["upper_95"],
                "scale": "95% CI", "pd_or_p": c["p_value"],
            })
        for c in sens_ols["coefs"]:
            rows.append({
                "outcome": outcome_label, "arm": "sensitivity",
                "model": "OLS", "param": c["param"],
                "estimate": c["estimate"],
                "lower_95": c["lower_95"], "upper_95": c["upper_95"],
                "scale": "95% CI", "pd_or_p": c["p_value"],
            })
        for arm_label, bres in [("main", main_bayes), ("sensitivity", sens_bayes)]:
            sdf = bres["summary_df"]
            for idx in sdf.index:
                lower = sdf.loc[idx, "hdi95_lb"] if "hdi95_lb" in sdf.columns else np.nan
                upper = sdf.loc[idx, "hdi95_ub"] if "hdi95_ub" in sdf.columns else np.nan
                rows.append({
                    "outcome": outcome_label, "arm": arm_label,
                    "model": "Bayes", "param": idx,
                    "estimate": sdf.loc[idx, "mean"],
                    "lower_95": lower, "upper_95": upper,
                    "scale": "95% HDI", "pd_or_p": "",
                })
    coefs_df = pd.DataFrame(rows)
    coefs_path = RESULTS_DIR / "robustness_sensitivity_menstrual_coefs.csv"
    coefs_df.to_csv(coefs_path, index=False)
    print(f"Wrote: {coefs_path}")


if __name__ == "__main__":
    iso = load_iso_sample_clean()
    print(f"n = {len(iso)}")
    print()
    print("Descriptive stats:")
    print(iso.describe())
    print()

    ols_sprint = ols_continuous(iso, "Sprint_20m_Best", PREDICTORS)
    print("=" * 70)
    print("OLS — Sprint 20m (continuous, n = 22)")
    print("=" * 70)
    print(ols_sprint["summary_text"])

    ols_codd = ols_continuous(iso, "CODD_Rel", PREDICTORS)
    print("\n" + "=" * 70)
    print("OLS — CODD relativo (continuous, n = 22)")
    print("=" * 70)
    print(ols_codd["summary_text"])

    print("\n" + "=" * 70)
    print("Bayesian regression — Sprint 20m (weakly informative, n = 22)")
    print("=" * 70)
    bayes_sprint = bayes_regression(iso, "Sprint_20m_Best", PREDICTORS)
    print(bayes_sprint["summary_df"].to_string())
    print(f"\nBayes R²: median = {bayes_sprint['r2_median']:.3f}, "
          f"95% HDI [{bayes_sprint['r2_hdi_lower']:.3f}, "
          f"{bayes_sprint['r2_hdi_upper']:.3f}]")

    print("\n" + "=" * 70)
    print("Bayesian regression — CODD relativo (weakly informative, n = 22)")
    print("=" * 70)
    bayes_codd = bayes_regression(iso, "CODD_Rel", PREDICTORS)
    print(bayes_codd["summary_df"].to_string())
    print(f"\nBayes R²: median = {bayes_codd['r2_median']:.3f}, "
          f"95% HDI [{bayes_codd['r2_hdi_lower']:.3f}, "
          f"{bayes_codd['r2_hdi_upper']:.3f}]")

    write_main_summary(ols_sprint, ols_codd, bayes_sprint, bayes_codd, iso)

    # ----- Sensitivity arm: menstrual_binary as a 4th covariate -----
    iso_with_menstrual = load_iso_sample_with_menstrual()
    n_menstruating = int(iso_with_menstrual[MENSTRUAL_BINARY_COL].sum())
    n_not_menstruating = int((1 - iso_with_menstrual[MENSTRUAL_BINARY_COL]).sum())
    sens = run_sensitivity_arm(iso_with_menstrual)
    main = {
        "ols_sprint": ols_sprint, "ols_codd": ols_codd,
        "bayes_sprint": bayes_sprint, "bayes_codd": bayes_codd,
    }
    write_sensitivity_summary(main, sens, n_menstruating, n_not_menstruating)
