"""Publication figures (TIFF 300 dpi).

Main text:

* Figure 1 — Efficiency Paradox: bivariate scatterplots of relative peak
  extensor torque vs 20-m sprint time (panel A) and vs relative CODD
  (panel B), each with a univariate OLS fit and bivariate R² annotated.
* Figure 2 — Multivariate Coefficient Forest: side-by-side horizontal
  forest of standardised β coefficients (per 1 SD of predictor) for the
  three mechanical predictors across the two outcomes. Same-axis layout
  makes the paradoxical sign-flip of every predictor between Sprint and
  CODD visually obvious.
* Figure 3 — Continuous Typology: scatter of 20-m sprint time vs
  relative CODD for the 22 athletes, with points colour-encoded by
  relative peak extensor torque on a continuous grey-to-red gradient.
  Replaces the legacy median-split typology with a non-dichotomised
  visualisation while preserving the storyline (high-torque athletes
  cluster in the "fast sprint × high CODD" region).

Supplementary:

* Supplementary Figure S1 — Mechanism: median-split bar charts comparing
  the High-Torque and Low-Torque halves on sprint, H:Q ratio and CODD,
  with Welch's t-test significance brackets. Descriptive only; the
  inferential anchor is the continuous regression in the main text.
* Supplementary Figure S2 — Typology Quadrant: the legacy quadrant
  scatter (categorical High vs Low Torque colouring + quadrant labels)
  kept for continuity with the earlier circulated draft.

Outputs:
    figures/Figure1_Paradox.tiff
    figures/Figure2_CoefficientForest.tiff
    figures/Figure3_ContinuousTypology.tiff
    figures/SupplementaryFigureS1_Mechanism.tiff
    figures/SupplementaryFigureS2_TypologyQuadrant.tiff

Run from notebooks/:
    ../.venv/bin/python figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import OUTCOMES, PALETTE, PREDICTORS, standardize  # noqa: E402
from data_prep import get_iso_sample, load_data  # noqa: E402

FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DPI = 300
FMT = "tiff"

# Standardised palette across all figures.
COLOR_PRINCIPAL = PALETTE["secondary"]   # red #c0392b — Rel_Peak_Ext_Torque (paradox driver)
COLOR_NEUTRAL = "#7f8c8d"                # grey — neutral secondary
COLOR_LIGHT_GREY = "#bdc3c7"             # light grey — Low-Torque group / background
COLOR_ZERO = "#34495e"                   # dark slate — zero / reference lines

# Categorical median-split palette (kept for supplementary figures).
GROUP_PALETTE = {"High Torque": COLOR_PRINCIPAL, "Low Torque": COLOR_LIGHT_GREY}

# Continuous colormap: light grey ↔ saturated red, anchored on the standard palette.
TORQUE_COLORMAP = LinearSegmentedColormap.from_list(
    "torque_grey_red", [COLOR_LIGHT_GREY, COLOR_PRINCIPAL]
)

# Pretty predictor labels for forest-plot axis.
PRETTY_PREDICTOR = {
    "Rel_Peak_Ext_Torque": "Rel. Peak Ext. Torque",
    "HQ_Ratio": "H:Q Ratio",
    "RSI_DJ30": "RSI DJ30",
}
PRINCIPAL_PREDICTOR = "Rel_Peak_Ext_Torque"


def configure_journal_style() -> None:
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def fit_continuous_ols(df, outcome, predictors):
    """Quick refit so figures.py is self-contained (mirrors inferential.py)."""
    X = sm.add_constant(
        np.column_stack([standardize(df[p]).values for p in predictors])
    )
    model = sm.OLS(df[outcome].values, X).fit()
    coefs = []
    for i, name in enumerate(predictors, start=1):
        ci = model.conf_int()
        coefs.append({
            "predictor": name,
            "estimate": float(model.params[i]),
            "lower_95": float(ci[i, 0]),
            "upper_95": float(ci[i, 1]),
            "p_value": float(model.pvalues[i]),
        })
    return coefs


def figure_1_paradox(iso) -> None:
    X = sm.add_constant(iso["Rel_Peak_Ext_Torque"])
    model_sprint = sm.OLS(iso["Sprint_20m_Best"], X).fit()
    r2_sprint = model_sprint.rsquared
    p_sprint = model_sprint.pvalues.iloc[1]
    model_codd = sm.OLS(iso["CODD_Rel"], X).fit()
    r2_codd = model_codd.rsquared
    p_codd = model_codd.pvalues.iloc[1]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    sns.regplot(
        x="Rel_Peak_Ext_Torque", y="Sprint_20m_Best", data=iso,
        ax=axes[0], color=COLOR_NEUTRAL,
        scatter_kws={"s": 30, "alpha": 0.7}, line_kws={"linewidth": 1.5},
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Relative Peak Ext. Torque (N·m·kg⁻¹)")
    axes[0].set_ylabel("20 m Sprint (s)")
    axes[0].set_title("A", loc="left", fontweight="bold")
    p_txt_sprint = f"p = {p_sprint:.3f}" if p_sprint >= 0.001 else "p < 0.001"
    axes[0].text(
        0.05, 0.05, f"R² = {r2_sprint:.2f}\n{p_txt_sprint}",
        transform=axes[0].transAxes, fontsize=8, verticalalignment="bottom",
    )

    sns.regplot(
        x="Rel_Peak_Ext_Torque", y="CODD_Rel", data=iso,
        ax=axes[1], color=COLOR_PRINCIPAL,
        scatter_kws={"s": 30, "alpha": 0.7}, line_kws={"linewidth": 1.5},
    )
    axes[1].set_xlabel("Relative Peak Ext. Torque (N·m·kg⁻¹)")
    axes[1].set_ylabel("CODD Relative (%)")
    axes[1].set_title("B", loc="left", fontweight="bold")
    p_txt_codd = f"p = {p_codd:.3f}" if p_codd >= 0.001 else "p < 0.001"
    axes[1].text(
        0.05, 0.95, f"R² = {r2_codd:.2f}\n{p_txt_codd}",
        transform=axes[1].transAxes, fontsize=8, verticalalignment="top",
    )

    sns.despine()
    fig.tight_layout()
    out = FIGURES_DIR / f"Figure1_Paradox.{FMT}"
    fig.savefig(out, dpi=DPI, format=FMT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name} | Sprint R²={r2_sprint:.3f} p={p_sprint:.4f} | "
          f"CODD R²={r2_codd:.3f} p={p_codd:.4f}")


def figure_2_coefficient_forest(iso) -> None:
    """Forest plot of standardised β coefficients across both outcomes."""
    coefs_sprint = fit_continuous_ols(iso, "Sprint_20m_Best", PREDICTORS)
    coefs_codd = fit_continuous_ols(iso, "CODD_Rel", PREDICTORS)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)

    # Predictor ordering: principal predictor on top
    order = PREDICTORS  # already principal first
    y_positions = np.arange(len(order))[::-1]   # top-down

    for ax, coefs, outcome_label, outcome_unit in [
        (axes[0], coefs_sprint, "20 m Sprint", "s"),
        (axes[1], coefs_codd, "Relative CODD", "%"),
    ]:
        for ypos, pred in zip(y_positions, order):
            entry = next(c for c in coefs if c["predictor"] == pred)
            colour = COLOR_PRINCIPAL if pred == PRINCIPAL_PREDICTOR else COLOR_NEUTRAL
            # 95% CI line + point estimate marker
            ax.plot(
                [entry["lower_95"], entry["upper_95"]],
                [ypos, ypos],
                color=colour, linewidth=1.6,
            )
            ax.plot(
                entry["estimate"], ypos,
                marker="o", markersize=7, color=colour, zorder=3,
            )
            # CI end-caps
            cap = 0.12
            for x in (entry["lower_95"], entry["upper_95"]):
                ax.plot([x, x], [ypos - cap, ypos + cap], color=colour, linewidth=1.2)
            # Inline numerical annotation
            ax.annotate(
                f"{entry['estimate']:+.3f}",
                xy=(entry["estimate"], ypos),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7, color=colour,
            )

        ax.axvline(0, color=COLOR_ZERO, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([PRETTY_PREDICTOR[p] for p in order])
        ax.set_xlabel(f"β per 1 SD (outcome in {outcome_unit})")
        ax.set_title(outcome_label, loc="left", fontweight="bold")
        ax.set_ylim(-0.5, len(order) - 0.5)

    sns.despine()
    fig.tight_layout()
    out = FIGURES_DIR / f"Figure2_CoefficientForest.{FMT}"
    fig.savefig(out, dpi=DPI, format=FMT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def figure_3_continuous_typology(iso) -> None:
    """Scatter sprint vs CODD with continuous torque colouring."""
    fig, ax = plt.subplots(figsize=(5.2, 5))

    torque = iso["Rel_Peak_Ext_Torque"].values
    t_min, t_max = torque.min(), torque.max()
    norm = plt.Normalize(vmin=t_min, vmax=t_max)

    scatter = ax.scatter(
        iso["Sprint_20m_Best"], iso["CODD_Rel"],
        c=torque, cmap=TORQUE_COLORMAP, norm=norm,
        s=60, alpha=0.9, edgecolors="white", linewidth=0.6,
    )

    sprint_med = iso["Sprint_20m_Best"].median()
    codd_med = iso["CODD_Rel"].median()
    ax.axvline(sprint_med, color=COLOR_ZERO, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axhline(codd_med, color=COLOR_ZERO, linestyle=":", linewidth=0.6, alpha=0.5)

    # Axes inverted: upper-left = fast & efficient
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel("20 m Sprint (s) ← faster")
    ax.set_ylabel("CODD Relative (%) ← more efficient")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Rel. Peak Ext. Torque (N·m·kg⁻¹)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    sns.despine()
    fig.tight_layout()
    out = FIGURES_DIR / f"Figure3_ContinuousTypology.{FMT}"
    fig.savefig(out, dpi=DPI, format=FMT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def supp_figure_s1_mechanism(iso) -> None:
    """Median-split bar charts (legacy Figure 2)."""
    variables = [
        ("Sprint_20m_Best", "20 m Sprint (s)"),
        ("HQ_Ratio", "H:Q Ratio (%)"),
        ("CODD_Rel", "CODD Relative (%)"),
    ]
    panel_labels = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 3))

    for ax, (var, ylabel), label in zip(axes, variables, panel_labels):
        sns.barplot(
            x="Group", y=var, hue="Group", data=iso, ax=ax,
            palette=GROUP_PALETTE, errorbar="sd", capsize=0.1,
            order=["High Torque", "Low Torque"],
            hue_order=["High Torque", "Low Torque"],
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(label, loc="left", fontweight="bold")

        high = iso.loc[iso["Group"] == "High Torque", var].dropna()
        low = iso.loc[iso["Group"] == "Low Torque", var].dropna()
        t_stat, p_val = stats.ttest_ind(high, low, equal_var=False)

        y_max = iso[var].max()
        y_range = iso[var].max() - iso[var].min()
        bracket_y = y_max + y_range * 0.15
        bar_x = [0, 1]
        ax.plot(bar_x, [bracket_y, bracket_y], color="k", linewidth=0.8)
        ax.plot([bar_x[0], bar_x[0]], [bracket_y - y_range * 0.02, bracket_y],
                color="k", linewidth=0.8)
        ax.plot([bar_x[1], bar_x[1]], [bracket_y - y_range * 0.02, bracket_y],
                color="k", linewidth=0.8)

        p_txt = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        ax.text(
            0.5, bracket_y + y_range * 0.02, p_txt,
            ha="center", va="bottom", fontsize=7,
        )
        ax.set_ylim(top=bracket_y + y_range * 0.15)

    sns.despine()
    fig.tight_layout()
    out = FIGURES_DIR / f"SupplementaryFigureS1_Mechanism.{FMT}"
    fig.savefig(out, dpi=DPI, format=FMT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def supp_figure_s2_typology_quadrant(iso) -> None:
    """Typology quadrant (legacy Figure 3 — categorical High vs Low)."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for group, color in GROUP_PALETTE.items():
        mask = iso["Group"] == group
        ax.scatter(
            iso.loc[mask, "Sprint_20m_Best"],
            iso.loc[mask, "CODD_Rel"],
            c=color, label=group, s=40, alpha=0.75,
            edgecolors="white", linewidth=0.5,
        )

    sprint_med = iso["Sprint_20m_Best"].median()
    codd_med = iso["CODD_Rel"].median()
    ax.axvline(sprint_med, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(codd_med, color="k", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.invert_xaxis()
    ax.invert_yaxis()

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    quadrant_kw = dict(fontsize=7, fontstyle="italic", alpha=0.7,
                       ha="center", va="center")
    ax.text(sprint_med + (x_min - sprint_med) * 0.5,
            codd_med + (y_min - codd_med) * 0.5,
            "Fast & Efficient", **quadrant_kw)
    ax.text(sprint_med + (x_max - sprint_med) * 0.5,
            codd_med + (y_min - codd_med) * 0.5,
            "Slow & Efficient", **quadrant_kw)
    ax.text(sprint_med + (x_min - sprint_med) * 0.5,
            codd_med + (y_max - codd_med) * 0.5,
            "Fast & Inefficient\n(Paradox)", **quadrant_kw)
    ax.text(sprint_med + (x_max - sprint_med) * 0.5,
            codd_med + (y_max - codd_med) * 0.5,
            "Slow & Inefficient", **quadrant_kw)

    ax.set_xlabel("20 m Sprint (s) ← faster")
    ax.set_ylabel("CODD Relative (%) ← more efficient")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    sns.despine()
    fig.tight_layout()
    out = FIGURES_DIR / f"SupplementaryFigureS2_TypologyQuadrant.{FMT}"
    fig.savefig(out, dpi=DPI, format=FMT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    df = load_data()
    iso = get_iso_sample(df).copy()

    median_val = iso["Rel_Peak_Ext_Torque"].median()
    iso["Group"] = np.where(
        iso["Rel_Peak_Ext_Torque"] >= median_val, "High Torque", "Low Torque"
    )

    n_high = (iso["Group"] == "High Torque").sum()
    n_low = (iso["Group"] == "Low Torque").sum()
    print(f"Isokinetic subsample n = {len(iso)} | "
          f"median torque = {median_val:.3f} | "
          f"High = {n_high}, Low = {n_low}\n")

    configure_journal_style()
    figure_1_paradox(iso)
    figure_2_coefficient_forest(iso)
    figure_3_continuous_typology(iso)
    supp_figure_s1_mechanism(iso)
    supp_figure_s2_typology_quadrant(iso)
