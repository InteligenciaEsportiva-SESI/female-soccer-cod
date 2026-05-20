"""Publication figures (TIFF 300 dpi).

Three figures keyed to the manuscript:

* Figure 1 — Efficiency Paradox: scatterplots of relative peak extensor
  torque vs 20-m sprint time (panel A) and vs relative CODD (panel B),
  each with a bivariate OLS fit and the bivariate R² annotated.
* Figure 2 — Mechanism: median-split bar charts comparing the High-Torque
  and Low-Torque groups on sprint, H:Q ratio and CODD (Welch's t-test
  significance bracket per panel).
* Figure 3 — Typology: quadrant scatter of sprint vs CODD with median
  reference lines and a quadrant labels overlay (axes inverted so "fast"
  and "efficient" point to the upper-left).

Outputs:
    figures/Figure1_Paradox.tiff
    figures/Figure2_Mechanism.tiff
    figures/Figure3_Typology.tiff

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
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import PALETTE  # noqa: E402
from data_prep import get_iso_sample, load_data  # noqa: E402

FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DPI = 300
FMT = "tiff"

# Local figure palette (median-split groups).
COLOR_HIGH = PALETTE["secondary"]   # deep red — High-Torque group / paradox
COLOR_LOW = "#95a5a6"                # neutral grey — Low-Torque group
COLOR_SPRINT = "#7f8c8d"             # neutral grey — sprint regression line
GROUP_PALETTE = {"High Torque": COLOR_HIGH, "Low Torque": COLOR_LOW}


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


def figure_1_paradox(iso) -> None:
    X = sm.add_constant(iso["Rel_Peak_Ext_Torque"])

    model_sprint = sm.OLS(iso["Sprint_20m_Best"], X).fit()
    r2_sprint = model_sprint.rsquared
    p_sprint = model_sprint.pvalues.iloc[1]

    model_codd = sm.OLS(iso["CODD_Rel"], X).fit()
    r2_codd = model_codd.rsquared
    p_codd = model_codd.pvalues.iloc[1]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A — Sprint 20m
    sns.regplot(
        x="Rel_Peak_Ext_Torque", y="Sprint_20m_Best", data=iso,
        ax=axes[0], color=COLOR_SPRINT, scatter_kws={"s": 30, "alpha": 0.7},
        line_kws={"linewidth": 1.5},
    )
    axes[0].invert_yaxis()  # lower sprint time = better
    axes[0].set_xlabel("Relative Peak Ext. Torque (N·m·kg⁻¹)")
    axes[0].set_ylabel("20 m Sprint (s)")
    axes[0].set_title("A", loc="left", fontweight="bold")
    p_txt_sprint = f"p = {p_sprint:.3f}" if p_sprint >= 0.001 else "p < 0.001"
    axes[0].text(
        0.05, 0.05, f"R² = {r2_sprint:.2f}\n{p_txt_sprint}",
        transform=axes[0].transAxes, fontsize=8, verticalalignment="bottom",
    )

    # Panel B — CODD relativo
    sns.regplot(
        x="Rel_Peak_Ext_Torque", y="CODD_Rel", data=iso,
        ax=axes[1], color=COLOR_HIGH, scatter_kws={"s": 30, "alpha": 0.7},
        line_kws={"linewidth": 1.5},
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


def figure_2_mechanism(iso) -> None:
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
    out = FIGURES_DIR / f"Figure2_Mechanism.{FMT}"
    fig.savefig(out, dpi=DPI, format=FMT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def figure_3_typology(iso) -> None:
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

    # Invert both axes so the upper-left quadrant is "fast & efficient".
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
    out = FIGURES_DIR / f"Figure3_Typology.{FMT}"
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
    figure_2_mechanism(iso)
    figure_3_typology(iso)
