"""
Shared data loading and variable derivation for female-soccer-cod project.

Usage:
    from data_prep import load_data, get_field_sample, get_iso_sample, RELIABILITY_PAIRS
"""

import pandas as pd
import numpy as np

# Path relative to notebooks/ directory
DATA_PATH = "../data/Database_Soccer_ana02.csv"

# Columns that are identifiers, not numeric measures
ID_COLS = [
    "Order", "Id", "Start_Date", "Category", "Name",
    "Birth_Date", "Position", "Dominance",
    "Tactical Position Abbreviations",
]

# Trial pairs for reliability analysis (label, trial_1, trial_2)
RELIABILITY_PAIRS = [
    ("Sprint 5m", "Sprint_5m_1", "Sprint_5m_2"),
    ("Sprint 10m", "Sprint_10m_1", "Sprint_10m_2"),
    ("Sprint 20m", "Sprint_20m_1", "Sprint_20m_2"),
    ("ZigZag", "ZigZag_1", "ZigZag_2"),
    ("Squat Jump", "SJ_1", "SJ_2"),
    ("CMJ", "CMJ_1", "CMJ_2"),
    ("CMJ Free Arms", "CMJ_Free_1", "CMJ_Free_2"),
    ("Horizontal Jump", "Horiz_Jump_1", "Horiz_Jump_2"),
    ("RSI 30cm", "DJ30_RSI_1", "DJ30_RSI_2"),
    ("RSI 40cm", "DJ40_RSI_1", "DJ40_RSI_2"),
]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert European decimal format (comma) to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "."), errors="coerce"
    )


def load_data() -> pd.DataFrame:
    """Load CSV, clean column names, convert numeric columns.

    Returns a DataFrame with all original + derived variables.
    """
    df = pd.read_csv(DATA_PATH, encoding="latin1", sep=";")
    df.columns = df.columns.str.strip()

    # Convert all measurement columns to numeric
    numeric_cols = [c for c in df.columns if c not in ID_COLS]
    for col in numeric_cols:
        df[col] = _to_numeric(df[col])

    # --- Derived variables (computed once, used everywhere) ---
    derived = pd.DataFrame(index=df.index)

    # Best trial selections (lower = better for time, higher = better for jump/RSI)
    derived["Sprint_20m_Best"] = df[["Sprint_20m_1", "Sprint_20m_2"]].min(axis=1)
    derived["ZigZag_Best"] = df[["ZigZag_1", "ZigZag_2"]].min(axis=1)
    derived["SJ_Max"] = df[["SJ_1", "SJ_2"]].max(axis=1)
    derived["CMJ_Max"] = df[["CMJ_1", "CMJ_2"]].max(axis=1)
    derived["CMJ_Free_Max"] = df[["CMJ_Free_1", "CMJ_Free_2"]].max(axis=1)
    derived["Horiz_Jump_Max"] = df[["Horiz_Jump_1", "Horiz_Jump_2"]].max(axis=1)

    # RSI: use DJ30 only as primary metric
    derived["RSI_DJ30"] = df[["DJ30_RSI_1", "DJ30_RSI_2"]].max(axis=1)

    # Change of Direction Deficit (relative)
    derived["CODD_Rel"] = (
        (derived["ZigZag_Best"] - derived["Sprint_20m_Best"])
        / derived["Sprint_20m_Best"]
        * 100
    )

    # Isokinetic bilateral means (60 deg/s)
    derived["Rel_Peak_Ext_Torque"] = df[
        ["PT_Ext_R_60_Rel", "PT_Ext_L_60_Rel"]
    ].mean(axis=1)
    derived["HQ_Ratio"] = df[["HQ_Ratio_R_60", "HQ_Ratio_L_60"]].mean(axis=1)

    # Drop any pre-existing columns that we recompute to avoid duplicates
    overlap = df.columns.intersection(derived.columns)
    if len(overlap):
        df = df.drop(columns=overlap)

    df = pd.concat([df, derived], axis=1)
    return df


def get_field_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Field sample: athletes with valid sprint and agility data (n~43)."""
    return df.dropna(subset=["Sprint_20m_1", "ZigZag_1"]).copy()


def get_iso_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Isokinetic subsample: athletes with valid torque + field data (n~22)."""
    return df.dropna(
        subset=["Sprint_20m_Best", "CODD_Rel", "Rel_Peak_Ext_Torque", "HQ_Ratio"]
    ).copy()
