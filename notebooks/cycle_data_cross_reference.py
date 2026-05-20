"""Cross-reference the isokinetic subsample (n = 22) with the daily coaching-
staff wellness form responses, producing concurrent menstrual status and
contraceptive use per athlete.

The form was administered to the squads during the assessment window; this
script matches each iso athlete to her closest-date form response under a
two-tier name-matching strategy and writes both a personally-identifying
join (kept under ``data/``, gitignored) and an anonymised version keyed by
``iso_id`` (1..22) that is safe to commit alongside the analysis results.

Matching strategy:
    1. Exact match on the fully normalised name (accent fold + uppercase +
       collapsed whitespace).
    2. Fallback: token-subset match — flagged ``BY_SHORT_NAME`` for case-
       by-case review when the iso roster carries the full 4-token name but
       the form roster uses 2 tokens (e.g. first + maternal surname).

Outputs (idempotent across re-runs):
    results/cycle_data_cross_reference.txt    tracked — iso_id keyed, no names
    results/cycle_data_iso_match_anon.csv     tracked — iso_id keyed, no names
    data/cycle_data_iso_match.csv             gitignored — includes athlete names

Run from notebooks/:
    ../.venv/bin/python cycle_data_cross_reference.py
"""

from __future__ import annotations

import csv
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

sys.path.insert(0, str(SCRIPT_DIR))
from _common import RESULTS_DIR, build_file_header  # noqa: E402
from data_prep import get_iso_sample, load_data  # noqa: E402

FORM_CSV = DATA_DIR / "cycle_form_responses_2026-01-20_to_2026-02-16.csv"

FILE_HEADER = build_file_header({"Form CSV": FORM_CSV.name})


def normalize(s: str) -> str:
    """Strip accents, uppercase, and collapse whitespace."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    no_accents = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return " ".join(no_accents.strip().upper().split())


def name_tokens(s: str) -> frozenset[str]:
    """Tokenise a name into a frozenset for subset matching."""
    return frozenset(normalize(s).split())


def parse_form_date(s: str):
    try:
        clean = s.replace(" GMT-3", "").replace(" GMT+0", "")
        return datetime.strptime(clean, "%Y/%m/%d %I:%M:%S %p").date()
    except Exception:
        return None


def parse_test_date(s: str):
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except Exception:
        return None


def find_match(iso_name: str, form_by_full: dict, form_token_sets: list[tuple[frozenset, str]]):
    """Two-tier name matching.

    Returns ``(form_key, match_kind)`` or ``(None, None)``.

    1. Exact match on the full normalised name.
    2. Fallback: token-subset match — the iso tokens must be a superset of
       the form tokens, and the form must have at least 2 tokens to avoid
       single-token ambiguity. Returns ``BY_SHORT_NAME`` for a unique match
       and ``AMBIGUOUS_SHORT_NAME`` for multiple matches.
    """
    norm_full = normalize(iso_name)
    if norm_full in form_by_full:
        return norm_full, "EXACT"

    iso_tokens = name_tokens(iso_name)
    candidates = []
    for form_tokens, form_key in form_token_sets:
        if len(form_tokens) >= 2 and form_tokens.issubset(iso_tokens):
            candidates.append(form_key)

    if len(candidates) == 1:
        return candidates[0], "BY_SHORT_NAME"
    if len(candidates) > 1:
        return None, "AMBIGUOUS_SHORT_NAME"
    return None, None


def main():
    df = load_data()
    iso = get_iso_sample(df).reset_index(drop=True)

    form_by_full: dict[str, list[dict]] = defaultdict(list)

    with open(FORM_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw = r.get("Nome da atleta", "")
            name_norm = normalize(raw)
            if not name_norm:
                continue
            d = parse_form_date(r.get("Carimbo de data/hora", ""))
            if d is None:
                continue
            form_by_full[name_norm].append({
                "date": d,
                "phase": r.get("Em qual fase do ciclo você está?", "").strip(),
                "contraceptive_use": r.get("Faz uso de anticoncepcional?", "").strip(),
                "contraceptive_method": r.get("Qual método/contraceptivo você usa?", "").strip(),
            })

    form_token_sets = [(frozenset(k.split()), k) for k in form_by_full.keys()]

    rows = []
    for _, athlete_row in iso.iterrows():
        iso_name_raw = athlete_row["Name"]
        test_date_str = athlete_row["Start_Date"]
        test_date = parse_test_date(test_date_str)

        form_key, match_kind = find_match(iso_name_raw, form_by_full, form_token_sets)
        responses = form_by_full.get(form_key, []) if form_key else []

        if not responses:
            rows.append({
                "name": iso_name_raw,
                "test_date": test_date_str,
                "match_status": "NO_MATCH",
                "match_offset_days": None,
                "phase": "",
                "contraceptive_use": "",
                "contraceptive_method": "",
            })
            continue

        same_day = [r for r in responses if r["date"] == test_date]
        if same_day:
            same_day_flag = "_MULTI" if len(same_day) > 1 else ""
            match = same_day[0]
            offset = 0
            if match_kind == "EXACT":
                match_status = f"SAME_DAY{same_day_flag}"
            else:
                match_status = f"SAME_DAY_BY_SHORT_NAME{same_day_flag}"
        else:
            closest = min(responses, key=lambda r: abs((r["date"] - test_date).days))
            match = closest
            offset = (match["date"] - test_date).days
            if match_kind == "EXACT":
                match_status = f"OFFSET_{offset:+d}d"
            else:
                match_status = f"OFFSET_{offset:+d}d_BY_SHORT_NAME"

        rows.append({
            "name": iso_name_raw,
            "test_date": test_date_str,
            "match_status": match_status,
            "match_offset_days": offset,
            "phase": match["phase"],
            "contraceptive_use": match["contraceptive_use"],
            "contraceptive_method": match["contraceptive_method"],
        })

    matched = [r for r in rows if r["match_status"] != "NO_MATCH"]
    n_matched = len(matched)
    n_total = len(rows)

    phase_dist = Counter(r["phase"] for r in matched)
    contracept_dist = Counter(r["contraceptive_use"] for r in matched)
    method_dist = Counter(r["contraceptive_method"] for r in matched if r["contraceptive_method"])

    out_lines = ["=" * 80]
    out_lines.append("Cycle data cross-reference — isokinetic subsample (n = 22)")
    out_lines.append("=" * 80)
    out_lines.append("")
    out_lines.append(FILE_HEADER.rstrip())
    out_lines.append("")
    out_lines.append(f"Iso sample size: {n_total}")
    out_lines.append(f"Matched athletes (≥ 1 form response): {n_matched}/{n_total} "
                     f"({100 * n_matched / n_total:.1f}%)")
    out_lines.append("")
    out_lines.append("Cycle status distribution (among matched):")
    for phase, count in phase_dist.most_common():
        out_lines.append(f"  {count:>2}  {phase if phase else '(empty)'}")
    out_lines.append("")
    out_lines.append("Contraceptive use distribution (among matched):")
    for use, count in contracept_dist.most_common():
        out_lines.append(f"  {count:>2}  {use if use else '(empty)'}")
    out_lines.append("")
    out_lines.append("Contraceptive methods (among users):")
    for method, count in method_dist.most_common():
        out_lines.append(f"  {count:>2}  {method}")
    out_lines.append("")
    out_lines.append("-" * 80)
    out_lines.append("Per-athlete matching table (anonymised by iso_id, 1..22):")
    out_lines.append("-" * 80)
    out_lines.append(f"{'iso_id':>6}  {'Test date':<10}  {'Match status':<35}  "
                     f"{'Cycle phase':<28}  {'Contracept use':<14}  {'Method':<20}")
    for i, r in enumerate(rows, 1):
        out_lines.append(f"{i:>6}  {r['test_date']:<10}  {r['match_status']:<35}  "
                         f"{r['phase'][:28]:<28}  {r['contraceptive_use']:<14}  "
                         f"{r['contraceptive_method'][:20]:<20}")
    out_lines.append("")
    out_lines.append("Match-status codes:")
    out_lines.append("  SAME_DAY                  exact name; response on test day")
    out_lines.append("  SAME_DAY_BY_SHORT_NAME    short-name fallback; response on test day")
    out_lines.append("  OFFSET_+Nd / OFFSET_-Nd   exact name; closest response N days after/before")
    out_lines.append("  *_BY_SHORT_NAME           short-name fallback used")
    out_lines.append("  *_MULTI                   multiple same-day responses; first selected")
    out_lines.append("  NO_MATCH                  no form response found")

    summary_path = RESULTS_DIR / "cycle_data_cross_reference.txt"
    summary_path.write_text("\n".join(out_lines))
    print(f"Wrote (tracked, PII-free): {summary_path}")

    # CSV with names — kept under data/ (gitignored)
    join_path = DATA_DIR / "cycle_data_iso_match.csv"
    with open(join_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote (gitignored, contains names): {join_path}")

    # Anonymised CSV (iso_id only) — safe to commit
    anon_rows = []
    for i, r in enumerate(rows, 1):
        anon_rows.append({
            "iso_id": i,
            "test_date": r["test_date"],
            "match_status": r["match_status"],
            "match_offset_days": r["match_offset_days"],
            "phase": r["phase"],
            "contraceptive_use": r["contraceptive_use"],
            "contraceptive_method": r["contraceptive_method"],
        })
    anon_path = RESULTS_DIR / "cycle_data_iso_match_anon.csv"
    with open(anon_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(anon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(anon_rows)
    print(f"Wrote (tracked, PII-free): {anon_path}")


if __name__ == "__main__":
    main()
