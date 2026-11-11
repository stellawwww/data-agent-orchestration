"""Pick the flag rule's numbers from the alert budget. Uses no answers.

Runs the detector over the quiet weeks (Jun 8 - Sep 28, 2025: nothing planted, so every flag
is a false alarm) for a grid of z cutoffs and bookings-impact floors, and reports flags per
quiet week. Choose the loosest rule whose count fits the budget (about 5 a week).

It never reads scenarios.yaml, effect_sizes.csv or noise_cases.csv, so choosing from this
table can't overfit to the planted events.

    python -m evals.alert_budget
"""
from __future__ import annotations

import pandas as pd

from evals.detector_check import QUIET, week_inputs
from agents.detector import build_flags

Z_CUTOFFS = [3.0, 3.5, 4.0, 4.5, 5.0]
IMPACT_FLOORS = [0, 10, 20, 50, 100]          # bookings per week


def rule(z: float, floor: float) -> str:
    return f"abs(z) >= {z} AND abs(bookings_impact) >= {floor}"


def flags_per_quiet_week(z_cutoffs=Z_CUTOFFS, floors=IMPACT_FLOORS) -> pd.DataFrame:
    """Rows: z cutoff; columns: impact floor; values: average flags per quiet week (before the cap)."""
    table = {}
    for z in z_cutoffs:
        table[z] = {}
        for floor in floors:
            counts = [build_flags(*week_inputs(w), w, rule=rule(z, floor))["n_flags_before_cap"] for w in QUIET]
            table[z][floor] = round(sum(counts) / len(counts), 1)
    out = pd.DataFrame(table).T
    out.index.name = "z cutoff"
    out.columns.name = "impact floor (bookings/wk)"
    return out


if __name__ == "__main__":
    print("Flags per quiet week (all false alarms). Budget: about 5.\n")
    print(flags_per_quiet_week().to_string())
