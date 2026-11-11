"""Score the detector against the answer key. Evaluation code: it reads evals/, agents never do.

For a flag rule, runs the detector on every review week and reports
  - false alarms: flags per week in quiet weeks (Jun-Sep 2025, nothing planted)
  - recall: was each planted event flagged (in the top N an agent sees) in its first two weeks?
    was each real-world flu story flagged in any week of its window?
  - traps: was the noise cell or the Android decoy flagged?

    python -m evals.detector_check "abs(z) >= 3" "abs(z) >= 3 AND abs(r) >= 0.05"
"""
from __future__ import annotations

import datetime as dt
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from agents.db import Database
from agents.detector import TOP_N, build_flags, consistency, scores

EVALS = Path(__file__).resolve().parent
QUIET = pd.date_range("2025-06-08", "2025-09-28", freq="7D").date        # Sundays, nothing planted
REPLAY = pd.date_range("2025-10-12", "2026-02-01", freq="7D").date       # Sundays of the replay window


@lru_cache(maxsize=None)
def week_inputs(as_of: dt.date):
    """Scores for one week, computed once and reused by every rule."""
    db = Database(as_of)
    return scores(db), consistency(db)


def _scenarios() -> list[dict]:
    sc = yaml.safe_load((EVALS / "scenarios.yaml").read_text())["scenarios"]
    noise = pd.read_csv(EVALS / "noise_cases.csv").iloc[0]
    for s in sc:
        if s["id"] == "small_cell_noise":
            start = pd.Timestamp(noise.week_start).date()
            s["window"] = {"start": start, "end": start + dt.timedelta(days=6)}
            s["scope"] = {"market": [noise.market], "specialty": [noise.specialty], "platform": [noise.platform]}
    return sc


def _matches(flag: dict, scope: dict) -> bool:
    """A flag matches when, on every dimension the scenario names, the flag names a value in it.
    Lens E can't see platforms, so it matches on market and specialty only."""
    for dim in ["market", "specialty", "platform"]:
        if scope[dim] == "all" or (dim == "platform" and flag["lens"] == "E_consistency"):
            continue
        if flag[dim] not in scope[dim]:
            return False
    return True


def _weeks_touching(window: dict, first_n: int | None = None) -> list[dt.date]:
    """Review weeks (as Sundays) that overlap the window; optionally only the first n."""
    start = pd.Timestamp(window["start"]).date()
    end = pd.Timestamp(window["end"]).date() if window["end"] else REPLAY[-1]
    weeks = [w for w in REPLAY if w - dt.timedelta(days=6) <= end and w >= start]
    return weeks[:first_n] if first_n else weeks


def check(rule: str, top_n: int = TOP_N) -> dict:
    runs = {w: build_flags(*week_inputs(w), w, top_n, rule) for w in [*QUIET, *REPLAY]}
    quiet = [runs[w]["n_flags_before_cap"] for w in QUIET]
    replay = [runs[w]["n_flags_before_cap"] for w in REPLAY]

    scenarios = _scenarios()
    real = [s for s in scenarios if s["kind"] in ("planted", "real_world")]

    def explained(flag: dict, week: dt.date) -> bool:
        """The flag belongs to a real event active that week (so it isn't a trap's false alarm)."""
        return any(_matches(flag, s["scope"]) and week in _weeks_touching(s["window"]) for s in real)

    rows = []
    for s in scenarios:
        if s["kind"] == "expected":
            continue                       # scope is everything; covered by the per-week counts
        should_flag = s["kind"] in ("planted", "real_world")
        first_n = 2 if s["kind"] == "planted" else None
        hits = []
        for w in _weeks_touching(s["window"], first_n):
            for rank, f in enumerate(runs[w]["flags"], 1):
                if _matches(f, s["scope"]) and (should_flag or not explained(f, w)):
                    hits.append(f"{w:%m-%d} #{rank} {f['lens'][0]} {f['metric']}")
        rows.append({"scenario": s["id"], "kind": s["kind"],
                     "result": ("caught" if hits else "MISSED") if should_flag else ("FALSE ALARM" if hits else "ok"),
                     "first_hit": hits[0] if hits else ""})
    table = pd.DataFrame(rows)
    good = table.result.isin(["caught", "ok"]).sum()
    return {"rule": rule, "quiet_flags_per_week": round(sum(quiet) / len(quiet), 1),
            "replay_flags_per_week": round(sum(replay) / len(replay), 1),
            "scenarios_right": f"{good}/{len(table)}", "table": table,
            "per_week": pd.Series(replay, index=REPLAY, name="flags")}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for rule in sys.argv[1:] or ["abs(z) >= 3"]:
        out = check(rule)
        print(f"\n{rule}\n  flags per week: quiet {out['quiet_flags_per_week']}, replay {out['replay_flags_per_week']}"
              f"   scenarios right: {out['scenarios_right']}")
        print(out["table"].to_string(index=False))
