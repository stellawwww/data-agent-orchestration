"""Score the detector against a world's answer key. Evaluation code: agents never import it.

For a flag rule, runs the detector on every review week and reports
  - false alarms: flags per week in quiet weeks (Jun-Sep 2025, nothing planted)
  - recall: was each planted event flagged (in the top N an agent sees) in its first two weeks?
    was each real-world flu story flagged in any week of its window?
  - traps: was the noise cell or a decoy release flagged?
and, for held-out worlds, a power curve: share of injected events caught, by size and type.

Worlds: dev (look freely), validation (aggregate scores only), test (sealed: needs
FINAL_SCORING=1 and prints aggregates only).

    python -m evals.detector_check --world dev "abs(z) >= 4 AND abs(bookings_impact) >= 20"
    python -m evals.detector_check --world validation
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from agents.db import Database
from agents.detector import FLAG_RULE, TOP_N, build_flags, consistency, scores

ROOT = Path(__file__).resolve().parents[1]
QUIET = pd.date_range("2025-06-08", "2025-09-28", freq="7D").date        # Sundays, nothing planted
REPLAY = pd.date_range("2025-10-12", "2026-02-01", freq="7D").date       # Sundays of the replay window
WORLDS = ["dev", "validation", "test"]


def data_dir(world: str) -> Path:
    return ROOT / "data" if world == "dev" else ROOT / "worlds" / world / "data"


def answers_dir(world: str) -> Path:
    """The answer key. The test world's is sealed until the final, one-time scoring run."""
    if world == "test" and os.environ.get("FINAL_SCORING") != "1":
        raise PermissionError("The test world is sealed. Score it once, at the end, with FINAL_SCORING=1.")
    return ROOT / "evals" if world == "dev" else ROOT / "worlds" / world / "answers"


@lru_cache(maxsize=None)
def week_inputs(as_of: dt.date, world: str = "dev"):
    """Scores for one week of one world, computed once and reused by every rule."""
    db = Database(as_of, data_dir(world))
    return scores(db), consistency(db)


def _scenarios(world: str) -> list[dict]:
    ans = answers_dir(world)
    sc = yaml.safe_load((ans / "scenarios.yaml").read_text())["scenarios"]
    noise = pd.read_csv(ans / "noise_cases.csv").iloc[0]
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


def check(rule: str = FLAG_RULE, top_n: int = TOP_N, world: str = "dev") -> dict:
    runs = {w: build_flags(*week_inputs(w, world), w, top_n, rule) for w in [*QUIET, *REPLAY]}
    quiet = [runs[w]["n_flags_before_cap"] for w in QUIET]
    replay = [runs[w]["n_flags_before_cap"] for w in REPLAY]

    scenarios = _scenarios(world)
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
    return {"rule": rule, "world": world, "quiet_flags_per_week": round(sum(quiet) / len(quiet), 1),
            "replay_flags_per_week": round(sum(replay) / len(replay), 1),
            "scenarios_right": f"{good}/{len(table)}", "table": table,
            "per_week": pd.Series(replay, index=REPLAY, name="flags")}


# The step each effect type moves, as measured by the counterfactual in effect_sizes.csv.
SIZE_COLUMNS = {"supply": ["slots_offered_change"], "demand": ["searches_change"],
                "funnel": ["click_rate_change", "booking_cvr_change"], "tracking": ["bookings_recorded_change"]}
SIZE_BUCKETS = [0, 0.05, 0.10, 0.20, 0.35, 10]
BUCKET_NAMES = ["under 5%", "5-10%", "10-20%", "20-35%", "35% or more"]


def power_curve(rule: str = FLAG_RULE, top_n: int = TOP_N, world: str = "validation") -> dict:
    """Share of injected events caught (in their first two weeks, inside the top N), by true size and type."""
    out = check(rule, top_n, world)
    sizes = pd.read_csv(answers_dir(world) / "effect_sizes.csv").set_index("id")
    planted = {s["id"]: s for s in _scenarios(world) if s["kind"] == "planted"}
    rows = []
    for sid, s in planted.items():
        cols = SIZE_COLUMNS[s["effect"]["type"]]
        size = sizes.loc[sid, cols].abs().max()
        caught = out["table"].set_index("scenario").loc[sid, "result"] == "caught"
        rows.append({"label": s.get("label", sid), "type": s["effect"]["type"], "size": size, "caught": caught})
    ev = pd.DataFrame(rows)
    ev["size_bucket"] = pd.cut(ev["size"], SIZE_BUCKETS, labels=BUCKET_NAMES, right=False)
    by_size = ev.groupby("size_bucket", observed=False).caught.agg(events="size", caught="sum")
    by_size["recall"] = (by_size.caught / by_size.events).round(2)
    by_type = ev.groupby("type").caught.agg(events="size", caught="sum")
    by_type["recall"] = (by_type.caught / by_type.events).round(2)
    return {"recall": round(ev.caught.mean(), 2), "events": len(ev), "by_size": by_size, "by_type": by_type,
            "quiet_flags_per_week": out["quiet_flags_per_week"], "replay_flags_per_week": out["replay_flags_per_week"],
            "traps": out["table"][out["table"].kind.isin(["decoy", "noise"])].result.value_counts().to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="dev", choices=WORLDS)
    ap.add_argument("rules", nargs="*", default=[FLAG_RULE])
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for rule in args.rules:
        if args.world == "dev":
            out = check(rule, world="dev")
            print(f"\n{rule}\n  flags per week: quiet {out['quiet_flags_per_week']}, replay {out['replay_flags_per_week']}"
                  f"   scenarios right: {out['scenarios_right']}")
            print(out["table"].to_string(index=False))
        else:     # held-out worlds: aggregates only, never the individual events
            pc = power_curve(rule, world=args.world)
            print(f"\n[{args.world}] {rule}\n  recall {pc['recall']} on {pc['events']} injected events; "
                  f"flags per week: quiet {pc['quiet_flags_per_week']}, replay {pc['replay_flags_per_week']}; traps {pc['traps']}")
            print(pc["by_size"].to_string())
            print(pc["by_type"].to_string())
