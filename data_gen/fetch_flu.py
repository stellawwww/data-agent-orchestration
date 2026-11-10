"""Download real weekly flu activity (CDC ILINet) for each market's region.

Source: CMU Delphi Epidata API, `fluview` endpoint, which republishes CDC ILINet.
Output: data/flu_activity.csv (one row per region per week) and a raw JSON snapshot,
so the generator never depends on the API being up.

Run once:  python data_gen/fetch_flu.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.delphi.cmu.edu/epidata/fluview/"
FIRST_EPIWEEK = 202352   # a few weeks before the data starts, for interpolation
LAST_EPIWEEK = 202607    # a few weeks after it ends


def mmwr_week_start(year: int, week: int) -> dt.date:
    """Sunday that starts CDC (MMWR) epiweek `week` of `year`.

    Week 1 is the first Sunday-to-Saturday week with at least 4 days in the new year.
    """
    jan1 = dt.date(year, 1, 1)
    days_since_sunday = (jan1.weekday() + 1) % 7      # Monday=0 in Python
    if days_since_sunday <= 3:
        week1 = jan1 - dt.timedelta(days=days_since_sunday)
    else:
        week1 = jan1 + dt.timedelta(days=7 - days_since_sunday)
    return week1 + dt.timedelta(weeks=week - 1)


def main() -> None:
    world = yaml.safe_load((ROOT / "config/world.yaml").read_text())
    regions = sorted({m["flu_region"] for m in world["markets"].values()})

    url = f"{API}?regions={','.join(regions)}&epiweeks={FIRST_EPIWEEK}-{LAST_EPIWEEK}"
    req = urllib.request.Request(url, headers={"User-Agent": "data-agent-orchestration/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    if payload.get("result") != 1:
        raise SystemExit(f"Delphi API error: {payload.get('message')}")

    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cdc_fluview.json").write_text(json.dumps(payload, indent=1))

    rows = []
    for r in payload["epidata"]:
        pct = r["wili"] if r["wili"] is not None else r["ili"]
        if pct is None:
            continue
        year, week = divmod(r["epiweek"], 100)
        rows.append({
            "week_start": mmwr_week_start(year, week).isoformat(),
            "epiweek": r["epiweek"],
            "region": r["region"],
            "wili": round(pct / 100, 6),     # percent -> decimal
        })
    rows.sort(key=lambda x: (x["region"], x["week_start"]))

    out = ROOT / "data/flu_activity.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["week_start", "epiweek", "region", "wili"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    counts = {g: sum(1 for x in rows if x["region"] == g) for g in regions}
    print(f"wrote {len(rows)} rows to {out.relative_to(ROOT)}: {counts}")


if __name__ == "__main__":
    main()
