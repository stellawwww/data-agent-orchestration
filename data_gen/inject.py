"""Random events for the held-out worlds (validation, test).

The dev world has 4 hand-written events that we have already seen and designed around. A
held-out world instead gets ~30 events drawn at random: random type, size, scope, start date
and duration. Their true sizes are measured by the same counterfactual as dev
(`World.effect_sizes`), so we can score recall and draw a power curve (effect size -> share
of events caught).

Each event is a scenario dict in the format `World` already applies:
  supply    {"type": "supply", "providers": 0.7}                      providers x 0.7
  funnel    {"type": "funnel", "click_rate": 1.1, "booking_cvr": 0.9}
  demand    {"type": "demand", "searches": 1.3, "booking_cvr": 0.95}  (cvr: diluted traffic)
  tracking  {"type": "tracking", "recorded_bookings": 0.2}            app records 20% of bookings

Split of work: `draw_effect` decides WHAT happens (type, size, how long, whether it gets
logged). The code below decides WHERE and WHEN (scope, start date) and keeps events apart.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np

N_EVENTS = 30
N_DECOYS = 4               # change_log entries with no effect behind them
MIN_DAYS_BEFORE_END = 7    # an event must start at least a week before the data ends
FOREVER = dt.date(2100, 1, 1)   # end date used for permanent events when checking overlap


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A size between lo and hi, spread evenly on a log scale (as many 5-10% as 20-40% moves)."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# Share of injected events by kind (from Stella's marketplace experience, 2026-11-11):
# small app/funnel bugs about monthly, big ones about twice a year; large providers often pause and
# sales wins them back, some churn for good; nobody reaches out to small providers.
# (A held-out world packs ~30 events into 4 months, far more than reality, to get enough
# events for a power curve. The mix, not the rate, is what should be realistic.)
EVENT_MIX = {
    "app_small_bug": 0.30,       # funnel: click or CVR -2..-3%, fixed next release (3-21 days)
    "app_big_bug": 0.08,         # funnel: -10..-20% (about 15% is a big bug), hotfixed (2-10 days)
    "supply_big_paused": 0.14,   # one large provider pauses: -15..-50% providers in a cell, back in 2-6 weeks
    "supply_big_churn": 0.06,    # one large provider leaves for good
    "supply_small_loss": 0.12,   # small providers leave (-3..-12%), no sales outreach, permanent
    "campaign": 0.18,            # demand: searches +10..+60%, CVR diluted, 2-5 weeks
    "tracking_bug": 0.12,        # app records 5-60% of bookings for 1-3 days
}

# Supply events are about providers in one place, so they get a narrower scope than SCOPES["supply"].
LABEL_SCOPES = {"supply_big_paused": ["cell"], "supply_big_churn": ["cell"], "supply_small_loss": ["cell", "market"]}


def draw_effect(rng: np.random.Generator) -> tuple[str, dict, int | None, bool]:
    """Draw WHAT happens in one event.

    Returns (label, effect, duration_days, logged):
      label          short name for the kind of event, e.g. "supply_loss", "app_cvr_bug",
                     "campaign", "tracking_bug". Used for the summary and to pick a scope.
      effect         dict in one of the formats in the module docstring.
      duration_days  how long it lasts; None = permanent (until the end of the data).
      logged         whether the team that caused it wrote it in change_log. Ignored for supply:
                     provider changes don't go in the product change log.

    Helpers: rng.random() for a coin flip, rng.choice([...]) to pick, rng.integers(lo, hi),
    log_uniform(rng, lo, hi) for a size.
    """
    label = str(rng.choice(list(EVENT_MIX), p=list(EVENT_MIX.values())))
    drop = lambda lo, hi: 1 - log_uniform(rng, lo, hi)            # e.g. drop(0.05, 0.5) -> x0.5 .. x0.95

    if label in ("app_small_bug", "app_big_bug"):
        size = drop(0.02, 0.03) if label == "app_small_bug" else drop(0.10, 0.20)
        step = str(rng.choice(["click_rate", "booking_cvr", "offsetting"], p=[0.4, 0.4, 0.2]))
        effect = ({"type": "funnel", "click_rate": 2 - size, "booking_cvr": size} if step == "offsetting"
                  else {"type": "funnel", step: size})              # offsetting: more clicks, fewer book
        days = int(rng.integers(3, 22)) if label == "app_small_bug" else int(rng.integers(2, 11))
        return label, effect, days, rng.random() < 0.6               # the release is usually logged

    if label.startswith("supply"):
        big = label != "supply_small_loss"
        effect = {"type": "supply", "providers": drop(0.15, 0.50) if big else drop(0.03, 0.12)}
        days = int(rng.integers(14, 43)) if label == "supply_big_paused" else None   # sales wins big ones back
        return label, effect, days, False

    if label == "campaign":
        effect = {"type": "demand", "searches": log_uniform(rng, 1.10, 1.60), "booking_cvr": drop(0.03, 0.15)}
        return label, effect, int(rng.integers(14, 36)), rng.random() < 0.5

    effect = {"type": "tracking", "recorded_bookings": log_uniform(rng, 0.05, 0.60)}   # tracking_bug
    return label, effect, int(rng.integers(1, 4)), rng.random() < 0.1    # nobody knows yet, so rarely logged


# Where an event can happen depends on its type. Supply and demand have no platform
# (providers and patients aren't per platform); funnel and tracking problems usually are.
SCOPES = {
    "supply": ["cell", "specialty", "market"],
    "demand": ["cell", "market", "specialty"],
    "funnel": ["platform", "cell_platform", "specialty_platform"],
    "tracking": ["platform", "market_platform"],
}

CHANGE_LOG = {   # how a logged event reads in change_log (event_type, description template)
    "funnel": ("product_launch", "Search and profile page update ({where})"),
    "funnel_app": ("app_release", "{platform} app {version} (UI changes)"),
    "demand": ("marketing", "Paid campaign ({where})"),
    "tracking": ("app_release", "{platform} app {version} (analytics SDK update)"),
}


def _scope(rng, etype: str, world: dict, label: str = "") -> dict:
    markets, specs, plats = list(world["markets"]), list(world["specialties"]), list(world["platforms"])
    pick = lambda xs: [str(rng.choice(xs))]
    level = str(rng.choice(LABEL_SCOPES.get(label, SCOPES[etype])))
    return {
        "cell": {"market": pick(markets), "specialty": pick(specs), "platform": "all"},
        "specialty": {"market": "all", "specialty": pick(specs), "platform": "all"},
        "market": {"market": pick(markets), "specialty": "all", "platform": "all"},
        "platform": {"market": "all", "specialty": "all", "platform": pick(plats)},
        "cell_platform": {"market": pick(markets), "specialty": pick(specs), "platform": pick(plats)},
        "specialty_platform": {"market": "all", "specialty": pick(specs), "platform": pick(plats)},
        "market_platform": {"market": pick(markets), "specialty": "all", "platform": pick(plats)},
    }[level]


def _overlaps(a: dict, b: dict) -> bool:
    """Same cells at the same time (so a flag could belong to either event)."""
    for dim in ["market", "specialty", "platform"]:
        if a["scope"][dim] != "all" and b["scope"][dim] != "all" and not set(a["scope"][dim]) & set(b["scope"][dim]):
            return False
    a0, a1 = a["window"]["start"], a["window"]["end"] or FOREVER
    b0, b1 = b["window"]["start"], b["window"]["end"] or FOREVER
    return a0 <= b1 + dt.timedelta(days=7) and b0 <= a1 + dt.timedelta(days=7)   # 1-week buffer


def _where(scope: dict) -> str:
    parts = [", ".join(v) for v in scope.values() if v != "all"]
    return " / ".join(parts) if parts else "all"


CATEGORY = {"supply": "supply", "demand": "traffic_quality", "funnel": "product_change", "tracking": "instrumentation"}


def random_scenarios(seed: int, world: dict, replay_window: dict, n: int = N_EVENTS) -> list[dict]:
    """n non-overlapping random events plus N_DECOYS logged non-events, as scenario dicts."""
    rng = np.random.default_rng(seed)
    start = dt.date.fromisoformat(str(replay_window["start"]))
    last = dt.date.fromisoformat(str(replay_window["end"])) - dt.timedelta(days=MIN_DAYS_BEFORE_END)
    days = (last - start).days
    events: list[dict] = []
    tries = 0
    while len(events) < n and tries < 5000:
        tries += 1
        label, effect, duration, logged = draw_effect(rng)
        etype = effect["type"]
        s = start + dt.timedelta(days=int(rng.integers(0, days + 1)))
        end = None if duration is None else s + dt.timedelta(days=duration - 1)
        ev = {"id": f"inj_{len(events) + 1:02d}_{label}", "kind": "planted", "category": CATEGORY[etype],
              "label": label, "window": {"start": s, "end": end}, "scope": _scope(rng, etype, world, label),
              "effect": effect}
        if any(_overlaps(ev, e) for e in events):
            continue
        if logged and etype != "supply":
            key = "funnel_app" if etype == "funnel" and ev["scope"]["platform"] != "all" \
                and ev["scope"]["platform"][0] != "Web" else etype
            event_type, tpl = CHANGE_LOG[key]
            platform = ev["scope"]["platform"][0] if ev["scope"]["platform"] != "all" else ""
            version = f"{int(rng.integers(5, 8))}.{int(rng.integers(0, 10))}"
            ev["change_log"] = {"event_type": event_type,
                                "description": tpl.format(where=_where(ev["scope"]), platform=platform, version=version)}
        events.append(ev)

    for k in range(N_DECOYS):     # logged releases where nothing measurable changed
        platform = str(rng.choice(list(world["platforms"])))
        s = start + dt.timedelta(days=int(rng.integers(0, days + 1)))
        events.append({"id": f"decoy_{k + 1}", "kind": "decoy", "category": "no_effect", "label": "decoy",
                       "window": {"start": s, "end": s}, "scope": {"market": "all", "specialty": "all", "platform": [platform]},
                       "effect": {"type": "none"},
                       "change_log": {"event_type": "app_release",
                                      "description": f"{platform} app {int(rng.integers(5, 8))}.{int(rng.integers(0, 10))} (bug fixes)"}})
    return events


def summarize(events: list[dict]) -> str:
    """Aggregate view only (safe to print for the sealed test world): counts, never the events."""
    planted = [e for e in events if e["kind"] == "planted"]
    by_label = Counter(e["label"] for e in planted)
    logged = sum("change_log" in e for e in planted)
    lasting = Counter("permanent" if e["window"]["end"] is None else "temporary" for e in planted)
    return (f"{len(planted)} injected events, {len(events) - len(planted)} decoys; "
            f"{logged} logged in change_log; {dict(lasting)}; by label: {dict(by_label)}")
