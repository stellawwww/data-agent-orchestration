"""SQL anomaly detector: which cells moved this week? No LLM.

Runs on Monday morning for the review week Monday..Sunday (`as_of` is that Sunday) and
returns a ranked list of flags for the agents to investigate. It favors recall: a missed
issue can't be recovered downstream, while a false alarm can be ruled out by an agent.

Every metric is scored at every level of the hierarchy (total, platform, market,
specialty, and their combinations) through four lenses:

  A  YoY-adjusted shift   this week's change vs last year, minus the same comparison for the
                          prior 4 weeks. Seasonality and holidays that repeat each year cancel.
  B  vs peers             this cell vs its peers (iOS vs the other platforms, NYC vs the other
                          markets), this week vs the prior 4 weeks. Shocks that hit every peer
                          (flu, holidays, seasonality) cancel.
  D  funnel steps         A and B run on every step of the KPI tree, not only bookings, so
                          offsetting changes inside the funnel still show.
  E  data consistency     daily, bookings recorded by the app (daily_funnel) vs by the
                          scheduling system (daily_supply). The two should agree.

A and B produce a residual r (log ratio; r = -0.10 is about -10%). Noise is the robust
spread of the same cell's r over the previous 26 weeks (1.4826 x MAD), and z = r / sigma.
Each move is also priced in bookings per week (bookings_impact). A flag must be both real
(z) and material (impact); flags are ranked by impact. Lens E flags always come first.
"""
from __future__ import annotations

import datetime as dt
import math

import duckdb
import pandas as pd

from pathlib import Path

from agents.db import DATA_DIR, Database

TOP_N = 15                 # flags passed to the agents each week
E_MIN_GAP = 0.02           # lens E: daily gap between the two booking systems worth flagging
MIN_HISTORY_WEEKS = 13     # weeks of residuals needed before a cell's sigma is trusted

# A row of `scores()` is flagged when this SQL condition is true. Columns you can use:
#   lens         'A_yoy_shift' or 'B_vs_peers'
#   metric       searches, availability_rate, click_rate, booking_cvr, bookings, slots_offered, fill_rate
#   r            residual, log ratio (r = -0.10 is about a 10% drop beyond what was expected)
#   z            r / sigma
#   base_volume  weekly volume behind the metric over the prior 4 weeks
#                (searches for availability_rate, profile views for booking_cvr, the count itself for counts)
#   bookings_impact  bookings per week this move adds or costs: (e^r - 1) x the cell's average
#                weekly bookings over the prior 4 weeks. The business unit: bookings, so revenue.
# Rule shape (decided): a z gate ("is the move real?") plus a bookings-impact floor ("is it worth
# acting on?"). Both numbers are chosen from flags per quiet week (evals/alert_budget.py), not
# from the answer key. Flags are ranked by |bookings_impact|.
# Chosen 2026-11-11: about 4 flags per quiet week. z >= 4 because ~2,300 tests run each week;
# a low floor (20 bookings, ~1.3% of Phoenix's week) keeps small-market problems visible.
FLAG_RULE = "abs(z) >= 4 AND abs(bookings_impact) >= 20"

DIMS = ["market", "specialty", "platform"]

METRICS = {                # name: (numerator, denominator); None = a count, not a rate
    "searches": ("searches", None),
    "availability_rate": ("swa", "searches"),
    "click_rate": ("pv", "swa"),
    "booking_cvr": ("bookings", "pv"),
    "bookings": ("bookings", None),
    "slots_offered": ("slots_offered", None),
    "fill_rate": ("slots_booked", "slots_offered"),
}

# Weekly totals at every level. 'all' means summed over that dimension. Supply has no platform.
CELLS_SQL = """
SELECT f.* EXCLUDE (days), s.slots_offered, s.slots_booked
FROM (
  SELECT date_trunc('week', date)::DATE AS week,
         coalesce(market, 'all') AS market, coalesce(specialty, 'all') AS specialty,
         coalesce(platform, 'all') AS platform, count(DISTINCT date) AS days,
         sum(searches) AS searches, sum(searches_with_availability) AS swa,
         sum(profile_views) AS pv, sum(bookings) AS bookings
  FROM daily_funnel
  GROUP BY GROUPING SETS (
    (date_trunc('week', date)),
    (date_trunc('week', date), platform), (date_trunc('week', date), market), (date_trunc('week', date), specialty),
    (date_trunc('week', date), market, specialty), (date_trunc('week', date), specialty, platform),
    (date_trunc('week', date), market, platform), (date_trunc('week', date), market, specialty, platform))
) f
LEFT JOIN (
  SELECT date_trunc('week', date)::DATE AS week,
         coalesce(market, 'all') AS market, coalesce(specialty, 'all') AS specialty, 'all' AS platform,
         sum(slots_offered) AS slots_offered, sum(slots_booked) AS slots_booked
  FROM daily_supply
  GROUP BY GROUPING SETS (
    (date_trunc('week', date)), (date_trunc('week', date), market),
    (date_trunc('week', date), specialty), (date_trunc('week', date), market, specialty))
) s USING (week, market, specialty, platform)
WHERE f.days = 7                                   -- complete weeks only
"""


def _long_sql() -> str:
    """One row per cell x week x metric, as numerator n and denominator d (d = 1 for counts)."""
    parts = []
    for name, (num, den) in METRICS.items():
        parts.append(
            f"SELECT week, market, specialty, platform, '{name}' AS metric, {den is not None} AS is_rate, "
            f"{num}::DOUBLE AS n, {den or '1'}::DOUBLE AS d, bookings::DOUBLE AS bk "
            f"FROM cells WHERE {num} IS NOT NULL")
    return "\nUNION ALL ".join(parts)


def _scores_sql(review_week: dt.date) -> str:
    return f"""
WITH cells AS ({CELLS_SQL}),
base AS ({_long_sql()}),
-- prior 4 weeks (sums, so rates are ratios of sums)
w AS (
  SELECT *, sum(n) OVER p4 AS n4, sum(d) OVER p4 AS d4, sum(bk) OVER p4 AS bk4
  FROM base
  WINDOW p4 AS (PARTITION BY market, specialty, platform, metric ORDER BY week
                ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)
),
-- same weeks one year (52 weeks) earlier
y AS (
  SELECT *, lag(n, 52) OVER c AS n_ly, lag(d, 52) OVER c AS d_ly,
            lag(n4, 52) OVER c AS n4_ly, lag(d4, 52) OVER c AS d4_ly
  FROM w WINDOW c AS (PARTITION BY market, specialty, platform, metric ORDER BY week)
),
lens_a AS (
  SELECT week, market, specialty, platform, metric, 'A_yoy_shift' AS lens, 'last year' AS compared_with,
         n / d AS value, n4 / d4 AS value_prev4, n_ly / d_ly AS value_compared,
         ln(nullif(n / d, 0) / nullif(n_ly / d_ly, 0))
           - ln(nullif(n4 / d4, 0) / nullif(n4_ly / d4_ly, 0)) AS r,
         CASE WHEN is_rate THEN d4 ELSE n4 END / 4 AS base_volume,
         bk4 / 4 AS weekly_bookings
  FROM y
),
-- peers = parent minus this cell, e.g. NYC iOS vs NYC (Android + Web)
lens_b AS (
  SELECT c.week, c.market, c.specialty, c.platform, c.metric, 'B_vs_peers' AS lens,
         CASE WHEN c.platform <> 'all' AND p.platform = 'all' THEN 'other platforms' ELSE 'other markets' END AS compared_with,
         c.n / c.d AS value, c.n4 / c.d4 AS value_prev4,
         (p.n - c.n) / CASE WHEN c.is_rate THEN p.d - c.d ELSE 1 END AS value_compared,
         ln(nullif(c.n / c.d, 0) / nullif((p.n - c.n) / CASE WHEN c.is_rate THEN p.d - c.d ELSE 1 END, 0))
           - ln(nullif(c.n4 / c.d4, 0) / nullif((p.n4 - c.n4) / CASE WHEN c.is_rate THEN p.d4 - c.d4 ELSE 4 END, 0)) AS r,
         CASE WHEN c.is_rate THEN c.d4 ELSE c.n4 END / 4 AS base_volume,
         c.bk4 / 4 AS weekly_bookings
  FROM w c JOIN w p
    ON p.week = c.week AND p.metric = c.metric AND p.specialty = c.specialty
   AND ((c.platform <> 'all' AND p.platform = 'all' AND p.market = c.market)
     OR (c.market <> 'all' AND p.market = 'all' AND p.platform = c.platform))
),
resid AS (SELECT * FROM lens_a UNION ALL SELECT * FROM lens_b),
noise AS (
  SELECT *, 1.4826 * mad(r) OVER h AS sigma, count(r) OVER h AS history_weeks
  FROM resid
  WINDOW h AS (PARTITION BY market, specialty, platform, metric, lens, compared_with ORDER BY week
               ROWS BETWEEN 26 PRECEDING AND 1 PRECEDING)
)
SELECT *, r / nullif(sigma, 0) AS z,
       (exp(r) - 1) * weekly_bookings AS bookings_impact   -- bookings/week this move adds or costs
FROM noise
WHERE week = DATE '{review_week.isoformat()}' AND r IS NOT NULL AND history_weeks >= {MIN_HISTORY_WEEKS}
"""


def review_week(as_of: dt.date) -> dt.date:
    """Monday that starts the review week ending on `as_of` (a Sunday)."""
    if as_of.weekday() != 6:
        raise ValueError(f"as_of must be a Sunday, the last day of the review week (got {as_of:%A})")
    return as_of - dt.timedelta(days=6)


def scores(db: Database) -> pd.DataFrame:
    """Lens A and B scores for every cell x metric in the review week (unfiltered)."""
    return db.query(_scores_sql(review_week(db.as_of)))


def consistency(db: Database) -> pd.DataFrame:
    """Lens E: days in the review week where the two booking systems disagree."""
    start = review_week(db.as_of)
    return db.query(f"""
WITH j AS (
  SELECT s.date, s.market, s.specialty, f.bookings AS app_bookings, s.slots_booked AS scheduler_bookings
  FROM daily_supply s
  JOIN (SELECT date, market, specialty, sum(bookings) AS bookings FROM daily_funnel GROUP BY ALL) f
    USING (date, market, specialty)
  WHERE s.date BETWEEN DATE '{start}' AND DATE '{db.as_of}'
),
g AS (
  SELECT date, coalesce(market, 'all') AS market, coalesce(specialty, 'all') AS specialty,
         sum(app_bookings) AS app_bookings, sum(scheduler_bookings) AS scheduler_bookings
  FROM j GROUP BY GROUPING SETS ((date), (date, market), (date, specialty), (date, market, specialty))
)
, d AS (   -- share of the scheduler's bookings the app did not record (negative = app recorded more)
  SELECT *, (scheduler_bookings - app_bookings) / greatest(scheduler_bookings, 1) AS share_missing FROM g
)
SELECT market, specialty, 'all' AS platform,
       count(*) FILTER (WHERE abs(share_missing) >= {E_MIN_GAP}) AS days_with_gap,
       string_agg(strftime(date, '%a %m-%d'), ', ' ORDER BY date)
         FILTER (WHERE abs(share_missing) >= {E_MIN_GAP}) AS gap_days,
       max(share_missing) AS max_share_missing,
       sum(app_bookings) AS app_bookings, sum(scheduler_bookings) AS scheduler_bookings
FROM d GROUP BY ALL
HAVING days_with_gap > 0
ORDER BY max_share_missing DESC
""")


def _covers(a: pd.Series, b: pd.Series) -> bool:
    """True if cell a contains cell b and is coarser (e.g. iOS total contains NYC iOS)."""
    return all(a[c] == "all" or a[c] == b[c] for c in DIMS) and any(a[c] == "all" and b[c] != "all" for c in DIMS)


OWNS = 0.5    # a child whose share x move accounts for this much of its parent's move "owns" it


def _one_level_per_issue(flags: pd.DataFrame, same: list[str], size: str, weight: str) -> pd.DataFrame:
    """The same move shows at several levels (NYC iOS, iOS, total). Keep one level.

    Among flags that agree on `same`:
    1. A parent is dropped when one flagged child accounts for at least OWNS of its move,
       measured as the child's share of the parent (`weight`) x the child's move (`size`).
       Chicago derm is 21% of derm slots and fell 42%: 0.21 x 42% = 8.8%, all of derm's -8.2%.
       The problem is in Chicago, so keep Chicago derm and drop all derm.
    2. Any flag that still has a kept parent is dropped: the parent summarizes a broad move.
       The iOS tracking bug hits every market alike, so iOS stays and NYC iOS goes.
    """
    flags = flags.reset_index(drop=True)
    owned = set()
    for _, grp in flags.groupby(same, dropna=False):
        for j, parent in grp.iterrows():
            for i, child in grp.iterrows():
                if i != j and _covers(parent, child) and parent[weight] > 0:
                    share = child[weight] / parent[weight]
                    if share * abs(child[size]) >= OWNS * abs(parent[size]):
                        owned.add(j)
    summarized = set()
    for _, grp in flags.groupby(same, dropna=False):
        for i, child in grp.iterrows():
            if any(j not in owned and i != j and _covers(parent, child) for j, parent in grp.iterrows()):
                summarized.add(i)
    return flags.drop(index=list(owned | summarized))


def select_flags(s: pd.DataFrame, rule: str = FLAG_RULE) -> pd.DataFrame:
    """Apply the flag rule, then remove the echoes of the same issue."""
    f = duckdb.query_df(s, "s", f"SELECT * FROM s WHERE {rule}").df()
    if f.empty:
        return f
    f["direction"] = f.r.gt(0).map({True: "up", False: "down"})
    # B: when one peer moves, the others look like they moved the opposite way.
    # Keep only the odd one out (largest |r|) among the peers in each group.
    b = f[f.lens == "B_vs_peers"].copy()
    if not b.empty:
        b["group"] = b.apply(lambda x: (x.compared_with, x.metric,
                                        x.specialty, x.market if x.compared_with == "other platforms" else x.platform), axis=1)
        b = b.loc[b.r.abs().groupby(b.group).idxmax()].drop(columns="group")
    f = pd.concat([f[f.lens == "A_yoy_shift"], b])
    return _one_level_per_issue(f, ["lens", "compared_with", "metric", "direction"], "r", "base_volume")


def detect(as_of: str | dt.date, top_n: int = TOP_N, rule: str = FLAG_RULE,
           data_dir: Path = DATA_DIR) -> dict:
    """Flags for the review week ending on `as_of`, ready to hand to an agent as JSON.
    `data_dir` picks the world (dev by default; held-out worlds live in worlds/<name>/data)."""
    db = Database(as_of, data_dir)
    return build_flags(scores(db), consistency(db), db.as_of, top_n, rule)


def build_flags(s: pd.DataFrame, e: pd.DataFrame, as_of: dt.date,
                top_n: int = TOP_N, rule: str = FLAG_RULE) -> dict:
    """Apply the rule to precomputed scores (lets a calibration loop try many rules cheaply)."""
    week = review_week(as_of)
    flags = select_flags(s, rule)
    e = _one_level_per_issue(e, ["platform"], "max_share_missing", "scheduler_bookings")
    n_before_cap = len(flags) + len(e)

    out = []
    for _, x in e.iterrows():      # data problems first, whatever their size: they change how every other number reads
        out.append({"lens": "E_consistency", "market": x.market, "specialty": x.specialty, "platform": "all",
                    "metric": "bookings (app vs scheduler)", "days_with_gap": int(x.days_with_gap),
                    "gap_days": x.gap_days, "max_share_missing": round(float(x.max_share_missing), 3)})
    if not flags.empty:
        flags = flags.reindex(flags.bookings_impact.abs().sort_values(ascending=False).index).head(top_n)
    for _, x in flags.iterrows():
        out.append({"lens": x.lens, "market": x.market, "specialty": x.specialty, "platform": x.platform,
                    "metric": x.metric, "direction": x.direction,
                    "value": round(float(x.value), 4), "value_prev4": round(float(x.value_prev4), 4),
                    "compared_with": x.compared_with, "value_compared": round(float(x.value_compared), 4),
                    "residual_pct": round(100 * (math.exp(x.r) - 1), 1),   # beyond what was expected
                    "bookings_impact": round(float(x.bookings_impact)),
                    "z": round(float(x.z), 1), "base_volume": round(float(x.base_volume))})
    return {"review_week": f"{week} to {as_of}", "n_flags": len(out),
            "n_flags_before_cap": n_before_cap, "flags": out}
