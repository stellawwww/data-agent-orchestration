"""Story checks: does the generated data tell the story the answer key says it tells?

Each check looks only at the generated tables (what an agent would see), plus
evals/effect_sizes.csv for the true planted sizes. It prints PASS/FAIL per check and
saves docs/story_check.png: real flu vs synthetic urgent care demand per market,
with planted events marked.

Run after generate.py:  python data_gen/checks.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str) -> None:
    results.append((bool(ok), name, detail))


def load():
    f = pd.read_csv(DATA / "daily_funnel.csv", parse_dates=["date"])
    s = pd.read_csv(DATA / "daily_supply.csv", parse_dates=["date"])
    flu = pd.read_csv(DATA / "flu_activity.csv", parse_dates=["week_start"])
    mk = pd.read_csv(DATA / "markets.csv")
    return f, s, flu, mk


def between(df, a, b):
    return df[(df.date >= a) & (df.date <= b)]


def yoy(df, a, b, col="searches"):
    """Sum over [a, b] vs the same weekdays 364 days earlier."""
    lag = pd.Timedelta(days=364)
    now = between(df, pd.Timestamp(a), pd.Timestamp(b))[col].sum()
    then = between(df, pd.Timestamp(a) - lag, pd.Timestamp(b) - lag)[col].sum()
    return now / then


def main() -> None:
    f, s, flu, mk = load()

    # 1. funnel never grows as it narrows -----------------------------------
    bad = ((f.searches < f.searches_with_availability) | (f.searches_with_availability < f.profile_views)
           | (f.profile_views < f.bookings) | (f.bookings < 0)).sum()
    check(bad == 0, "Funnel is monotone", f"{bad} rows where a later step exceeds an earlier one")

    # 2. bookings never exceed slots ----------------------------------------
    over = (s.slots_booked > s.slots_offered).sum()
    check(over == 0, "Bookings within supply", f"{over} cells with slots_booked > slots_offered")

    # 3. two booking systems agree, except during the iOS tracking bug -------
    fb = f.groupby(["date", "market", "specialty"]).bookings.sum().reset_index()
    j = fb.merge(s, on=["date", "market", "specialty"])
    j["gap"] = j.slots_booked - j.bookings
    bug = j.date.between("2025-12-26", "2025-12-27")
    outside = (j.loc[~bug, "gap"] != 0).sum()
    ios = f[f.platform == "iOS"]
    ios_bug = between(ios, "2025-12-26", "2025-12-27").bookings.sum()
    ios_week = between(ios, "2025-12-19", "2025-12-20").bookings.sum()
    gap_share = j.loc[bug, "gap"].sum() / j.loc[bug, "slots_booked"].sum()
    check(outside == 0 and gap_share > 0.25, "Tracking bug only shows as a gap",
          f"mismatched cells outside bug days: {outside}; bug days: recorded iOS bookings {ios_bug:,} "
          f"vs {ios_week:,} a week earlier; {gap_share:.0%} of real bookings missing from funnel table")

    # 4. urgent care follows the real flu curve -----------------------------
    uc = f[f.specialty == "Urgent care"].copy()
    uc["week_start"] = uc.date - pd.to_timedelta((uc.date.dt.dayofweek + 1) % 7, unit="D")
    wk = uc.groupby(["market", "week_start"]).searches.sum().reset_index()
    wk = wk.merge(mk, on="market").merge(flu, left_on=["flu_region", "week_start"],
                                         right_on=["region", "week_start"])
    wk = wk[wk.groupby("market").week_start.transform("count") > 0]
    full_weeks = wk.groupby("market").week_start.transform(lambda x: x.between(x.min() + pd.Timedelta(days=7), x.max() - pd.Timedelta(days=7)))
    wk = wk[full_weeks]
    corrs = wk.groupby("market").apply(lambda g: np.corrcoef(np.log(g.searches), np.log(g.wili))[0, 1])
    check((corrs > 0.75).all(), "Urgent care tracks real flu",
          "corr(log weekly searches, log wILI): " + ", ".join(f"{m} {c:.2f}" for m, c in corrs.items()))

    # 5-6. flu seasons: compare peaks and timing, not fixed date windows ------
    uc_all = f[f.specialty == "Urgent care"].copy()
    uc_all["week"] = uc_all.date - pd.to_timedelta(uc_all.date.dt.dayofweek, unit="D")
    wk_uc = uc_all.groupby(["market", "week"]).searches.sum()

    def season(m):
        """Peak weekly searches / early-November level, per season; and how many weeks earlier."""
        g = wk_uc[m]
        def one(a, b, na, nb):
            s_ = g[(g.index >= a) & (g.index <= b)]
            base = g[(g.index >= na) & (g.index <= nb)].mean()
            return s_.max() / base, s_.idxmax()
        e0, w0 = one("2024-10-01", "2025-03-31", "2024-10-28", "2024-11-17")
        e1, w1 = one("2025-10-01", "2026-01-25", "2025-10-27", "2025-11-16")
        return e0, e1, (w0 + pd.Timedelta(days=364) - w1).days // 7, w0, w1

    lines, ok = [], True
    for m in ["NYC", "Atlanta"]:
        e0, e1, earlier, w0, w1 = season(m)
        ok &= e1 / e0 > 1.1 and earlier >= 3
        lines.append(f"{m} surge {e1:.2f}x vs {e0:.2f}x last season, peak {earlier} weeks earlier ({w1.date()})")
    check(ok, "NYC, Atlanta: bigger and earlier flu surge", "; ".join(lines))

    lines, ok = [], True
    for m in ["Houston", "Chicago"]:
        e0, e1, earlier, w0, w1 = season(m)
        ok &= e1 / e0 < 0.95 and earlier >= 3
        lines.append(f"{m} surge {e1:.2f}x vs {e0:.2f}x last season, peak {earlier} weeks earlier ({w1.date()})")
    hou = f[(f.market == "Houston") & (f.specialty == "Urgent care")]
    dec, late_jan = yoy(hou, "2025-12-14", "2025-12-27"), yoy(hou, "2026-01-18", "2026-02-01")
    ok &= dec > 1.1 and late_jan < 1.0
    lines.append(f"Houston YoY Dec 14-27 {dec:.2f}x, Jan 18-Feb 1 {late_jan:.2f}x")
    check(ok, "Houston, Chicago: earlier but milder", "; ".join(lines))

    # 7. Chicago dermatology supply loss -------------------------------------
    cd = s[(s.market == "Chicago") & (s.specialty == "Dermatology")]
    cf = f[(f.market == "Chicago") & (f.specialty == "Dermatology")]
    pre, post = ("2025-10-13", "2025-11-09"), ("2025-11-10", "2025-12-07")
    slots_r = between(cd, *post).slots_offered.sum() / between(cd, *pre).slots_offered.sum()
    srch_r = between(cf, *post).searches.sum() / between(cf, *pre).searches.sum()
    book_r = between(cf, *post).bookings.sum() / between(cf, *pre).bookings.sum()
    other = s[(s.market != "Chicago") & (s.specialty == "Dermatology")]
    other_r = between(other, *post).slots_offered.sum() / between(other, *pre).slots_offered.sum()
    check(slots_r < 0.75 and srch_r > 0.95, "Chicago derm: slots fall, searches don't",
          f"4 weeks after vs before Nov 10: slots {slots_r:.2f}x (other markets {other_r:.2f}x), "
          f"searches {srch_r:.2f}x, bookings {book_r:.2f}x")

    # 8. Web redesign: clicks up, CVR down, web only -------------------------
    def rates(d):
        return d.profile_views.sum() / d.searches_with_availability.sum(), d.bookings.sum() / d.profile_views.sum()
    pre, post = ("2025-09-16", "2025-10-13"), ("2025-10-14", "2025-11-10")
    out = {}
    for p in ["Web", "iOS"]:
        d = f[f.platform == p]
        (c0, v0), (c1, v1) = rates(between(d, *pre)), rates(between(d, *post))
        out[p] = (c1 / c0 - 1, v1 / v0 - 1)
    check(out["Web"][0] > 0.05 and out["Web"][1] < -0.03 and abs(out["iOS"][0]) < 0.02,
          "Web redesign shows in web funnel only",
          f"4 weeks after vs before Oct 14: Web click {out['Web'][0]:+.1%}, CVR {out['Web'][1]:+.1%}; "
          f"iOS click {out['iOS'][0]:+.1%}, CVR {out['iOS'][1]:+.1%}")

    # 9. NYC therapy campaign on top of January season -----------------------
    th = f[f.specialty == "Therapy"]
    nyc = yoy(th[th.market == "NYC"], "2026-01-12", "2026-02-01")
    rest = yoy(th[th.market != "NYC"], "2026-01-12", "2026-02-01")
    nyc_cvr = rates(between(th[th.market == "NYC"], "2026-01-12", "2026-02-01"))[1]
    nyc_cvr0 = rates(between(th[th.market == "NYC"], "2025-12-08", "2026-01-04"))[1]
    check(nyc / rest > 1.25, "NYC therapy stands out from other markets",
          f"therapy searches Jan 12-Feb 1 YoY: NYC {nyc:.2f}x vs other markets {rest:.2f}x; "
          f"NYC therapy CVR {nyc_cvr0:.3f} -> {nyc_cvr:.3f}")

    # 10. noise trap: big-looking drop that is inside normal variation -------
    nc = pd.read_csv(ROOT / "evals/noise_cases.csv")
    if len(nc):
        n = nc.iloc[0]
        check(n.wow <= -0.30 and abs(n.z) < 2, "Noise case looks alarming but isn't",
              f"{n.market} x {n.specialty} x {n.platform}, week of {n.week_start}: "
              f"{n.prev_week_bookings} -> {n.bookings} bookings ({n.wow:+.0%}); "
              f"normal weekly swing sd {n.wow_sd:.0%}; z = {n.z:.2f}")
    else:
        check(False, "Noise case looks alarming but isn't", "no candidate found")

    # ------------------------------------------------------------------ report
    width = max(len(n) for _, n, _ in results)
    for ok, name, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{sum(ok for ok, _, _ in results)}/{len(results)} checks passed")

    plot(f, flu, mk)


def plot(f, flu, mk) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = {"Web redesign": "2025-10-14", "Chicago derm supply loss": "2025-11-10",
              "iOS tracking bug": "2025-12-26", "NYC therapy campaign": "2026-01-12"}
    uc = f[f.specialty == "Urgent care"].copy()
    uc["week_start"] = uc.date - pd.to_timedelta((uc.date.dt.dayofweek + 1) % 7, unit="D")
    wk = uc.groupby(["market", "week_start"]).searches.sum().reset_index()
    markets = list(mk.market)
    fig, axes = plt.subplots(len(markets), 1, figsize=(11, 2.3 * len(markets)), sharex=True)
    for ax, m in zip(axes, markets):
        g = wk[(wk.market == m)].iloc[1:-1]
        region = mk.set_index("market").flu_region[m]
        fl = flu[(flu.region == region) & flu.week_start.between(g.week_start.min(), g.week_start.max())]
        ax.plot(g.week_start, g.searches, color="#2347C5", lw=1.6, label="Urgent care searches (synthetic)")
        ax.set_ylabel(m, rotation=0, ha="right", va="center", fontsize=10)
        ax.tick_params(labelsize=8)
        ax2 = ax.twinx()
        ax2.plot(fl.week_start, fl.wili * 100, color="#BF3A2B", lw=1.2, ls="--", label=f"CDC wILI % ({region}, real)")
        ax2.tick_params(labelsize=8, colors="#BF3A2B")
        for label, d in events.items():
            ax.axvline(pd.Timestamp(d), color="#8a8f98", lw=0.8, ls=":")
        ax.axvspan(pd.Timestamp("2025-10-06"), pd.Timestamp("2026-02-01"), color="#E7ECFB", zorder=0)
        if m == markets[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)
            for n_, (label, d) in enumerate(events.items(), start=1):
                ax.annotate(str(n_), xy=(pd.Timestamp(d), ax.get_ylim()[1]), xytext=(0, 2),
                            textcoords="offset points", fontsize=8, ha="center", va="bottom", color="#59606b")
    key = "   ".join(f"{n}  {label}" for n, label in enumerate(events, start=1))
    axes[-1].set_xlabel(f"Week (shaded = replay window)\nPlanted events:   {key}", fontsize=9)
    fig.suptitle("Story check: real flu activity vs synthetic urgent care demand", fontsize=12, y=0.995)
    fig.tight_layout()
    out = ROOT / "docs/story_check.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
