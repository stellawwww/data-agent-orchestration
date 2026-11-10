"""Generate the synthetic marketplace tables.

Inputs
  config/world.yaml        assumptions about patients and providers
  evals/scenarios.yaml     planted events (the `effect` of each scenario)
  data/flu_activity.csv    real CDC flu data (run fetch_flu.py first)

Outputs (visible to agents)
  data/daily_funnel.csv    date x market x specialty x platform
  data/daily_supply.csv    date x market x specialty
  data/calendar.csv        one row per date
  data/change_log.csv      product launches, releases, campaigns
  data/markets.csv         market -> flu region

Output (answer key, hidden from agents)
  evals/effect_sizes.csv   true size of each planted effect, measured against a
                           counterfactual world with the same noise but no effect

How it works: every quantity is first computed as an expected value, with the same
random shocks in both the actual and the counterfactual world. Counts (searches,
views, bookings) are then sampled once, from the actual world only.

Run:  python data_gen/generate.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# ---------------------------------------------------------------- helpers

def md(s: str) -> tuple[int, int]:
    m, d = s.split("-")
    return int(m), int(d)


def holidays_for_year(y: int) -> dict[dt.date, str]:
    nov1 = dt.date(y, 11, 1)
    thanksgiving = nov1 + dt.timedelta(days=(3 - nov1.weekday()) % 7 + 21)
    may31 = dt.date(y, 5, 31)
    memorial = may31 - dt.timedelta(days=may31.weekday())
    sep1 = dt.date(y, 9, 1)
    labor = sep1 + dt.timedelta(days=(0 - sep1.weekday()) % 7)
    return {
        dt.date(y, 1, 1): "New Year's Day",
        dt.date(y, 1, 2): "Day after New Year's",
        memorial: "Memorial Day",
        dt.date(y, 7, 4): "Independence Day",
        labor: "Labor Day",
        thanksgiving: "Thanksgiving",
        thanksgiving + dt.timedelta(days=1): "Day after Thanksgiving",
        dt.date(y, 12, 24): "Christmas Eve",
        dt.date(y, 12, 25): "Christmas Day",
        dt.date(y, 12, 26): "Day after Christmas",
        dt.date(y, 12, 31): "New Year's Eve",
    }


def in_md_window(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """True for dates inside a month-day window that may wrap the new year."""
    key = dates.month * 100 + dates.day
    s, e = (lambda a: a[0] * 100 + a[1])(md(start)), (lambda a: a[0] * 100 + a[1])(md(end))
    return (key >= s) | (key <= e) if s > e else (key >= s) & (key <= e)


def seasonal_bump(dates: pd.DatetimeIndex, center: str, width: float) -> np.ndarray:
    """Gaussian bump around a month-day, using the nearest occurrence in any year."""
    m, d = md(center)
    dist = np.full(len(dates), np.inf)
    for y in range(dates.year.min() - 1, dates.year.max() + 2):
        c = pd.Timestamp(year=y, month=m, day=d)
        dist = np.minimum(dist, np.abs((dates - c).days.values))
    return np.exp(-0.5 * (dist / width) ** 2)


def scope_mask(scope_val, names: list[str]) -> np.ndarray:
    if scope_val == "all":
        return np.ones(len(names), bool)
    return np.isin(names, scope_val)


# ---------------------------------------------------------------- calendar & flu

def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    hol = {}
    for y in range(dates.year.min(), dates.year.max() + 1):
        hol.update(holidays_for_year(y))
    names = [hol.get(d.date(), "") for d in dates]
    closures = {"New Year's Day", "Memorial Day", "Independence Day", "Labor Day",
                "Thanksgiving", "Christmas Day"}
    thanks = {d for d, n in hol.items() if n == "Thanksgiving"}
    thanks_week = {t + dt.timedelta(days=k) for t in thanks for k in range(-1, 4)}
    year_end = in_md_window(dates, "12-22", "01-02")
    return pd.DataFrame({
        "date": dates.date,
        "day_of_week": dates.day_name(),
        "is_holiday": [n in closures for n in names],
        "holiday_name": names,
        "is_holiday_week": year_end | np.array([d.date() in thanks_week for d in dates]),
    })


def daily_flu(dates: pd.DatetimeIndex, regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily wILI (decimal) per region, interpolated from mid-week, and flu_index vs summer baseline."""
    flu = pd.read_csv(DATA / "flu_activity.csv", parse_dates=["week_start"])
    wili, index = {}, {}
    for r in regions:
        s = flu[flu.region == r].set_index("week_start")["wili"]
        s.index = s.index + pd.Timedelta(days=3)                    # Wednesday = mid-week
        full = s.reindex(s.index.union(dates)).interpolate("time").ffill().bfill().reindex(dates)
        baseline = s[s.index.month.isin([6, 7, 8])].median()          # summer = off-season
        wili[r] = full.values
        index[r] = full.values / baseline
    return pd.DataFrame(wili, index=dates), pd.DataFrame(index, index=dates)


# ---------------------------------------------------------------- the world

class World:
    def __init__(self, world: dict, scenarios: dict):
        self.w = world
        self.sc = scenarios["scenarios"]
        self.dates = pd.date_range(world["start_date"], world["end_date"], freq="D")
        self.markets = list(world["markets"])
        self.specs = list(world["specialties"])
        self.plats = list(world["platforms"])
        self.D, self.M, self.S, self.P = len(self.dates), len(self.markets), len(self.specs), len(self.plats)
        self.cal = build_calendar(self.dates)
        regions = [world["markets"][m]["flu_region"] for m in self.markets]
        self.wili, self.flu_index = daily_flu(self.dates, sorted(set(regions)))
        self.fi = self.flu_index[regions].values                       # (D, M)

        rng = np.random.default_rng(world["seed"])
        dn = world["demand"]
        self.shock_market = rng.lognormal(-dn["market_day_noise_sd"] ** 2 / 2, dn["market_day_noise_sd"], (self.D, self.M))
        self.shock_cell = rng.lognormal(-dn["day_noise_sd"] ** 2 / 2, dn["day_noise_sd"], (self.D, self.M, self.S))
        sn = world["supply"]["slot_noise_sd"]
        self.shock_slots = rng.lognormal(-sn ** 2 / 2, sn, (self.D, self.M, self.S))
        self.sample_rng = np.random.default_rng(world["seed"] + 1)

    # -- scenario plumbing
    def window_mask(self, sc) -> np.ndarray:
        start = pd.Timestamp(sc["window"]["start"])
        end = pd.Timestamp(sc["window"]["end"]) if sc["window"]["end"] else self.dates[-1]
        return (self.dates >= start) & (self.dates <= end)

    def mask(self, sc, with_platform: bool) -> np.ndarray:
        sp = sc["scope"]
        m = (self.window_mask(sc)[:, None, None]
             & scope_mask(sp["market"], self.markets)[None, :, None]
             & scope_mask(sp["specialty"], self.specs)[None, None, :])
        if with_platform:
            m = m[..., None] & scope_mask(sp["platform"], self.plats)[None, None, None, :]
        return m

    def effects(self, kind: str, off: frozenset):
        return [s for s in self.sc if s["effect"]["type"] == kind and s["id"] not in off]

    # -- expected-value pipeline
    def expected(self, off: frozenset = frozenset()) -> dict:
        """Expected values with every planted effect applied, except the scenario ids in `off`."""
        w, dates = self.w, self.dates
        t = (dates - dates[0]).days.values / 365.0
        wd = dates.dayofweek.values
        names = self.cal["holiday_name"].values

        # demand ---------------------------------------------------------
        lam = np.zeros((self.D, self.M, self.S))
        season_all = np.zeros((self.D, self.S))
        growth = (1 + w["demand"]["annual_growth"]) ** t
        for j, s in enumerate(self.specs):
            cfg = w["specialties"][s]
            season = np.ones(self.D)
            for b in cfg["season"]:
                season += b["amp"] * seasonal_bump(dates, b["center"], b["width"])
            season_all[:, j] = season
            dow = np.array(cfg["dow"])[wd]
            hol = np.array([w["holidays"].get(n, 1.0) for n in names])
            hol = 1 - (1 - hol) * cfg["holiday_damping"]
            flu = 1 + cfg["flu_beta"] * np.clip(self.fi - 1, -0.5, None)          # (D, M)
            if cfg.get("flu_months"):
                flu = np.where(np.isin(dates.month, cfg["flu_months"])[:, None], flu, 1.0)
            for i, m in enumerate(self.markets):
                lam[:, i, j] = (cfg["base_searches"] * w["markets"][m]["size"] * growth
                                * dow * season * hol * flu[:, i])
        lam *= self.shock_market[:, :, None] * self.shock_cell
        for sc in self.effects("demand", off):
            lam = np.where(self.mask(sc, False), lam * sc["effect"].get("searches", 1.0), lam)

        ps = w["platform_share"]
        ios = ps["iOS_start"] + (ps["iOS_end"] - ps["iOS_start"]) * t / t[-1]
        share = np.stack([ios, np.full(self.D, ps["Android"]), 1 - ios - ps["Android"]], axis=1)  # (D, P)
        lam_p = lam[..., None] * share[:, None, None, :]

        # supply ---------------------------------------------------------
        sup = w["supply"]
        g_sup = (1 + sup["annual_growth"]) ** t
        year_end = in_md_window(dates, sup["year_end_vacation"]["start"], sup["year_end_vacation"]["end"])
        roster = np.zeros((self.D, self.M, self.S))
        open_share = np.zeros((self.D, self.M, self.S))
        spp = np.zeros(self.S)
        for j, s in enumerate(self.specs):
            cfg = w["specialties"][s]
            sc_ = cfg["supply"]
            spp[j] = sc_["slots_per_provider"]
            follow = 1 + sc_.get("follows_season", 0) * (season_all[:, j] - 1)
            weekend = np.where(wd == 5, sc_["weekend"][0], np.where(wd == 6, sc_["weekend"][1], 1.0))
            hol_open = np.array([sup["holiday_open_share"].get(n, np.nan) for n in names])
            hol_open = np.where(np.isnan(hol_open),
                                np.where(year_end, sup["year_end_vacation"]["open_share"], 1.0), hol_open)
            if s == "Urgent care":
                hol_open = 1 - (1 - hol_open) * sup["urgent_care_holiday_damping"]
            for i, m in enumerate(self.markets):
                base = cfg["base_searches"] * w["markets"][m]["size"] * sc_["supply_ratio"] / sc_["slots_per_provider"]
                roster[:, i, j] = base * g_sup * follow
                open_share[:, i, j] = weekend * hol_open
        open_share *= (1 - sup["flu_sickness"] * np.clip(self.fi - 1, 0, None))[:, :, None]
        for sc in self.effects("supply", off):
            roster = np.where(self.mask(sc, False), roster * sc["effect"]["providers"], roster)
        roster = np.maximum(np.round(roster), 1)
        open_share = np.clip(open_share, 0, 1)
        active = roster * open_share
        slots = active * spp[None, None, :] * self.shock_slots

        # funnel ---------------------------------------------------------
        fn = w["funnel"]
        k = w["availability_k"]
        lam_tot = lam_p.sum(-1)
        avail = 1 - np.exp(-k * slots / np.maximum(lam_tot, 1e-9))          # (D, M, S)
        click = np.ones((self.D, self.M, self.S, self.P))
        cvr = np.ones((self.D, self.M, self.S, self.P))
        for j, s in enumerate(self.specs):
            for q, p in enumerate(self.plats):
                click[:, :, j, q] = fn["click_rate"][p] * w["specialties"][s]["click_mult"]
                cvr[:, :, j, q] = fn["booking_cvr"][p] * w["specialties"][s]["cvr_mult"]
        for sc in self.effects("funnel", off):
            msk = self.mask(sc, True)
            click = np.where(msk, click * sc["effect"].get("click_rate", 1.0), click)
            cvr = np.where(msk, cvr * sc["effect"].get("booking_cvr", 1.0), cvr)
        for sc in self.effects("demand", off):
            cvr = np.where(self.mask(sc, True), cvr * sc["effect"].get("booking_cvr", 1.0), cvr)
        click, cvr = np.clip(click, 0, 1), np.clip(cvr, 0, 1)

        swa = lam_p * avail[..., None]
        views = swa * click
        book = views * cvr
        cap = fn["max_fill"] * slots
        scale = np.minimum(1, cap / np.maximum(book.sum(-1), 1e-9))
        book = book * scale[..., None]
        thin = np.ones_like(book)
        for sc in self.effects("thin_bookings", off):
            thin = np.where(self.mask(sc, True), sc["effect"]["factor"], thin)
        book = book * thin
        recorded = np.ones_like(book)
        for sc in self.effects("tracking", off):
            recorded = np.where(self.mask(sc, True), sc["effect"]["recorded_bookings"], recorded)

        return dict(lam_p=lam_p, roster=roster, open_share=open_share, active=active, slots=slots,
                    avail=avail, click=click, cvr=cvr, scale=scale, thin=thin, recorded=recorded,
                    swa=swa, views=views, book=book, rec_book=book * recorded)

    # -- sampling
    def sample(self, e: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = self.sample_rng
        active = rng.binomial(e["roster"].astype(int), e["open_share"])
        slots = np.round(active * (e["slots"] / np.maximum(e["active"], 1e-9))).astype(int)
        searches = rng.poisson(e["lam_p"])
        tot = searches.sum(-1)
        k = self.w["availability_k"]
        avail = 1 - np.exp(-k * slots / np.maximum(tot, 1))
        swa = rng.binomial(searches, np.broadcast_to(avail[..., None], searches.shape))
        views = rng.binomial(swa, e["click"])
        book = rng.binomial(views, e["cvr"])
        cap = np.floor(self.w["funnel"]["max_fill"] * slots)
        over = book.sum(-1) > cap
        if over.any():
            ratio = np.where(over, cap / np.maximum(book.sum(-1), 1), 1.0)
            book = np.floor(book * ratio[..., None]).astype(int)
        book = rng.binomial(book, e["thin"])
        slots_booked = book.sum(-1)
        recorded = rng.binomial(book, e["recorded"])

        idx = pd.MultiIndex.from_product([self.dates.date, self.markets, self.specs, self.plats],
                                         names=["date", "market", "specialty", "platform"])
        funnel = pd.DataFrame({
            "searches": searches.ravel(),
            "searches_with_availability": swa.ravel(),
            "profile_views": views.ravel(),
            "bookings": recorded.ravel(),
        }, index=idx).reset_index()
        idx3 = pd.MultiIndex.from_product([self.dates.date, self.markets, self.specs],
                                          names=["date", "market", "specialty"])
        supply = pd.DataFrame({
            "active_providers": active.ravel(),
            "slots_offered": slots.ravel(),
            "slots_booked": slots_booked.ravel(),
        }, index=idx3).reset_index()
        return funnel, supply

    # -- answer key: true effect sizes
    def effect_sizes(self, act: dict) -> pd.DataFrame:
        """Each scenario vs a counterfactual where only that scenario's effect is removed."""
        rows = []
        for sc in self.sc:
            if sc["effect"]["type"] == "none":
                continue
            cf = self.expected(off=frozenset({sc["id"]}))
            m3, m4 = self.mask(sc, False), self.mask(sc, True)
            # platform-scoped effects: measure on the scoped platforms only
            def tot(e, key, four=True):
                return (e[key] * m4).sum() if four else (e[key] * m3).sum()
            r = {"id": sc["id"]}
            for label, a, c in [
                ("searches", tot(act, "lam_p"), tot(cf, "lam_p")),
                ("searches_with_availability", tot(act, "swa"), tot(cf, "swa")),
                ("profile_views", tot(act, "views"), tot(cf, "views")),
                ("bookings_true", tot(act, "book"), tot(cf, "book")),
                ("bookings_recorded", tot(act, "rec_book"), tot(cf, "rec_book")),
                ("slots_offered", tot(act, "slots", False), tot(cf, "slots", False)),
            ]:
                r[f"{label}_change"] = round(a / c - 1, 4) if c else np.nan
            s_a, s_c = tot(act, "lam_p"), tot(cf, "lam_p")
            r["availability_rate_change"] = round((tot(act, "swa") / s_a) / (tot(cf, "swa") / s_c) - 1, 4)
            r["click_rate_change"] = round((tot(act, "views") / tot(act, "swa")) / (tot(cf, "views") / tot(cf, "swa")) - 1, 4)
            r["booking_cvr_change"] = round((tot(act, "book") / tot(act, "views")) / (tot(cf, "book") / tot(cf, "views")) - 1, 4)
            rows.append(r)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- noise cases

def find_noise_cases(funnel: pd.DataFrame, scen: dict, top: int = 3) -> pd.DataFrame:
    """Small cell-weeks in the replay window with big week-over-week drops that are still noise.

    A drop counts as noise when |z| < 2, where z compares it to the cell's own history of
    week-over-week changes. Holiday weeks and cells touched by planted events are skipped.
    """
    f = funnel.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["week"] = f.date - pd.to_timedelta(f.date.dt.dayofweek, unit="D")
    wk = f.groupby(["market", "specialty", "platform", "week"]).bookings.sum().reset_index()
    wk = wk[wk.week > wk.week.min()]                       # drop the partial first week
    wk = wk.sort_values("week")
    wk["prev"] = wk.groupby(["market", "specialty", "platform"]).bookings.shift(1)
    wk["wow"] = wk.bookings / wk.prev - 1

    rw = scen["replay_window"]
    in_window = wk.week.between(pd.Timestamp(rw["start"]) + pd.Timedelta(days=7), pd.Timestamp(rw["end"]) - pd.Timedelta(days=6))
    holiday = wk.week.between("2025-11-24", "2025-12-07") | wk.week.between("2025-12-22", "2026-01-11")
    touched = np.zeros(len(wk), bool)
    for sc in scen["scenarios"]:
        if sc["kind"] != "planted":
            continue
        sp = sc["scope"]
        hit = np.ones(len(wk), bool)
        for col in ["market", "specialty", "platform"]:
            if sp[col] != "all":
                hit &= wk[col].isin(sp[col]).values
        touched |= hit
    stats = (wk[~in_window].groupby(["market", "specialty", "platform"]).wow
             .agg(wow_mean="mean", wow_sd="std").reset_index())
    small = (wk[in_window].groupby(["market", "specialty", "platform"]).bookings.median()
             .rename("median_weekly_bookings").reset_index())
    c = wk[in_window & ~holiday & ~touched].merge(stats, on=["market", "specialty", "platform"]) \
        .merge(small, on=["market", "specialty", "platform"])
    c = c[c.median_weekly_bookings < 60]
    c["z"] = (c.wow - c.wow_mean) / c.wow_sd
    c = c[(c.z.abs() < 2) & (c.wow < 0)].sort_values("wow").head(top)
    c = c.rename(columns={"week": "week_start", "prev": "prev_week_bookings"})
    c["prev_week_bookings"] = c.prev_week_bookings.astype(int)
    return c[["market", "specialty", "platform", "week_start", "bookings", "prev_week_bookings",
              "wow", "wow_sd", "z", "median_weekly_bookings"]].round(3)


# ---------------------------------------------------------------- main

def main() -> None:
    world = yaml.safe_load((ROOT / "config/world.yaml").read_text())
    scen = yaml.safe_load((ROOT / "evals/scenarios.yaml").read_text())
    W = World(world, scen)

    actual = W.expected()
    funnel, supply = W.sample(actual)

    funnel.to_csv(DATA / "daily_funnel.csv", index=False)
    supply.to_csv(DATA / "daily_supply.csv", index=False)
    W.cal.to_csv(DATA / "calendar.csv", index=False)
    pd.DataFrame([{"market": m, "flu_region": v["flu_region"]} for m, v in world["markets"].items()]) \
        .to_csv(DATA / "markets.csv", index=False)

    log = [dict(e) for e in scen.get("extra_change_log", [])]
    for sc in scen["scenarios"]:
        if "change_log" in sc:
            sp = sc["scope"]
            parts = [", ".join(v) for k, v in sp.items() if v != "all"]
            log.append({"event_date": sc["window"]["start"], "event_type": sc["change_log"]["event_type"],
                        "scope": " / ".join(parts) if parts else "all",
                        "description": sc["change_log"]["description"]})
    pd.DataFrame(log).sort_values("event_date").to_csv(DATA / "change_log.csv", index=False)

    sizes = W.effect_sizes(actual)
    sizes.to_csv(ROOT / "evals/effect_sizes.csv", index=False)

    noise = find_noise_cases(funnel, scen)
    noise.to_csv(ROOT / "evals/noise_cases.csv", index=False)
    print("noise case:", noise.iloc[0].to_dict())

    print(f"daily_funnel: {len(funnel):,} rows   daily_supply: {len(supply):,} rows")
    print(f"dates {W.dates[0].date()} -> {W.dates[-1].date()}, change_log rows: {len(log)}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sizes.set_index("id").T)


if __name__ == "__main__":
    main()
