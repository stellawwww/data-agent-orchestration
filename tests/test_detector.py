"""Tests for the SQL detector: it looks only backwards and catches the obvious cases."""
import datetime as dt

import pandas as pd
import pytest

from agents.db import Database
from agents.detector import detect, review_week, scores

KEYS = ["market", "specialty", "platform", "metric", "lens", "compared_with"]


def test_as_of_must_be_a_sunday():
    with pytest.raises(ValueError):
        review_week(dt.date(2025, 11, 14))
    assert review_week(dt.date(2025, 11, 16)) == dt.date(2025, 11, 10)


def test_scores_do_not_look_ahead():
    # Scores for the week of Nov 10 must not change when a later week's data is loaded.
    # Run against a database cut a week later, filtered back to the same review week.
    now = scores(Database("2025-11-16"))
    later_db = Database("2025-11-23")
    later_db.as_of = dt.date(2025, 11, 16)         # same review week, but next week's rows are loaded
    later = scores(later_db)
    merged = now.merge(later, on=KEYS, suffixes=("", "_later"))
    assert len(merged) == len(now) > 0
    pd.testing.assert_series_equal(merged.z, merged.z_later, check_names=False)


def test_chicago_dermatology_supply_loss_is_flagged():
    # Ranked by bookings impact, so it need not be #1 (iOS searches +1.4% is about the same size).
    flags = detect("2025-11-16")["flags"]
    chi = [f for f in flags if f["market"] == "Chicago" and f["specialty"] == "Dermatology"]
    assert any(f["metric"] == "slots_offered" and f["direction"] == "down" for f in chi)
    assert flags.index(chi[0]) < 3


def test_flags_are_ranked_by_bookings_impact():
    flags = [f for f in detect("2025-11-16")["flags"] if f["lens"] != "E_consistency"]
    impacts = [abs(f["bookings_impact"]) for f in flags]
    assert impacts == sorted(impacts, reverse=True)


def test_tracking_bug_shows_as_a_gap_between_booking_systems():
    flags = detect("2025-12-28")["flags"]
    e = [f for f in flags if f["lens"] == "E_consistency"]
    assert e and "12-26" in e[0]["gap_days"] and "12-27" in e[0]["gap_days"]


def test_flags_are_capped_and_json_ready():
    out = detect("2025-12-28", top_n=5)
    assert out["n_flags"] <= 5 + 1                  # E flags are always included, on top of the cap
    for f in out["flags"]:
        assert all(isinstance(v, (str, int, float)) for v in f.values())
