"""Tests for the data layer: no future data, no file access, no writes."""
import datetime as dt

import pandas as pd
import pytest

from agents.db import Database

AS_OF = dt.date(2025, 12, 14)          # a Sunday in the replay window, mid flu surge


@pytest.fixture(scope="module")
def db():
    return Database(AS_OF)


def max_date(db, table, col):
    return pd.Timestamp(db.query(f"SELECT max({col}) AS m FROM {table}").m[0]).date()


# ---------------------------------------------------------------- no future data

@pytest.mark.parametrize("table,col", [
    ("daily_funnel", "date"),
    ("daily_supply", "date"),
    ("change_log", "event_date"),
])
def test_no_rows_after_as_of(db, table, col):
    assert max_date(db, table, col) <= AS_OF


def test_flu_weeks_are_complete(db):
    # A CDC week (Sunday to Saturday) has no number until it is over.
    assert max_date(db, "flu_activity", "week_start") + dt.timedelta(days=6) <= AS_OF


def test_future_change_log_hidden():
    # The NYC therapy campaign starts 2026-01-12; a run in mid-December must not see it.
    early = Database(dt.date(2025, 12, 14))
    assert early.query("SELECT count(*) AS n FROM change_log WHERE event_date > DATE '2025-12-14'").n[0] == 0


def test_history_is_kept(db):
    # The cutoff must not throw away the past: year-over-year comparisons need it.
    first = pd.Timestamp(db.query("SELECT min(date) AS m FROM daily_funnel").m[0]).date()
    assert first == dt.date(2024, 2, 1)
    assert db.query("SELECT count(*) AS n FROM markets").n[0] == 5


# ---------------------------------------------------------------- no file access

@pytest.mark.parametrize("sql", [
    "SELECT * FROM 'evals/effect_sizes.csv'",
    "SELECT * FROM read_csv('evals/noise_cases.csv')",
    "SELECT * FROM read_text('evals/scenarios.yaml')",
    "SELECT * FROM read_csv('data/daily_funnel.csv') WHERE date > DATE '2025-12-14'",
])
def test_cannot_read_files(db, sql):
    assert db.run_sql(sql).startswith("ERROR")


# ---------------------------------------------------------------- read-only

@pytest.mark.parametrize("sql", [
    "DROP TABLE daily_funnel",
    "DELETE FROM daily_funnel",
    "INSERT INTO markets VALUES ('X', 'y')",
    "SELECT 1; DROP TABLE daily_funnel",
    "COPY daily_funnel TO 'leak.csv'",
    "SET enable_external_access = true",
    "ATTACH 'other.db'",
])
def test_only_select_allowed(db, sql):
    assert db.run_sql(sql).startswith("ERROR")
    assert db.query("SELECT count(*) AS n FROM markets").n[0] == 5


def test_with_select_allowed(db):
    out = db.run_sql("WITH m AS (SELECT market FROM markets) SELECT count(*) AS n FROM m")
    assert out.startswith("1 rows.") and "5" in out


# ---------------------------------------------------------------- agent-facing output

def test_errors_come_back_as_text(db):
    out = db.run_sql("SELECT no_such_column FROM daily_funnel")
    assert out.startswith("ERROR") and "no_such_column" in out


def test_large_results_are_capped(db):
    out = db.run_sql("SELECT * FROM daily_funnel")
    assert "showing the first 200" in out
    assert len(out.splitlines()) == 1 + 1 + 200        # note line, CSV header, 200 rows


def test_describe_lists_every_table(db):
    text = db.describe()
    for t in ["daily_funnel", "daily_supply", "flu_activity", "calendar", "change_log", "markets"]:
        assert t in text
    assert "2025-12-14" in text
