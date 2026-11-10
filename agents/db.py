"""Read-only, time-bounded access to the marketplace tables.

Every agent reads data only through `Database`. Three layers keep the answer key and the
future out of reach:

1. Time cutoff: each table is loaded with only the rows an analyst could have seen
   on the morning after `as_of` (see `cutoff_filter`). Future rows never enter memory.
2. No file access: after loading, DuckDB's external access is switched off and the
   setting is locked, so queries can't read `evals/` or any other file.
3. SELECT only: each query must be exactly one SELECT statement (WITH ... SELECT is fine).

Two entry points:
  db.query(sql)    -> pandas DataFrame, for trusted code such as the SQL detector
  db.run_sql(sql)  -> text, for agents as a tool; never raises, errors come back as text
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# table name -> CSV file in data/
TABLES = {
    "daily_funnel": "daily_funnel.csv",  # date, market, specialty, platform, funnel counts
    "daily_supply": "daily_supply.csv",  # date, market, specialty, providers and slots
    "flu_activity": "flu_activity.csv",  # week_start (Sunday), epiweek, region, wili
    "calendar": "calendar.csv",  # date, holidays
    "change_log": "change_log.csv",  # event_date, event_type, scope, description
    "markets": "markets.csv",  # market -> flu_region (no dates)
}


class QueryRejected(Exception):
    """The query is not a single read-only SELECT."""


def cutoff_filter(table: str, as_of: dt.date) -> str | None:
    """SQL condition that keeps only the rows of `table` an analyst could see on `as_of`.

    `as_of` is the last day of data the run is allowed to see (for a weekly review,
    the last day of the week being reviewed). Return a condition for a WHERE clause,
    such as "date <= DATE '2025-12-28'", or None to load the whole table.
    Try conditions interactively in notebooks/01_cutoff_rules.ipynb first.
    """
    d = f"DATE '{as_of.isoformat()}'"  # e.g. DATE '2025-12-14'; use {d} inside the strings below

    conditions = {
        "daily_funnel": f"date <= {d}",
        "daily_supply": f"date <= {d}",
        "flu_activity": f"week_start + INTERVAL 12 DAY <= {d}",  # published the Friday after the week ends
        "calendar": None,
        "change_log": f"event_date <= {d}",
        "markets": None,
    }
    return conditions[table]


class Database:
    def __init__(
        self, as_of: str | dt.date, data_dir: Path = DATA_DIR, max_rows: int = 200
    ):
        self.as_of = pd.Timestamp(as_of).date()
        self.max_rows = max_rows
        self.con = duckdb.connect(":memory:")
        for name, filename in TABLES.items():
            path = (Path(data_dir) / filename).as_posix()
            sql = f"CREATE TABLE {name} AS SELECT * FROM read_csv('{path}')"
            where = cutoff_filter(name, self.as_of)
            if where:
                sql += f" WHERE {where}"
            self.con.execute(sql)
        # From here on, no query can touch the file system or undo this setting.
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    # -- safety
    def _check(self, sql: str) -> None:
        statements = self.con.extract_statements(sql)
        if len(statements) != 1:
            raise QueryRejected(
                f"Send exactly one SQL statement (got {len(statements)})."
            )
        if statements[0].type != duckdb.StatementType.SELECT:
            raise QueryRejected(
                "Only read-only SELECT queries are allowed (WITH ... SELECT is fine)."
            )

    # -- for trusted code
    def query(self, sql: str) -> pd.DataFrame:
        self._check(sql)
        return self.con.sql(sql).df()

    # -- for agents
    def run_sql(self, sql: str) -> str:
        """Run one SELECT and return the result as CSV text, capped at `max_rows` rows.

        Errors (rejected statements, syntax errors, unknown columns) are returned as
        text starting with "ERROR:" so the agent can read them and fix its query.
        """
        try:
            self._check(sql)
            df = self.con.sql(sql).df()
        except QueryRejected as e:
            return f"ERROR: {e}"
        except duckdb.Error as e:
            return f"ERROR: {type(e).__name__}: {e}"
        return format_result(df, self.max_rows)

    def describe(self) -> str:
        """Tables, columns, row counts and date ranges: what an agent sees at the start of a run."""
        lines = [f"Data available through {self.as_of.isoformat()}."]
        for name in TABLES:
            cols = self.con.sql(f"DESCRIBE {name}").df()
            n = self.con.sql(f"SELECT count(*) FROM {name}").fetchall()[0][0]
            date_col = next(
                (
                    c
                    for c in ("date", "week_start", "event_date")
                    if c in set(cols.column_name)
                ),
                None,
            )
            span = ""
            if date_col and n:
                lo, hi = self.con.sql(
                    f"SELECT min({date_col}), max({date_col}) FROM {name}"
                ).fetchall()[0]
                span = f", {date_col} {lo} to {hi}"
            col_list = ", ".join(
                f"{c} {t}" for c, t in zip(cols.column_name, cols.column_type)
            )
            lines.append(f"- {name} ({n:,} rows{span}): {col_list}")
        return "\n".join(lines)


def format_result(df: pd.DataFrame, max_rows: int) -> str:
    if df.empty:
        return "0 rows."
    shown = df.head(max_rows).copy()
    for c in shown.select_dtypes("float").columns:
        shown[c] = shown[c].round(4)
    header = f"{len(df):,} rows."
    if len(df) > max_rows:
        header = (
            f"{len(df):,} rows; showing the first {max_rows}. "
            "Aggregate further or add ORDER BY ... LIMIT to see what matters."
        )
    return header + "\n" + shown.to_csv(index=False)
