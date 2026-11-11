"""Tests for the three worlds: dev is unchanged, held-out worlds are well-formed, test stays sealed."""
import itertools

import pandas as pd
import pytest
import yaml

from data_gen.generate import ROOT, build_world, world_dirs
from data_gen.inject import _overlaps, random_scenarios
from evals.alert_budget import flags_per_quiet_week
from evals.detector_check import answers_dir

TABLES = ["daily_funnel", "daily_supply", "calendar", "change_log", "markets", "flu_activity"]


def test_dev_world_regenerates_byte_identical(tmp_path):
    world = yaml.safe_load((ROOT / "config/world.yaml").read_text())
    scen = yaml.safe_load((ROOT / "evals/scenarios.yaml").read_text())
    build_world(world, scen, tmp_path / "data", tmp_path / "answers", verbose=False)
    for t in ["daily_funnel", "daily_supply", "change_log"]:
        assert (tmp_path / "data" / f"{t}.csv").read_bytes() == (ROOT / "data" / f"{t}.csv").read_bytes()


def test_injected_events_are_reproducible_and_apart():
    world = yaml.safe_load((ROOT / "config/world.yaml").read_text())
    window = {"start": "2025-10-06", "end": "2026-02-01"}
    a, b = random_scenarios(123, world, window), random_scenarios(123, world, window)
    assert a == b
    planted = [e for e in a if e["kind"] == "planted"]
    assert len(planted) == 30
    assert not any(_overlaps(x, y) for x, y in itertools.combinations(planted, 2))


@pytest.mark.parametrize("name", ["validation", "test"])
def test_heldout_worlds_have_the_dev_schema(name):
    data, _ = world_dirs(name)
    for t in TABLES:
        dev_cols = pd.read_csv(ROOT / "data" / f"{t}.csv", nrows=1).columns.tolist()
        assert pd.read_csv(data / f"{t}.csv", nrows=1).columns.tolist() == dev_cols


def test_test_world_is_sealed(monkeypatch):
    monkeypatch.delenv("FINAL_SCORING", raising=False)
    with pytest.raises(PermissionError):
        answers_dir("test")
    with pytest.raises(PermissionError):
        flags_per_quiet_week(world="test")
