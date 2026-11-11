"""Fixtures shared by the test files."""

from __future__ import annotations

import pytest

from data.seed import build_database


@pytest.fixture
def small_sample(tmp_path_factory, monkeypatch):
    """Every workspace's sample store is copied from one small database built for this test."""
    master = build_database(tmp_path_factory.mktemp("master") / "sample.db", n_orders=250)
    monkeypatch.setattr("data.registry.master_sample_db", lambda directory, builder=None: master)
    return master
