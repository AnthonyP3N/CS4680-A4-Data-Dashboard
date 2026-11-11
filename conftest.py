"""Makes the project root importable when tests are started with plain `pytest`.

Question limits are switched off for every test, since a test session asks far more
questions a minute than a person would; tests/test_security.py turns them on.
"""

import pytest

RATE_SETTINGS = ("ASKDATA_RATE_PER_MINUTE", "ASKDATA_RATE_PER_DAY", "ASKDATA_RATE_PER_CLIENT_MINUTE",
                 "ASKDATA_RATE_GLOBAL_PER_MINUTE")


@pytest.fixture(autouse=True)
def _no_question_limits(monkeypatch):
    for name in RATE_SETTINGS:
        monkeypatch.setenv(name, "0")
