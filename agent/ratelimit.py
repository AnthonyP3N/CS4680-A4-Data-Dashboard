"""Limits on how many questions can be asked, shared by every session of the app.

Each question costs a model call on someone's API key, so a public deployment
needs limits that a visitor can't get around by opening a new tab or a new
private link. Three kinds of limit apply together, and a question is refused if
any of them is reached:

  per workspace  questions a minute and a day for one workspace
  per client     questions a minute from one network address (when the app can
                 see it), so new private links don't bring new allowances
  app-wide       questions a minute answered with the app's own key, across all
                 visitors; protects the owner's quota and bill

Questions answered with a visitor's own key count toward the first two only.
A refused question is not counted. One limiter lives for the whole process
(`shared_limiter`), so the limits hold across Streamlit sessions. The counts are
kept in memory: they start again when the app restarts.

Settings (secrets or environment), 0 switches a limit off:
  ASKDATA_RATE_PER_MINUTE         per workspace, default 10
  ASKDATA_RATE_PER_DAY            per workspace, default 300
  ASKDATA_RATE_PER_CLIENT_MINUTE  per network address, default 20
  ASKDATA_RATE_GLOBAL_PER_MINUTE  app-wide with the app's key, default 60
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

GetSecret = Callable[[str, Any], Any]

DEFAULTS = {
    "ASKDATA_RATE_PER_MINUTE": 10,
    "ASKDATA_RATE_PER_DAY": 300,
    "ASKDATA_RATE_PER_CLIENT_MINUTE": 20,
    "ASKDATA_RATE_GLOBAL_PER_MINUTE": 60,
}
MINUTE, DAY = 60.0, 86_400.0
MAX_KEYS = 10_000               # windows kept per limit; the least recently used are dropped beyond this


@dataclass(frozen=True)
class RateLimits:
    per_minute: int = DEFAULTS["ASKDATA_RATE_PER_MINUTE"]
    per_day: int = DEFAULTS["ASKDATA_RATE_PER_DAY"]
    per_client_minute: int = DEFAULTS["ASKDATA_RATE_PER_CLIENT_MINUTE"]
    global_per_minute: int = DEFAULTS["ASKDATA_RATE_GLOBAL_PER_MINUTE"]

    @classmethod
    def from_settings(cls, get_secret: GetSecret) -> "RateLimits":
        def number(name: str) -> int:
            try:
                return max(0, int(get_secret(name, DEFAULTS[name])))
            except (TypeError, ValueError):
                return DEFAULTS[name]
        return cls(number("ASKDATA_RATE_PER_MINUTE"), number("ASKDATA_RATE_PER_DAY"),
                   number("ASKDATA_RATE_PER_CLIENT_MINUTE"), number("ASKDATA_RATE_GLOBAL_PER_MINUTE"))


@dataclass
class Decision:
    allowed: bool
    message: str = ""
    retry_after_s: float = 0.0
    limit: str = ""              # which limit refused it: "workspace_minute", "workspace_day", "client", "global"


class _Window:
    """Timestamps of recent events per key, for one limit."""

    def __init__(self, limit: int, period: float):
        self.limit, self.period = limit, period
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def wait(self, key: str, now: float) -> float:
        """0 if one more event fits for `key` now; otherwise seconds until it would."""
        if self.limit <= 0 or not key:
            return 0.0
        events = self.events.get(key)
        if events is None:
            return 0.0
        while events and now - events[0] >= self.period:
            events.popleft()
        if len(events) < self.limit:
            return 0.0
        return self.period - (now - events[0])

    def record(self, key: str, now: float) -> None:
        if self.limit <= 0 or not key:
            return
        events = self.events.pop(key, None) or deque()
        events.append(now)
        self.events[key] = events             # re-inserted: dicts keep order, so the oldest keys come first
        while len(self.events) > MAX_KEYS:
            self.events.pop(next(iter(self.events)))


class RequestLimiter:
    def __init__(self, limits: RateLimits, clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.clock = clock
        self.lock = threading.Lock()
        self.windows = {
            "workspace_minute": _Window(limits.per_minute, MINUTE),
            "workspace_day": _Window(limits.per_day, DAY),
            "client": _Window(limits.per_client_minute, MINUTE),
            "global": _Window(limits.global_per_minute, MINUTE),
        }

    def check(self, workspace_id: str, client_id: str = "", *, server_key: bool = True) -> Decision:
        """Count one question if every limit allows it; otherwise refuse it and count nothing."""
        keys = {"workspace_minute": workspace_id, "workspace_day": workspace_id, "client": client_id,
                "global": "app" if server_key else ""}
        with self.lock:
            now = self.clock()
            for name, key in keys.items():
                wait = self.windows[name].wait(key, now)
                if wait > 0:
                    return Decision(False, _message(name, self.windows[name].limit, wait), wait, name)
            for name, key in keys.items():
                self.windows[name].record(key, now)
        return Decision(True)


def _message(limit_name: str, limit: int, wait: float) -> str:
    when = _duration(wait)
    if limit_name == "workspace_day":
        return f"This workspace has asked {limit} questions today, the daily limit. Try again in {when}."
    if limit_name == "global":
        return (f"The app is answering a lot of questions right now. Try again in {when}, "
                "or use your own API key in Settings.")
    if limit_name == "client":
        return f"Too many questions from your network in the last minute. Try again in {when}."
    return f"You've asked {limit} questions in the last minute, the limit. Try again in {when}."


def _duration(seconds: float) -> str:
    seconds = max(1, int(seconds + 0.999))
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = (seconds + 59) // 60
    if minutes < 90:
        return f"{minutes} minutes"
    return f"{(minutes + 59) // 60} hours"


_SALT = os.urandom(16)        # new each process: the hashes can't be matched to addresses later


def client_key(address: str | None) -> str:
    """A short hash of a network address for counting; the address itself is not kept."""
    if not address:
        return ""
    return hashlib.sha256(_SALT + str(address).encode()).hexdigest()[:16]


_shared: dict[RateLimits, RequestLimiter] = {}
_shared_lock = threading.Lock()


def shared_limiter(get_secret: GetSecret) -> RequestLimiter:
    """The one limiter for this process and these settings."""
    limits = RateLimits.from_settings(get_secret)
    with _shared_lock:
        if limits not in _shared:
            _shared[limits] = RequestLimiter(limits)
        return _shared[limits]
