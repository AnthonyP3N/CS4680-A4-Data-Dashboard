"""Append-only audit log of every action the agent takes.

Each event is one JSON line in logs/audit.jsonl, tagged with a session id,
so the full trail (question -> LLM output -> validation -> execution ->
confirmation) can be reconstructed for any request.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.adapters import redact

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"
_lock = threading.Lock()


class AuditLog:
    def __init__(self, session_id: str, path: Path = LOG_PATH, workspace_id: str | None = None):
        self.session_id = session_id
        self.path = path
        self.workspace_id = workspace_id     # when set, records carry it and read() is scoped to it
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, *, request_id: str | None = None, status: str = "ok", **details) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": self.session_id,
            "workspace": self.workspace_id,
            "request": request_id,
            "event": event,
            "status": status,
            # Secrets never reach the log: secret-looking keys, URL credentials and API-key shapes are masked
            **redact({k: v for k, v in details.items() if v is not None}),
        }
        line = json.dumps(record, default=str)
        with _lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record

    def _mine(self, record: dict) -> bool:
        """This workspace's records (across sessions) when a workspace is set, else this session's."""
        if self.workspace_id is not None:
            return record.get("workspace") == self.workspace_id
        return record.get("session") == self.session_id

    def read(self, all_sessions: bool = False, limit: int = 500) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all_sessions or self._mine(r):
                    rows.append(r)
        rows = rows[-limit:][::-1]  # newest first
        return pd.DataFrame(rows)
