"""Durable record of webhook deliveries we have already processed.

Used so that on service restart we can ask GitHub to redeliver any webhooks
that arrived while the VM was offline, and so that the same delivery cannot
trigger a duplicate build if it arrives twice.

The data is a single JSON document written atomically via tmpfile + rename.
History is capped to avoid unbounded growth; GitHub itself only keeps
deliveries for 30 days so older entries are not useful anyway.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class State:
    def __init__(self, path: Path, *, max_history: int = 2000) -> None:
        self.path = path
        self.max_history = max_history
        self._lock = threading.Lock()
        self._completed: list[str] = []
        self._completed_set: set[str] = set()
        if path.exists():
            self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save_locked()

    def _load(self) -> None:
        with self.path.open() as f:
            data = json.load(f)
        self._completed = list(data.get("completed", []))
        self._completed_set = set(self._completed)

    def _save_locked(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump({"completed": self._completed}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def is_completed(self, guid: str) -> bool:
        with self._lock:
            return guid in self._completed_set

    def mark_completed(self, guid: str) -> None:
        with self._lock:
            if guid in self._completed_set:
                return
            self._completed.append(guid)
            self._completed_set.add(guid)
            if len(self._completed) > self.max_history:
                drop = self._completed[: len(self._completed) - self.max_history]
                self._completed = self._completed[-self.max_history:]
                self._completed_set.difference_update(drop)
            self._save_locked()

    def completed(self) -> set[str]:
        with self._lock:
            return set(self._completed_set)
