"""Append-only event journal with optional durable JSONL persistence."""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ExecutionJournal:
    """One run per journal. Disk journals use exclusive creation, never overwrite.

    Flush/fsync precedes tool execution for action-start records. Exceptions are
    intentionally propagated: execution must not proceed without its audit trail.
    Persisted output may contain task data; the host selects the storage location.
    """

    def __init__(self, path: Path | None = None):
        self.run_id = uuid4().hex
        self.path = Path(path) if path else None
        self._entries = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8"):
                pass

    @property
    def entries(self):
        return tuple(json.loads(json.dumps(e)) for e in self._entries)

    def emit(self, event_type, actor, payload):
        with self._lock:
            entry = {"run_id": self.run_id, "sequence": len(self._entries) + 1,
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "event": event_type, "actor": actor, "payload": payload}
            encoded = json.dumps(entry, ensure_ascii=False, default=str, allow_nan=False)
            if self.path:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self._entries.append(json.loads(encoded))

    @staticmethod
    def read(path: Path):
        entries = []
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["sequence"] != len(entries) + 1:
                    raise ValueError("journal sequence is incomplete")
                if entries and entry["run_id"] != entries[0]["run_id"]:
                    raise ValueError("journal contains multiple runs")
                entries.append(entry)
        return tuple(entries)
