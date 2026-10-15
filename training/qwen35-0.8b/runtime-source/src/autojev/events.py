import fcntl
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

START = time.monotonic()


def record(kind: str, **fields: object) -> dict[str, object]:
    event: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "process_elapsed_seconds": time.monotonic() - START,
        "kind": kind,
        **fields,
    }
    path = Path(os.getenv("AUTOJEV_EVENTS", "progress/events.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return event
