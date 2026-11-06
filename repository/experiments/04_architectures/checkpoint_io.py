"""Crash-tolerant JSON Lines checkpoint readers for long experiment campaigns.

A JSONL file is append-only.  A process interruption may leave only its final
record truncated; that single tail is safely discarded on resume.  Corruption
in any newline-terminated record is treated as data loss and is never silently
skipped.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON manifest after flushing its temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        # Persist the directory entry where supported.
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def read_jsonl_checkpoint(
    path: Path,
    *,
    repair: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Read a JSONL checkpoint and optionally repair only an incomplete tail.

    Returns records and a diagnostic dictionary.  In read-only mode, an
    incomplete final record is ignored but not modified.  In repair mode the
    file is truncated exactly at the beginning of that incomplete record.
    Valid final JSON without a newline is retained and normalized when repair
    is enabled.  Malformed newline-terminated records always raise.
    """
    path = Path(path)
    if not path.exists():
        return [], {"discarded_partial_record": False, "discarded_bytes": 0,
                    "added_final_newline": False}

    raw = path.read_bytes()
    if not raw:
        return [], {"discarded_partial_record": False, "discarded_bytes": 0,
                    "added_final_newline": False}

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    discarded_partial = False
    discarded_bytes = 0
    added_final_newline = False
    lines = raw.splitlines(keepends=True)

    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        terminated = line.endswith(b"\n")
        payload = line.rstrip(b"\r\n")
        if not payload.strip():
            offset += len(line)
            continue
        try:
            record = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_last and not terminated:
                discarded_partial = True
                discarded_bytes = len(raw) - offset
                if repair:
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
                        stream.flush()
                        os.fsync(stream.fileno())
                break
            raise ValueError(f"Corrupt JSONL checkpoint at {path}, byte offset {offset}: {exc}") from exc

        if not isinstance(record, dict):
            raise ValueError(f"JSONL checkpoint record at {path}, byte offset {offset} is not an object")
        key = record.get("run_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"JSONL checkpoint record at {path}, byte offset {offset} has no valid run_key")
        if key in seen:
            raise ValueError(f"Duplicate run_key in JSONL checkpoint {path}: {key}")
        seen.add(key)
        records.append(record)
        offset += len(line)

        if is_last and not terminated and repair:
            with path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            added_final_newline = True

    return records, {
        "discarded_partial_record": discarded_partial,
        "discarded_bytes": discarded_bytes,
        "added_final_newline": added_final_newline,
    }
