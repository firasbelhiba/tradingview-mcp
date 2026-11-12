#!/usr/bin/env python3
"""Copy the server's data directory somewhere it can be restored from.

The paper wallets and the alerts are SQLite files, and until this existed
there was one copy of each. A wiped wallet, a bad migration or a mistaken
``rm`` was the end of everyone's history.

    python3 scripts/backup_data.py DATA_DIR BACKUP_DIR [--keep 14]

Makes ``BACKUP_DIR/YYYY-MM-DD_HHMMSS/`` holding a copy of every file in
DATA_DIR at any depth, then deletes dated folders beyond the newest
``--keep``.

Databases are copied with SQLite's own backup API, not ``cp``: the files are
in WAL mode, and a plain copy taken while the server is writing can be torn.
Each copy is then opened and checked with ``PRAGMA integrity_check``, because
a backup nobody has read is a hope, not a backup. Any failure exits non-zero
and leaves the previous backups alone.

Standard library only, so it runs from cron on a host that has nothing else
of this project installed. It protects against mistakes and bad deploys. It
does NOT protect against losing the disk: BACKUP_DIR on the same machine
dies with it, so point it at other storage if that matters.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

_STAMP = "%Y-%m-%d_%H%M%S"
_DB_SUFFIXES = (".sqlite3", ".sqlite", ".db")
_SKIP_SUFFIXES = ("-wal", "-shm", "-journal")


def copy_database(source: Path, target: Path) -> int:
    """A consistent copy of one database. Returns the number of tables in it."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    # immutable: read the copy without taking locks. A plain read-only open of
    # a WAL-mode file still creates -shm and -wal beside it, and a backup
    # folder should hold the files that were backed up and nothing else.
    check = sqlite3.connect(f"file:{target}?mode=ro&immutable=1", uri=True)
    try:
        verdict = check.execute("PRAGMA integrity_check").fetchone()[0]
        if verdict != "ok":
            raise RuntimeError(f"{target.name}: integrity_check said {verdict!r}")
        return check.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    finally:
        check.close()


def backup(data_dir: Path, backup_dir: Path, keep: int = 14, now=None) -> Path:
    """Back up every file in data_dir. Returns the folder that was written."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"no data directory at {data_dir}")
    stamp = time.strftime(_STAMP, time.gmtime(now))
    target = backup_dir / stamp
    working = backup_dir / f".{stamp}.partial"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True)
    copied = []
    try:
        # Every file, at any depth: each person's space is a folder of its
        # own under spaces/, and a backup of the top level alone would have
        # kept the owner's wallet and nobody else's.
        for source in sorted(p for p in data_dir.rglob("*") if p.is_file()):
            if source.name.endswith(_SKIP_SUFFIXES) or source.name.endswith(".tmp"):
                continue
            relative = source.relative_to(data_dir)
            destination = working / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix in _DB_SUFFIXES:
                tables = copy_database(source, destination)
                copied.append(f"{relative.as_posix()} ({tables} tables)")
            else:
                shutil.copy2(source, destination)
                copied.append(relative.as_posix())
        if not copied:
            raise RuntimeError(f"nothing to back up in {data_dir}")
        # Renamed only once complete, so a folder with a date for a name is
        # always a whole backup and never one that was interrupted.
        working.rename(target)
    except BaseException:
        shutil.rmtree(working, ignore_errors=True)
        raise
    print(f"backed up {len(copied)} file(s) to {target}: {', '.join(copied)}")
    prune(backup_dir, keep)
    return target


def prune(backup_dir: Path, keep: int) -> list:
    """Delete dated folders beyond the newest ``keep``. Returns what went."""
    dated = sorted(
        p for p in backup_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and _is_stamp(p.name)
    )
    doomed = dated[:-keep] if keep > 0 else []
    for folder in doomed:
        shutil.rmtree(folder)
    if doomed:
        print(f"removed {len(doomed)} old backup(s), kept {min(keep, len(dated))}")
    return doomed


def _is_stamp(name: str) -> bool:
    try:
        time.strptime(name, _STAMP)
        return True
    except ValueError:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("backup_dir", type=Path)
    parser.add_argument("--keep", type=int, default=14,
                        help="how many backups to keep (default 14)")
    args = parser.parse_args(argv)
    try:
        backup(args.data_dir, args.backup_dir, args.keep)
    except Exception as exc:  # noqa: BLE001
        print(f"BACKUP FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
