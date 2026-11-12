"""The backup script. Offline, against real SQLite files in WAL mode."""
import importlib.util
import pathlib
import sqlite3

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backup_data.py"
_spec = importlib.util.spec_from_file_location("backup_data", _PATH)
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)


def _wallet(path, rows=3):
    """A database in WAL mode with an open connection, as the server keeps it."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE positions (id INTEGER PRIMARY KEY, symbol TEXT)")
    conn.executemany("INSERT INTO positions (symbol) VALUES (?)",
                     [(f"COIN{i}USDT",) for i in range(rows)])
    conn.commit()
    return conn


@pytest.fixture()
def dirs(tmp_path):
    data, backups = tmp_path / "data", tmp_path / "backups"
    data.mkdir()
    return data, backups


def test_a_backup_can_be_opened_and_holds_what_was_there(dirs):
    data, backups = dirs
    live = _wallet(data / "paper_trading.sqlite3")
    (data / "audit.log").write_text("2026-11-12T00:00:00Z [audit] tool=x\n")
    try:
        folder = bk.backup(data, backups, now=1_800_000_000)
    finally:
        live.close()
    assert folder.name == "2027-01-15_080000"
    copy = sqlite3.connect(folder / "paper_trading.sqlite3")
    assert copy.execute("SELECT count(*) FROM positions").fetchone()[0] == 3
    copy.close()
    assert (folder / "audit.log").read_text().startswith("2026-11-12")


def test_rows_still_in_the_write_ahead_log_are_in_the_copy(dirs):
    """Why this is not cp: with the server mid-write, the newest rows are in
    the -wal file, and copying the main file alone would silently lose them."""
    data, backups = dirs
    live = _wallet(data / "paper_trading.sqlite3", rows=2)
    live.execute("PRAGMA wal_autocheckpoint=0")
    live.execute("INSERT INTO positions (symbol) VALUES ('LATEUSDT')")
    live.commit()
    assert (data / "paper_trading.sqlite3-wal").stat().st_size > 0
    try:
        folder = bk.backup(data, backups)
    finally:
        live.close()
    copy = sqlite3.connect(folder / "paper_trading.sqlite3")
    symbols = {row[0] for row in copy.execute("SELECT symbol FROM positions")}
    copy.close()
    assert "LATEUSDT" in symbols
    leftovers = [f.name for f in folder.iterdir() if f.name.endswith(("-wal", "-shm"))]
    assert leftovers == [], "the copy is one whole file, with nothing beside it"


def test_only_the_newest_are_kept(dirs):
    data, backups = dirs
    _wallet(data / "alerts.sqlite3").close()
    for day in range(5):
        bk.backup(data, backups, keep=3, now=1_800_000_000 + day * 86_400)
    names = sorted(p.name for p in backups.iterdir())
    assert names == ["2027-01-17_080000", "2027-01-18_080000", "2027-01-19_080000"]


def test_pruning_never_touches_a_folder_it_did_not_make(dirs):
    data, backups = dirs
    _wallet(data / "alerts.sqlite3").close()
    (backups / "my-own-notes").mkdir(parents=True)
    for day in range(3):
        bk.backup(data, backups, keep=1, now=1_800_000_000 + day * 86_400)
    assert (backups / "my-own-notes").is_dir()


def test_a_broken_database_fails_loudly_and_leaves_no_half_backup(dirs):
    data, backups = dirs
    (data / "paper_trading.sqlite3").write_bytes(b"this is not a database" * 50)
    with pytest.raises(Exception):
        bk.backup(data, backups)
    assert not [p for p in backups.iterdir()], "no dated folder, no leftovers"
    assert bk.main([str(data), str(backups)]) == 1, "cron sees a failure"


def test_an_empty_or_missing_data_directory_is_an_error_not_a_success(dirs, tmp_path):
    data, backups = dirs
    assert bk.main([str(data), str(backups)]) == 1
    assert bk.main([str(tmp_path / "nowhere"), str(backups)]) == 1


def test_it_needs_nothing_but_the_standard_library():
    """It runs from cron on a host with none of this project installed."""
    source = _PATH.read_text(encoding="utf-8")
    assert "tradingview_mcp" not in source.split('"""', 2)[2]
