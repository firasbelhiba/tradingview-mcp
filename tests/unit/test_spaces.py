"""A private space per person, and a tool set per deployment. Offline.

On the hosted server there was one wallet and one list of alerts for everyone
holding the address: a friend asking for "my wallet" was shown the owner's
open short and could have closed it.
"""
import asyncio
import importlib.util
import json
import pathlib
import sqlite3

import pytest

from tradingview_mcp import server
from tradingview_mcp.core import limits, spaces
from tradingview_mcp.core.services import alert_service, alert_store
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_PAPER_DB", raising=False)
    monkeypatch.delenv("TRADINGVIEW_MCP_ALERTS_DB", raising=False)
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(tmp_path))
    store._initialised.clear()
    alert_store._initialised.clear()
    store.select_wallet(None)
    yield
    store.select_wallet(None)
    store._initialised.clear()
    alert_store._initialised.clear()


def _prices(monkeypatch):
    monkeypatch.setattr(svc.px, "mark_price", lambda s, m: (100.0, "test"))
    monkeypatch.setattr(svc.px, "bars_since", lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(svc.px, "funding_cost", lambda *a, **k: (0.0, 0))


class _In:
    """Run a block as if the call had arrived on a person's link."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.token = spaces.CURRENT.set(self.name)

    def __exit__(self, *exc):
        spaces.CURRENT.reset(self.token)


# ── The data follows the space ────────────────────────────────────────────────

def test_the_owner_keeps_exactly_the_files_they_had(tmp_path):
    assert store.db_path() == tmp_path / "paper_trading.sqlite3"
    assert alert_store.db_path() == tmp_path / "alerts.sqlite3"


def test_a_space_has_a_directory_of_its_own(tmp_path):
    with _In("ahmed"):
        assert store.db_path() == tmp_path / "spaces" / "ahmed" / "paper_trading.sqlite3"
        assert alert_store.db_path() == tmp_path / "spaces" / "ahmed" / "alerts.sqlite3"
        assert store.db_path("scalps").name == "paper-scalps.sqlite3"
        assert store.db_path("scalps").parent == tmp_path / "spaces" / "ahmed"


def test_two_people_cannot_see_or_touch_each_others_wallet(monkeypatch):
    """The regression itself."""
    _prices(monkeypatch)
    svc.reset_wallet(10000.0)
    svc.open_trade("BTCUSDT", "short", 6500.0, "futures", 2)

    with _In("ahmed"):
        assert svc.wallet_status()["error"]["code"] == "NO_DATA", "no wallet here yet"
        svc.reset_wallet(500.0)
        svc.open_trade("ETHUSDT", "long", 100.0)
        mine = svc.wallet_status()
        assert [p["symbol"] for p in mine["open_positions"]] == ["ETHUSDT"]
        assert svc.close_trade(1)["closed"]["symbol"] == "ETHUSDT", "their own position 1"

    with _In("sami"):
        assert svc.wallet_status()["error"]["code"] == "NO_DATA"

    owner = svc.wallet_status()
    assert [p["symbol"] for p in owner["open_positions"]] == ["BTCUSDT"]
    assert owner["wallet"]["starting_balance"] == 10000.0


def test_named_wallets_and_their_limit_are_per_space(monkeypatch):
    _prices(monkeypatch)
    monkeypatch.setattr(limits, "MAX_WALLETS", 2)
    svc.reset_wallet(100.0, wallet="one")
    svc.reset_wallet(100.0, wallet="two")
    with _In("ahmed"):
        assert store.list_wallet_names() == []
        assert "wallet" in svc.reset_wallet(100.0, wallet="one"), "their own count starts at zero"
        assert store.list_wallet_names() == ["one"]
    assert sorted(store.list_wallet_names()) == ["one", "two"]


def test_alerts_are_per_space_too():
    alert_service.create_alert("BTC", "price", "above", 99999.0)
    with _In("ahmed"):
        assert alert_store.list_alerts() == []
        alert_service.create_alert("ETH", "price", "above", 9999.0)
        assert [a["symbol"] for a in alert_store.list_alerts()] == ["ETHUSDT"]
    assert [a["symbol"] for a in alert_store.list_alerts()] == ["BTCUSDT"]


def test_an_explicit_database_path_is_scoped_the_same_way(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_PAPER_DB", str(tmp_path / "custom" / "w.sqlite3"))
    monkeypatch.setenv("TRADINGVIEW_MCP_ALERTS_DB", str(tmp_path / "custom" / "a.sqlite3"))
    assert store.db_path() == tmp_path / "custom" / "w.sqlite3"
    with _In("ahmed"):
        assert store.db_path() == tmp_path / "custom" / "spaces" / "ahmed" / "w.sqlite3"
        assert alert_store.db_path() == tmp_path / "custom" / "spaces" / "ahmed" / "a.sqlite3"


# ── The registry ──────────────────────────────────────────────────────────────

def test_adding_a_person_gives_a_long_random_key_that_opens_their_space():
    key = spaces.add("Ahmed")
    assert len(key) >= 20 and spaces.lookup(key) == "ahmed"
    assert spaces.link_suffix(key) == f"/u/{key}/mcp"


def test_a_key_nobody_issued_opens_nothing():
    spaces.add("ahmed")
    for guess in ("", "ahmed", "x" * 24, "../../etc"):
        assert spaces.lookup(guess) is None


def test_the_registry_is_reread_when_it_changes_so_no_restart_is_needed():
    assert spaces.lookup("anything-at-all-1234567") is None
    key = spaces.add("ahmed")
    assert spaces.lookup(key) == "ahmed"
    assert spaces.remove("ahmed") is True
    assert spaces.lookup(key) is None, "removing the entry closes the link at once"
    assert spaces.remove("ahmed") is False


def test_a_name_is_used_once_and_must_be_a_safe_directory_name():
    spaces.add("ahmed")
    with pytest.raises(ValueError):
        spaces.add("ahmed")
    for bad in ("", "../x", "a b", "A/b", "x" * 40, ".hidden"):
        with pytest.raises(ValueError):
            spaces.add(bad)


def test_a_hand_edited_registry_cannot_point_outside_the_data_directory(tmp_path):
    (tmp_path / spaces.REGISTRY_NAME).write_text(json.dumps({"spaces": {
        "k" * 24: {"name": "../../somewhere"},
        "j" * 24: {"name": "fine"},
        "short": {"name": "tooshortakey"},
    }}))
    assert spaces.lookup("k" * 24) is None
    assert spaces.lookup("short") is None
    assert spaces.lookup("j" * 24) == "fine"


def test_a_broken_registry_closes_every_link_rather_than_opening_any(tmp_path):
    (tmp_path / spaces.REGISTRY_NAME).write_text("{ not json")
    assert spaces.lookup("k" * 24) is None


def test_removing_a_person_keeps_their_data(monkeypatch, tmp_path):
    _prices(monkeypatch)
    spaces.add("ahmed")
    with _In("ahmed"):
        svc.reset_wallet(500.0)
    spaces.remove("ahmed")
    assert (tmp_path / "spaces" / "ahmed" / "paper_trading.sqlite3").exists()


# ── Routing a request ─────────────────────────────────────────────────────────

def _route(path, headers=()):
    """Send one request through the router. -> (what the app saw, or the status)."""
    seen, sent = {}, []

    async def app(scope, receive, send):
        seen.update(path=scope["path"], raw_path=scope.get("raw_path"),
                    headers=dict(scope["headers"]))

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": path, "raw_path": path.encode(),
             "headers": list(headers)}
    asyncio.run(spaces.SpaceRouter(app)(scope, None, send))
    return seen, next((m["status"] for m in sent if "status" in m), None)


def test_a_persons_link_reaches_the_server_as_the_plain_path_marked_with_their_space():
    key = spaces.add("ahmed")
    seen, status = _route(f"/u/{key}/mcp")
    assert status is None
    assert seen["path"] == "/mcp" and seen["raw_path"] == b"/mcp"
    assert seen["headers"][spaces.HEADER] == b"ahmed"


def test_the_owners_link_is_untouched_and_carries_no_space():
    seen, _ = _route("/mcp")
    assert seen["path"] == "/mcp"
    assert spaces.HEADER not in seen["headers"]


def test_a_space_cannot_be_claimed_by_sending_the_header():
    """Otherwise anyone on the owner's link could name the space they wanted."""
    spaces.add("ahmed")
    seen, _ = _route("/mcp", [(b"X-TV-Space", b"ahmed"), (b"x-tv-space", b"ahmed")])
    assert spaces.HEADER not in {k.lower() for k in seen["headers"]}
    key = spaces.add("sami")
    seen, _ = _route(f"/u/{key}/mcp", [(b"x-tv-space", b"ahmed")])
    assert seen["headers"][spaces.HEADER] == b"sami", "the link decides, not the client"


def test_a_wrong_key_is_an_ordinary_404_and_never_reaches_the_server():
    spaces.add("ahmed")
    seen, status = _route("/u/not-a-real-key-000000000/mcp")
    assert status == 404 and seen == {}


def test_other_traffic_passes_straight_through():
    seen, status = _route("/health")
    assert seen["path"] == "/health" and status is None


def test_the_space_is_read_from_the_request_that_carried_the_call():
    class Request:
        def __init__(self, headers):
            self.headers = headers

    assert spaces.from_request(Request({"x-tv-space": "ahmed"})) == "ahmed"
    assert spaces.from_request(Request({})) is None
    assert spaces.from_request(Request({"x-tv-space": "../etc"})) is None
    assert spaces.from_request(None) is None


def test_outside_an_http_request_there_is_no_space():
    """A local stdio install: every path is exactly what it was."""
    assert server._current_space() is None


def test_every_tool_is_wrapped_so_none_can_forget_whose_call_it_is():
    assert server._SCOPED_TOOL_COUNT == len(server.mcp._tool_manager.list_tools())


def test_the_audit_line_names_the_space_and_never_the_key(capsys):
    key = spaces.add("ahmed")
    with _In("ahmed"):
        limits.audit("paper_wallet_status", 0.2, "ok")
    limits.audit("paper_wallet_status", 0.2, "ok")
    first, second = capsys.readouterr().err.strip().splitlines()
    assert first.endswith("outcome=ok space=ahmed")
    assert second.endswith("outcome=ok")
    assert key not in first


# ── The operator's command ────────────────────────────────────────────────────

def test_the_command_creates_lists_and_removes(capsys):
    assert spaces.main(["add", "ahmed"]) == 0
    out = capsys.readouterr().out
    suffix = next(line.strip() for line in out.splitlines() if line.strip().startswith("/u/"))
    assert spaces.lookup(suffix.split("/")[2]) == "ahmed"
    assert spaces.main(["list"]) == 0
    assert "ahmed" in capsys.readouterr().out
    assert spaces.main(["add", "ahmed"]) == 1, "a second one is refused, not overwritten"
    assert spaces.main(["remove", "ahmed"]) == 0
    assert spaces.main(["remove", "ahmed"]) == 1


# ── The backup includes everyone ──────────────────────────────────────────────

def test_a_backup_contains_every_persons_wallet_not_only_the_owners(monkeypatch, tmp_path):
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backup_data.py"
    spec = importlib.util.spec_from_file_location("backup_data_for_spaces", path)
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)

    _prices(monkeypatch)
    data = tmp_path / "data"
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(data))
    svc.reset_wallet(10000.0)
    spaces.add("ahmed")
    with _In("ahmed"):
        svc.reset_wallet(500.0)
        svc.open_trade("ETHUSDT", "long", 100.0)

    folder = bk.backup(data, tmp_path / "backups")
    copy = folder / "spaces" / "ahmed" / "paper_trading.sqlite3"
    assert copy.exists() and (folder / "paper_trading.sqlite3").exists()
    conn = sqlite3.connect(copy)
    assert conn.execute("select symbol from positions").fetchall() == [("ETHUSDT",)]
    conn.close()
    assert (folder / spaces.REGISTRY_NAME).exists(), "the links are restorable too"


# ── Which tools to offer ──────────────────────────────────────────────────────

ALL = [t.name for t in server.mcp._tool_manager.list_tools()]


def test_with_no_tool_set_named_every_tool_is_offered():
    assert server.hidden_tools("", False, ALL) == set()
    assert server._HIDDEN_TOOL_COUNT == 0 and len(ALL) == 80


def test_the_crypto_set_leaves_out_stocks_and_exchange_futures():
    hidden = server.hidden_tools("crypto", True, ALL)
    assert len(hidden) == 18
    assert all(n.startswith(("egx_", "stock_", "futures_")) or n == "yahoo_price"
               for n in hidden)
    for kept in ("coin_analysis", "crypto_trade_plan", "paper_wallet_status",
                 "lead_traders", "crypto_bubbles", "market_brief", "get_started",
                 "funding_screener", "backtest_strategy"):
        assert kept not in hidden


def test_tools_that_need_a_news_key_are_left_out_only_while_there_is_none():
    """Without the key they can only answer 'Unavailable'."""
    news = {"market_sentiment", "financial_news", "combined_analysis"}
    assert news <= server.hidden_tools("crypto", False, ALL)
    assert not news & server.hidden_tools("crypto", True, ALL)
    assert len(server.hidden_tools("crypto", False, ALL)) == 21


def test_the_yardsticks_answers_all_survive_the_crypto_set():
    """Otherwise the hosted server could not be scored against its own test."""
    cases = json.loads((pathlib.Path(__file__).resolve().parents[2] / "evals"
                        / "tool_selection" / "cases.json").read_text(encoding="utf-8"))
    hidden = server.hidden_tools("crypto", False, ALL)
    for case in cases["cases"]:
        assert not set(case["best"]) & hidden, case["id"]
