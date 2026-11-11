"""Named wallets, one SQLite file each. Offline.

The wallet table is `CHECK (id = 1)` and positions carry no wallet column,
so two wallets cannot share a database without a schema rewrite. They get a
file each instead, which makes isolation total rather than a WHERE clause
somebody might forget.

The name becomes a filename on a server that answers from the open
internet, so most of what is pinned here is the refusal: a name that could
walk out of the data directory must never reach the filesystem.
"""
import pytest

from tradingview_mcp.core.services import paper_orders as po
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_PAPER_DB", raising=False)
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(tmp_path))
    store._initialised.clear()
    store.select_wallet(None)
    yield
    store.select_wallet(None)
    store._initialised.clear()


def _patch(monkeypatch, price=100.0):
    for mod in (svc.px, po.px):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (price, "test"))
        monkeypatch.setattr(mod, "bars_since",
                            lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(svc.px, "funding_cost", lambda *a, **k: (0.0, 0))


# ── The refusal, which is the security boundary ───────────────────────────────

@pytest.mark.parametrize("name", [
    "../escape", "../../etc/passwd", "a/b", "a\\b", ".hidden", "..",
    "has space", "Tab\tchar", "x" * 33, "sem;colon", "quote'name", "-leading",
])
def test_a_name_that_could_leave_the_directory_is_refused(name):
    with pytest.raises(store.BadWalletName):
        store.normalise_wallet_name(name)


@pytest.mark.parametrize("name", ["swing", "risky-ideas", "test_2", "a", "9lives"])
def test_an_ordinary_name_is_accepted(name):
    assert store.normalise_wallet_name(name) == name


def test_the_refusal_reaches_the_caller_as_an_envelope(monkeypatch):
    """It has to be an error a client can read, not a traceback."""
    _patch(monkeypatch)
    out = svc.wallet_status(wallet="../../etc/passwd")
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "becomes a filename" in out["error"]["message"]


def test_a_bad_name_creates_no_file(tmp_path, monkeypatch):
    """Validation before the first database touch, not after."""
    _patch(monkeypatch)
    svc.reset_wallet(1000.0, wallet="../sneaky")
    assert list(tmp_path.glob("**/*.sqlite3")) == []


def test_case_and_spacing_do_not_make_a_new_wallet():
    assert store.normalise_wallet_name("  SWING  ") == "swing"


def test_the_default_wallet_has_several_spellings():
    for spelling in (None, "", "   ", "main", "MAIN"):
        assert store.normalise_wallet_name(spelling) is None


# ── Isolation ─────────────────────────────────────────────────────────────────

def test_two_wallets_do_not_see_each_other(monkeypatch):
    _patch(monkeypatch)
    svc.reset_wallet(10000.0, wallet=None)
    svc.reset_wallet(500.0, wallet="risky")
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 1.0, wallet=None)

    main = svc.wallet_status(wallet=None)
    risky = svc.wallet_status(wallet="risky")
    assert main["wallet"]["starting_balance"] == 10000.0
    assert risky["wallet"]["starting_balance"] == 500.0
    assert main["open_count"] == 1
    assert risky["open_count"] == 0, "a position must not cross wallets"


def test_resting_orders_are_isolated_too(monkeypatch):
    """Orders were added after wallets existed, so this is worth pinning."""
    _patch(monkeypatch)
    svc.reset_wallet(10000.0, wallet="alpha")
    svc.reset_wallet(10000.0, wallet="beta")
    po.place_order("BTCUSDT", "long", 1000.0, 90.0, wallet="alpha")

    assert po.list_orders(wallet="alpha")["counts"]["pending"] == 1
    assert po.list_orders(wallet="beta")["counts"]["pending"] == 0
    assert svc.wallet_status(wallet="beta")["wallet"]["cash"] == 10000.0, (
        "alpha's held margin must not touch beta's cash"
    )


def test_wiping_one_wallet_leaves_the_other_alone(monkeypatch):
    _patch(monkeypatch)
    svc.reset_wallet(10000.0, wallet="keep")
    svc.open_trade("BTCUSDT", "long", 100.0, "futures", 1.0, wallet="keep")
    svc.reset_wallet(777.0, wallet="burn")

    assert svc.wallet_status(wallet="keep")["open_count"] == 1
    assert svc.wallet_status(wallet="keep")["wallet"]["starting_balance"] == 10000.0


def test_an_unknown_wallet_reports_no_wallet_rather_than_inventing_one(monkeypatch):
    _patch(monkeypatch)
    out = svc.wallet_status(wallet="never-made")
    assert "error" in out, "it must not quietly create an empty wallet"


# ── Where the files land ──────────────────────────────────────────────────────

def test_the_default_wallet_keeps_its_original_filename(tmp_path):
    """An installation that predates named wallets must open the same file."""
    assert store.db_path(None).name == "paper_trading.sqlite3"


def test_a_named_wallet_gets_its_own_file(tmp_path):
    assert store.db_path("swing").name == "paper-swing.sqlite3"
    assert store.db_path("swing").parent == tmp_path


def test_an_explicit_db_path_still_names_its_siblings(monkeypatch, tmp_path):
    """TRADINGVIEW_MCP_PAPER_DB points at one file; named wallets sit beside
    it rather than ignoring the setting."""
    monkeypatch.setenv("TRADINGVIEW_MCP_PAPER_DB", str(tmp_path / "custom.sqlite3"))
    assert store.db_path(None).name == "custom.sqlite3"
    assert store.db_path("swing").name == "paper-swing.sqlite3"
    assert store.db_path("swing").parent == tmp_path


def test_listing_shows_what_exists_and_nothing_else(monkeypatch):
    _patch(monkeypatch)
    assert store.list_wallet_names() == []
    svc.reset_wallet(1000.0, wallet=None)
    svc.reset_wallet(1000.0, wallet="swing")
    assert store.list_wallet_names() == ["main", "swing"]

    payload = svc.wallet_names()
    assert payload["wallets"] == ["main", "swing"]
    assert payload["default"] == "main"
    assert "every caller can name every wallet" in payload["scope"], (
        "separation is not privacy, and the result has to say so"
    )


def test_the_selection_does_not_leak_between_calls(monkeypatch):
    """Each call sets its own; a later call with no name gets the default."""
    _patch(monkeypatch)
    svc.reset_wallet(1000.0, wallet="swing")
    assert store.active_wallet() == "swing"
    svc.reset_wallet(2000.0, wallet=None)
    assert store.active_wallet() == "main"
    assert svc.wallet_status(wallet="swing")["wallet"]["starting_balance"] == 1000.0
