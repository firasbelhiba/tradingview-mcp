"""What a person means when they name a coin. Offline.

People say BTC. Measured across fifteen tools and six spellings before this
rule existed: coin_analysis("BTC") returned LSE:BTC, a London-listed stock
at 1.30, even with exchange="BINANCE"; three candle tools charted the Yahoo
ticker BTC, which is an ETF; an alert on BTC was stored against a symbol no
crypto venue has, so it could never fire; and most other tools refused.
Afterwards, none of the 75 cells refused and none answered about another
asset.
"""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from tradingview_mcp.core.services import alert_service, alert_store
from tradingview_mcp.core.services import binance_service as bn
from tradingview_mcp.core.services import crypto_analysis_service as crypto
from tradingview_mcp.core.services import paper_pricing as px
from tradingview_mcp.core.services import screener_service
from tradingview_mcp.core.utils.validators import (
    crypto_pair,
    normalize_tradingview_symbol,
)


# ── The rule ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("typed", ["BTC", "btc", " Btc ", "BTC/USDT", "BTC-USDT",
                                   "btc_usdt", "KUCOIN:BTC", "BINANCE:BTCUSDT",
                                   "BTCUSDT", "BTCUSD", "BTC-USD"])
def test_every_way_of_naming_bitcoin_is_the_same_pair(typed):
    assert crypto_pair(typed, "KUCOIN") == "BTCUSDT"
    assert crypto_pair(typed, "BINANCE") == "BTCUSDT"


@pytest.mark.parametrize("typed, pair", [
    ("sol", "SOLUSDT"), ("ETH", "ETHUSDT"), ("pepe", "PEPEUSDT"),
    ("1INCH", "1INCHUSDT"),
    ("WBTC", "WBTCUSDT"),          # not W quoted in BTC
])
def test_a_coins_name_means_its_usdt_pair(typed, pair):
    assert crypto_pair(typed, "KUCOIN") == pair


@pytest.mark.parametrize("typed", [
    "ETHBTC",        # a pair quoted in bitcoin is already a pair
    "SOLUSDC",
    "HYPEUSDT",      # a pair the bundled list may not have yet
    "BTC.D",         # dominance index, not an asset code
    "BTCUSDT.P",     # TradingView's perpetual suffix
    "GC1!",          # a futures contract
    "USDT",          # the quote asset itself has no USDT pair
])
def test_what_is_not_a_bare_coin_name_is_left_alone(typed):
    assert crypto_pair(typed, "KUCOIN") == typed


def test_a_name_the_list_has_never_heard_of_still_becomes_its_pair():
    """The bundled lists go stale. On a crypto venue a plain code means its
    pair, and guessing wrong there is an honest not-found."""
    assert crypto_pair("NEWLISTING", "KUCOIN") == "NEWLISTINGUSDT"


def test_the_cautious_mode_promotes_only_what_some_crypto_venue_lists():
    """For the Yahoo fallback, where AAPL is a company and BTC is an ETF."""
    assert crypto_pair("BTC", "BINANCE", known_only=True) == "BTCUSDT"
    assert crypto_pair("AAPL", "BINANCE", known_only=True) == "AAPL"
    assert crypto_pair("NEWLISTING", "BINANCE", known_only=True) == "NEWLISTING"


# ── Where the rule applies, and where it must not ─────────────────────────────

def test_on_a_crypto_venue_the_name_resolves():
    assert normalize_tradingview_symbol("btc", "KUCOIN") == "KUCOIN:BTCUSDT"
    assert normalize_tradingview_symbol("SOL", "BINANCE") == "BINANCE:SOLUSDT"
    assert normalize_tradingview_symbol("KUCOIN:BTC", "KUCOIN") == "KUCOIN:BTCUSDT"


def test_on_a_stock_venue_nothing_is_touched():
    """BTC really is a ticker in London, and whoever asks there means it."""
    assert normalize_tradingview_symbol("BTC", "LSE") == "LSE:BTC"
    assert normalize_tradingview_symbol("SOL", "ASX") == "ASX:SOL"
    assert normalize_tradingview_symbol("AAPL", "NASDAQ") == "NASDAQ:AAPL"
    assert normalize_tradingview_symbol("COMI", "EGX") == "EGX:COMI"


def test_a_prefix_someone_typed_is_kept_as_typed():
    assert normalize_tradingview_symbol("LSE:BTC", "KUCOIN") == "LSE:BTC"
    assert normalize_tradingview_symbol("CRYPTOCAP:BTC.D", "KUCOIN") == "CRYPTOCAP:BTC.D"


def test_commodity_aliases_still_win_on_a_crypto_venue():
    assert normalize_tradingview_symbol("XAUUSD", "KUCOIN") == "TVC:GOLD"
    assert normalize_tradingview_symbol("GOLD", "KUCOIN") == "TVC:GOLD"


# ── The four cleaners now agree ───────────────────────────────────────────────

@pytest.mark.parametrize("typed", ["BTC", "btc", "BTC/USDT", "KUCOIN:BTC", "BTCUSDT"])
def test_every_cleaner_gives_the_same_pair(typed):
    assert bn.perp_symbol_for(typed) == "BTCUSDT"
    assert crypto.binance_symbol_for(typed) == "BTCUSDT"
    assert px.normalise_symbol(typed) == "BTCUSDT"
    # A venue someone typed is kept; only the name after it is resolved.
    assert normalize_tradingview_symbol(typed, "BINANCE").split(":")[1] == "BTCUSDT"


def test_the_yahoo_fallback_charts_the_coin_not_the_etf():
    """The Yahoo ticker BTC is the Grayscale Bitcoin Mini Trust."""
    assert crypto.yahoo_crypto_symbol("BTC") == "BTC-USD"
    assert crypto.yahoo_crypto_symbol("AAPL") == "AAPL", "a company stays a company"


def test_a_position_already_in_a_wallet_keeps_its_symbol():
    """Wallets hold pairs. Reading one back must not rename it."""
    for stored in ("BTCUSDT", "ETHUSDT", "HYPEUSDT", "1000PEPEUSDT", "ETHBTC"):
        assert px.normalise_symbol(stored) == stored


# ── An alert that could never fire ────────────────────────────────────────────

@pytest.fixture()
def alerts(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_ALERTS_DB", str(tmp_path / "alerts.sqlite3"))
    alert_store._initialised.clear()
    yield
    alert_store._initialised.clear()


def test_an_alert_is_stored_under_the_symbol_it_will_be_checked_against(alerts):
    """'Tell me when btc goes above 80000' was stored as BTC, a symbol KuCoin
    does not have. It sat armed and reported nothing."""
    out = alert_service.create_alert("btc", "price", "above", 80000.0)
    assert out["symbol"] == "BTCUSDT"
    assert alert_store.list_alerts()[0]["symbol"] == "BTCUSDT"


def test_an_alert_typed_with_a_prefix_or_on_a_stock_venue_is_kept(alerts):
    a = alert_service.create_alert("LSE:BTC", "price", "above", 2.0)
    b = alert_service.create_alert("COMI", "price", "above", 90.0, exchange="EGX")
    assert a["symbol"] == "LSE:BTC"
    assert b["symbol"] == "COMI"


# ── When the answer really is a different kind of asset ───────────────────────

def _analysis():
    ind = defaultdict(lambda: None)
    ind.update({
        "close": 42.5, "open": 41.0, "high": 43.0, "low": 40.5, "volume": 1_000_000.0,
        "RSI": 55.0, "ATR": 1.2, "change": 2.1, "Recommend.All": 0.4,
        "MACD.macd": 0.5, "MACD.signal": 0.3, "EMA20": 41.8, "EMA50": 40.9,
        "SMA20": 41.5, "BB.upper": 44.0, "BB.lower": 39.0,
    })
    return SimpleNamespace(indicators=ind, summary={"RECOMMENDATION": "BUY"},
                           time=None, exchange="NASDAQ", symbol="AAPL")


def test_a_stock_answered_from_a_crypto_request_says_so_first(monkeypatch):
    """A company typed without its exchange is still found, as before. But
    the result used to carry that fact in a note at its very end."""
    asked = []

    def fake(screener, interval, symbols):
        asked.extend(symbols)
        return {s: _analysis() for s in symbols if s.startswith("NASDAQ:")}

    monkeypatch.setattr(screener_service, "get_multiple_analysis", fake)
    out = screener_service.analyze_coin("AAPL", "kucoin", "1D")
    assert list(out)[0] == "different_asset_warning", "first, not buried"
    assert "NOT a coin" in out["different_asset_warning"]
    assert "NASDAQ:AAPL" in out["different_asset_warning"]
    assert out["resolved_exchange"] == "NASDAQ"
    assert asked[0] == "KUCOIN:AAPLUSDT", "the coin reading is tried first"


def test_a_coin_is_never_answered_from_a_stock_exchange(monkeypatch):
    """The regression itself: BTC on a crypto venue is bitcoin."""
    def fake(screener, interval, symbols):
        return {s: _analysis() for s in symbols if s == "KUCOIN:BTCUSDT"}

    monkeypatch.setattr(screener_service, "get_multiple_analysis", fake)
    out = screener_service.analyze_coin("BTC", "kucoin", "1D")
    assert "different_asset_warning" not in out
    assert "resolved_exchange" not in out
    assert out["symbol"] == "KUCOIN:BTCUSDT"


def test_moving_between_two_crypto_venues_is_not_called_a_different_asset(monkeypatch):
    """HYPEUSDT on another crypto exchange is the same coin; that keeps its
    ordinary note."""
    def fake(screener, interval, symbols):
        return {s: _analysis() for s in symbols if s.startswith("KUCOIN:")}

    monkeypatch.setattr(screener_service, "get_multiple_analysis", fake)
    out = screener_service.analyze_coin("HYPEUSDT", "binance", "1D")
    assert "different_asset_warning" not in out
    assert out["resolved_exchange"] == "KUCOIN"
    assert "resolution_note" in out
