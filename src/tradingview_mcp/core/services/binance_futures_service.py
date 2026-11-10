"""
Binance Futures Service — perpetual positioning data.

The server's other 47 tools all read price and volume. None reads
*positioning*: who is on which side, what they are paying to stay there, and
whether open interest is building or unwinding. TradingView's scanner cannot
supply it — ``funding_rate`` and ``open_interest`` come back null even for
perp tickers like ``BINANCE:BTCUSDT.P`` that otherwise resolve — so this
module talks to Binance's public futures API directly.

No API key: every endpoint used here is public and unauthenticated.

Resilience mirrors the TradingView layer rather than reinventing it: a short
fresh cache, bounded retries with jittered backoff, a stale-while-error
fallback so a blip serves the last good answer instead of failing, and the
proxy opener the rest of the package already uses when a direct call is
blocked.

Environment:
  TRADINGVIEW_MCP_BINANCE_CACHE_TTL   default 60    (seconds, fresh cache)
  TRADINGVIEW_MCP_BINANCE_STALE_TTL   default 1800  (30 min, fallback cache)
  TRADINGVIEW_MCP_BINANCE_RETRIES     default 2
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error

_BASE = "https://fapi.binance.com"
_UA = "tradingview-mcp/0.9.0 positioning"
_TIMEOUT = 15

# Binance's data endpoints accept their own period vocabulary. Our timeframe
# strings differ in case ("1D" vs "1d"), and the sub-5m intervals this server
# supports have no equivalent, so they round up to the nearest available.
_PERIOD_MAP = {
    "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "2h": "2h",
    "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1d", "1D": "1d",
    "1W": "1d", "1M": "1d",
}

# Funding is quoted per funding interval, not per day. Binance pays most
# perps every 8 hours but some every 4, and the interval is not published on
# the premiumIndex response — it is inferred from the funding history's
# timestamps so the annualised figure is right for the symbol in hand.
_DEFAULT_FUNDING_INTERVAL_H = 8.0

# Interpretation thresholds live in positioning_service, which owns the
# verdicts. This module only fetches.

_cache: Dict[Tuple, Tuple[float, Any]] = {}
_cache_lock = Lock()


def _ttl(name: str, default: str) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return float(default)


def _cache_get(key: Tuple, allow_stale: bool = False) -> Optional[Any]:
    fresh = _ttl("TRADINGVIEW_MCP_BINANCE_CACHE_TTL", "60")
    stale = _ttl("TRADINGVIEW_MCP_BINANCE_STALE_TTL", "1800")
    with _cache_lock:
        hit = _cache.get(key)
    if not hit:
        return None
    ts, value = hit
    age = time.time() - ts
    if age <= fresh:
        return value
    if allow_stale and age <= stale:
        return value
    return None


def _cache_put(key: Tuple, value: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), value)


class BinanceUnavailable(RuntimeError):
    """Binance could not be reached, or answered with an error status."""


def _fetch(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET a public endpoint, with cache, retries and a stale fallback."""
    query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    url = f"{_BASE}{path}" + (f"?{query}" if query else "")
    key = ("binance", path, query)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        retries = max(0, int(os.environ.get("TRADINGVIEW_MCP_BINANCE_RETRIES", "2")))
    except ValueError:
        retries = 2

    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        if attempt:
            # Jittered backoff; Binance rate-limits by weight per minute and
            # answers 429/418 when a client runs hot.
            time.sleep(min(8.0, 1.0 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
        for opener in _openers():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                with opener(req, timeout=_TIMEOUT) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                _cache_put(key, payload)
                return payload
            except Exception as exc:  # noqa: BLE001 - retried below
                last_exc = exc

    # Serve the last good answer rather than failing outright: positioning
    # from a few minutes ago still beats nothing.
    stale = _cache_get(key, allow_stale=True)
    if stale is not None:
        return stale
    raise BinanceUnavailable(f"{path} failed: {last_exc}") from last_exc


def _openers():
    """Direct first, then the package's proxy opener if one is configured."""
    yield urllib.request.urlopen
    try:
        from tradingview_mcp.core.services.proxy_manager import (
            build_opener_with_proxy,
            is_proxy_configured,
        )
        if is_proxy_configured():
            yield build_opener_with_proxy(_UA).open
    except Exception:
        return


# ── Symbol resolution ─────────────────────────────────────────────────────────

def binance_period_for(period: str) -> Optional[str]:
    """Translate one of this server's timeframes to Binance's vocabulary.

    Returns None when there is no equivalent, so callers can report the
    supported set rather than silently sampling a different interval.
    """
    if not period:
        return None
    return _PERIOD_MAP.get(period) or _PERIOD_MAP.get(period.lower())


def supported_periods() -> List[str]:
    return sorted(set(_PERIOD_MAP))


def perpetual_symbols() -> List[str]:
    """Every PERPETUAL contract symbol listed on Binance futures."""
    key = ("binance", "perp-universe")
    cached = _cache_get(key, allow_stale=True)
    if cached is not None:
        return cached
    info = _fetch("/fapi/v1/exchangeInfo")
    syms = sorted(
        s["symbol"] for s in info.get("symbols", [])
        if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
    )
    _cache_put(key, syms)
    return syms


def perp_symbol_for(symbol: str) -> str:
    """Map a pair symbol to its Binance perpetual symbol.

    USDT pairs map straight through (``BTCUSDT`` spot and perp share a name).
    An exchange prefix is stripped. Nothing is invented: whether the result
    actually lists as a perp is checked by the caller against
    ``perpetual_symbols()``, because most pairs have no perpetual market.
    """
    raw = (symbol or "").strip().upper()
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    return raw.replace("-", "").replace("/", "")


# ── Raw reads ─────────────────────────────────────────────────────────────────

def get_premium_index(symbol: str) -> dict:
    return _fetch("/fapi/v1/premiumIndex", {"symbol": symbol})


def get_open_interest(symbol: str) -> dict:
    return _fetch("/fapi/v1/openInterest", {"symbol": symbol})


def get_open_interest_history(symbol: str, period: str = "4h", limit: int = 30) -> list:
    return _fetch(
        "/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": limit},
    )


def get_funding_history(symbol: str, limit: int = 10) -> list:
    return _fetch("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})


def get_global_long_short(symbol: str, period: str = "4h", limit: int = 1) -> list:
    return _fetch(
        "/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": period, "limit": limit},
    )


def get_top_trader_positions(symbol: str, period: str = "4h", limit: int = 1) -> list:
    return _fetch(
        "/futures/data/topLongShortPositionRatio",
        {"symbol": symbol, "period": period, "limit": limit},
    )


def get_taker_flow(symbol: str, period: str = "4h", limit: int = 1) -> list:
    return _fetch(
        "/futures/data/takerlongshortRatio",
        {"symbol": symbol, "period": period, "limit": limit},
    )


def get_ticker_24h(symbol: str) -> dict:
    return _fetch("/fapi/v1/ticker/24hr", {"symbol": symbol})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def infer_funding_interval_hours(history: List[dict]) -> float:
    """Hours between funding payments, read from the history's timestamps.

    Binance pays most perps every 8 hours and some every 4, and the interval
    is not on the premiumIndex response. Annualising with a hardcoded 8 would
    understate a 4-hour symbol's cost by half, so it is measured instead.
    """
    stamps = sorted(
        t for t in (_f(h.get("fundingTime")) for h in history or []) if t
    )
    if len(stamps) < 2:
        return _DEFAULT_FUNDING_INTERVAL_H
    gaps = [(b - a) / 3_600_000.0 for a, b in zip(stamps, stamps[1:]) if b > a]
    if not gaps:
        return _DEFAULT_FUNDING_INTERVAL_H
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    # Snap to the intervals Binance actually uses; a clock skew or a missed
    # payment shouldn't produce a 7.4-hour "interval".
    return min((1.0, 2.0, 4.0, 8.0), key=lambda cand: abs(cand - median_gap))
