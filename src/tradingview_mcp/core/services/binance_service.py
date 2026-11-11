"""
Binance Service — positioning feeds and OHLC candles.

Two jobs, both over Binance's public API with no key:

  * **Positioning** (futures only): funding, open interest, the long/short
    splits and taker flow. TradingView's scanner cannot supply these —
    ``funding_rate`` and ``open_interest`` come back null even for perp
    tickers like ``BINANCE:BTCUSDT.P`` that otherwise resolve.
  * **Candles**: native OHLC klines at every interval this server supports,
    from the futures book first and the spot book second.

Measured against TradingView's 4h bars, Binance klines land within 0.01-0.05%
where the old Yahoo path was 0.10-0.11% out, and they need no resampling.
Coverage is the trade-off: of KuCoin's 1,006 symbols only 326 (32%) list on
Binance at all, so callers keep a fallback for the long tail.

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

_BASE = "https://fapi.binance.com"      # USD-M futures
_SPOT_BASE = "https://api.binance.com"  # spot
_UA = "tradingview-mcp/0.9.0 positioning"
_TIMEOUT = 15

# Candle intervals. Every timeframe this server supports maps to a NATIVE
# Binance interval, so candles need no resampling — unlike the Yahoo path,
# which serves 1h and 1d only and had to build 4h bars itself.
_KLINE_INTERVAL = {
    # 1m and 3m matter even though no tool exposes them as a timeframe: the
    # paper-trading settler picks the finest interval that covers a position's
    # age, so a position opened seconds ago asks for 1m bars. Leaving them out
    # made every freshly opened position unsettleable until it aged past 25
    # hours, and broke the spot price fallback, which reads 1m klines.
    "1m": "1m", "3m": "3m",
    "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "2h": "2h",
    "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "1D": "1d", "3d": "3d", "1w": "1w", "1W": "1w", "1M": "1M",
}
_MAX_KLINES = 1500  # Binance's per-request ceiling

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


def _fetch(path: str, params: Optional[Dict[str, Any]] = None,
           base: Optional[str] = None) -> Any:
    """GET a public endpoint, with cache, retries and a stale fallback.

    ``base`` selects the futures host (default) or the spot host; it is part
    of the cache key, so the same path on both books cannot collide.
    """
    base = base or _BASE
    query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    url = f"{base}{path}" + (f"?{query}" if query else "")
    key = ("binance", base, path, query)

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


# ── Candles ───────────────────────────────────────────────────────────────────

def kline_interval_for(timeframe: str) -> Optional[str]:
    """Translate one of this server's timeframes to a Binance kline interval."""
    if not timeframe:
        return None
    return _KLINE_INTERVAL.get(timeframe) or _KLINE_INTERVAL.get(timeframe.lower())


def kline_timeframes() -> List[str]:
    return sorted(set(_KLINE_INTERVAL))


def spot_symbols() -> List[str]:
    """Every spot symbol currently TRADING on Binance."""
    key = ("binance", "spot-universe")
    cached = _cache_get(key, allow_stale=True)
    if cached is not None:
        return cached
    info = _fetch("/api/v3/exchangeInfo", base=_SPOT_BASE)
    syms = sorted(
        s["symbol"] for s in info.get("symbols", [])
        if s.get("status") == "TRADING"
    )
    _cache_put(key, syms)
    return syms


def get_klines(symbol: str, timeframe: str = "4h", limit: int = 500,
               market: str = "futures") -> List[dict]:
    """Native OHLC candles for ``symbol``.

    Returns oldest-first dicts shaped like the rest of the package's candles
    (``date``/``open``/``high``/``low``/``close``/``volume``), so the
    indicator helpers take them unchanged.

    The final candle is the one still forming — deliberate, since the scans
    want the current bar — and carries ``complete: False``.
    """
    interval = kline_interval_for(timeframe)
    if not interval:
        raise ValueError(
            f"timeframe {timeframe!r} has no Binance kline interval; "
            f"supported: {', '.join(kline_timeframes())}"
        )
    base, path = (
        (_SPOT_BASE, "/api/v3/klines") if market == "spot" else (_BASE, "/fapi/v1/klines")
    )
    raw = _fetch(
        path,
        {"symbol": symbol, "interval": interval, "limit": max(2, min(limit, _MAX_KLINES))},
        base=base,
    )
    if not isinstance(raw, list):
        raise BinanceUnavailable(f"unexpected klines payload for {symbol}")

    out: List[dict] = []
    for i, k in enumerate(raw):
        try:
            open_ms = int(k[0])
            out.append({
                "date": time.strftime(
                    "%Y-%m-%d %H:%M", time.gmtime(open_ms / 1000.0)
                ),
                # Kept alongside the formatted date because the paper-trading
                # settler compares bar times against a position's open
                # timestamp, and the "%Y-%m-%d %H:%M" string has already lost
                # the precision that comparison needs.
                "open_ms": open_ms,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "complete": i < len(raw) - 1,
            })
        except (IndexError, TypeError, ValueError):
            continue
    return out


def fetch_candles(symbol: str, timeframe: str = "4h",
                  limit: int = 500) -> Tuple[List[dict], str]:
    """Candles for ``symbol`` from the futures book, else the spot book.

    Futures first because it tracks TradingView's reference most closely
    (measured at 0.01-0.02% versus 0.01-0.06% for spot), and because more of
    the liquid universe lists there. Returns ``(candles, source_label)``;
    raises ``BinanceUnavailable`` when the symbol is on neither book, which is
    the caller's cue to fall back to another provider.
    """
    attempts = []
    try:
        if symbol in set(perpetual_symbols()):
            attempts.append(("futures", "Binance USD-M futures klines"))
    except Exception:
        pass
    try:
        if symbol in set(spot_symbols()):
            attempts.append(("spot", "Binance spot klines"))
    except Exception:
        pass

    last: Optional[BaseException] = None
    for market, label in attempts:
        try:
            candles = get_klines(symbol, timeframe, limit, market)
            if candles:
                return candles, label
        except Exception as exc:  # noqa: BLE001 - try the next book
            last = exc

    raise BinanceUnavailable(
        f"{symbol} is not listed on Binance futures or spot"
        + (f" ({last})" if last else "")
    )


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


# ── Bulk reads (no symbol parameter = the whole board, one request) ───────────

def get_premium_index_all() -> list:
    """Funding and mark price for every perpetual, in ONE request (~945)."""
    payload = _fetch("/fapi/v1/premiumIndex")
    return payload if isinstance(payload, list) else [payload]


def get_ticker_24h_all() -> list:
    """24h change and quote volume for every symbol, in ONE request (~804)."""
    payload = _fetch("/fapi/v1/ticker/24hr")
    return payload if isinstance(payload, list) else [payload]


def funding_intervals() -> Dict[str, float]:
    """Funding interval in hours per symbol.

    ``/fapi/v1/fundingInfo`` lists only symbols with NON-DEFAULT settings, and
    that turns out to be most of the board: 820 of ~945 perps settle every 4
    hours, not 8. Annualising everything at 8h would therefore halve the real
    cost of carry for the majority of symbols. Anything absent from the list
    uses the 8h default.
    """
    key = ("binance", "funding-intervals")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    out: Dict[str, float] = {}
    try:
        for row in _fetch("/fapi/v1/fundingInfo") or []:
            hours = row.get("fundingIntervalHours")
            if row.get("symbol") and hours:
                out[row["symbol"]] = float(hours)
    except Exception:
        return {}  # callers fall back to the 8h default
    _cache_put(key, out)
    return out


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
