"""
Crypto Analysis Service — swing-trading tools for crypto pairs.

Mirrors the EGX service's trade-plan and Fibonacci tools for crypto exchanges,
and adds candle-based tools the server had no home for: true Supertrend and
Donchian breakout scans and RSI divergence detection.

Two data paths, chosen per tool:

  * **TradingView scanner** (``resilient_scan_columns`` /
    ``resilient_get_multiple_analysis``) — one request, every timeframe the
    server supports, no raw candles. Used by the Fibonacci, trade-plan and
    watchlist tools, exactly as the EGX equivalents do.
  * **Yahoo OHLC candles** (``backtest_service._fetch_ohlcv``) — real bar
    history, needed for the recursive indicators (Supertrend) and for
    pivot-based analysis (Donchian breakout, RSI divergence) that no scanner
    column exposes. Yahoo serves crypto as ``BTC-USD``, so pair symbols are
    converted; it offers 1h and 1d bars only, so 4h bars are resampled from
    1h (clean for crypto, which trades 24/7 with no session gaps).

All public functions return plain dicts / lists and are independently testable
without the MCP layer.
"""
from __future__ import annotations
from tradingview_mcp.core.utils.rounding import keep_digits as round  # noqa: A001 - a small price must keep its digits

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import binance_service as bn
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.indicators import (
    AVERAGE_VOLUME,
    RELATIVE_VOLUME,
    analyze_fibonacci_position,
    compute_fibonacci_levels,
    compute_trade_setup,
    detect_trend_for_fibonacci,
)
from tradingview_mcp.core.services.indicators_calc import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from tradingview_mcp.core.services.screener_provider import (
    fetch_atr_for_ticker,
    resilient_get_multiple_analysis,
    resilient_scan_columns,
)
from tradingview_mcp.core.utils.validators import (
    EXCHANGE_SCREENER,
    normalize_tradingview_symbol,
)

# ── Lookback windows, shared with the EGX Fibonacci tool ──────────────────────
# The scanner exposes period extremes as columns, so a swing high/low costs one
# request instead of a candle download.
_LOOKBACK_COLUMNS = {
    "1M": ("High.1M", "Low.1M"),
    "3M": ("High.3M", "Low.3M"),
    "6M": ("High.6M", "Low.6M"),
    "52W": ("price_52_week_high", "price_52_week_low"),
    "ALL": ("High.All", "Low.All"),
}

# Yahoo interval + range per timeframe, and the resample factor applied after.
# Yahoo caps 1h history at ~730 days; these ranges stay well inside it.
_YF_PLAN = {
    "1h": ("1h", "3mo", 1),
    "4h": ("1h", "6mo", 4),
    "1D": ("1d", "2y", 1),
}

_CANDLE_TIMEFRAMES = tuple(_YF_PLAN.keys())  # what the Yahoo fallback can serve


def _supported_timeframes() -> List[str]:
    """Timeframes some provider can serve — Binance's native set plus Yahoo's."""
    return sorted(set(bn.kline_timeframes()) | set(_YF_PLAN))


def _timeframe_has_provider(timeframe: str) -> bool:
    """Whether any provider could serve this timeframe.

    Whether a given SYMBOL can be served is a separate question answered at
    fetch time: a pair listed only on KuCoin falls back to Yahoo, which
    serves 1h/4h/1D alone.
    """
    return bool(bn.kline_interval_for(timeframe)) or timeframe in _YF_PLAN

# Quote currencies a crypto pair may be denominated in, longest first so that
# USDT is stripped before USD.
_QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "EUR", "BTC", "ETH")

_MAX_SCAN_SYMBOLS = 40   # candle fetches per scan — one upstream request each
_MAX_WATCHLIST = 30      # symbols per watchlist request (one scanner call)
_FETCH_WORKERS = 8
_KLINE_LIMIT = 500       # bars per Binance request (its ceiling is 1500)


def binance_symbol_for(symbol: str) -> str:
    """Binance's spelling of a symbol. A coin's bare name means its USDT pair."""
    from tradingview_mcp.core.utils.validators import crypto_pair

    return crypto_pair(symbol, "BINANCE")


# ── Symbol helpers ────────────────────────────────────────────────────────────

def yahoo_crypto_symbol(symbol: str) -> str:
    """Convert an exchange pair symbol to its Yahoo Finance form.

    ``BTCUSDT`` -> ``BTC-USD``, ``ETHUSDC`` -> ``ETH-USD``, ``SOLBTC`` ->
    ``SOL-BTC``. Symbols that already carry a dash, a dot or an index prefix
    pass through untouched, so ``BTC-USD`` and ``^GSPC`` are left alone.

    Yahoo quotes every dollar stablecoin pair against plain ``USD``; stripping
    the stablecoin and re-quoting is what makes ``BTCUSDT`` resolvable there.
    """
    raw = (symbol or "").strip().upper()
    if not raw or "-" in raw or "." in raw or raw.startswith("^") or "=" in raw:
        return raw
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    # The ticker BTC on Yahoo is an ETF, and this is the fallback for
    # crypto candles: a coin named bare has to become its pair first, or
    # the fallback quietly charts a fund.
    from tradingview_mcp.core.utils.validators import crypto_pair

    raw = crypto_pair(raw, "BINANCE", known_only=True)
    for quote in _QUOTE_ASSETS:
        if raw.endswith(quote) and len(raw) > len(quote):
            base = raw[: -len(quote)]
            # Yahoo has no stablecoin quotes — they all map to USD.
            quote_out = "USD" if quote not in ("BTC", "ETH", "EUR") else quote
            return f"{base}-{quote_out}"
    return raw


def _screener_for(exchange: str) -> str:
    return EXCHANGE_SCREENER.get((exchange or "").strip().lower(), "crypto")


def _full_symbol(symbol: str, exchange: str) -> str:
    return normalize_tradingview_symbol(symbol, exchange)


# ── Candle access ─────────────────────────────────────────────────────────────

def _resample(candles: List[dict], factor: int) -> List[dict]:
    """Aggregate 1h candles into ``factor``-hour bars, aligned to UTC midnight.

    Alignment matters: bucketing by position in the returned array would make
    every bar boundary depend on when the request ran, so the same symbol would
    produce different 4h candles minute to minute. Grouping on
    ``hour // factor`` pins the boundaries to 00:00/04:00/08:00 UTC the way an
    exchange chart draws them.
    """
    if factor <= 1:
        return candles

    def _parts(stamp: str) -> tuple:
        day, _, clock = stamp.partition(" ")
        try:
            hour = int(clock.split(":")[0]) if clock else 0
        except ValueError:
            hour = 0
        return day, hour

    # Yahoo appends a LIVE partial candle whose stamp is the request time, not
    # the top of the hour (e.g. a "19:26" bar sitting next to the finished
    # "19:00" one). Both share hour 19, so bucketing naively put five hourly
    # candles in a 4h bar and summed that hour's volume twice. Collapse each
    # hour to its latest observation first: the live bar wins, so the bucket
    # carries the most current price without counting the hour twice.
    by_hour: Dict[tuple, dict] = {}
    hour_order: List[tuple] = []
    for c in candles:
        key = _parts(c["date"])
        if key not in by_hour:
            hour_order.append(key)
        else:
            prev = by_hour[key]
            c = {
                **c,
                "open": prev["open"],
                "high": max(prev["high"], c["high"]),
                "low": min(prev["low"], c["low"]),
                "volume": max(prev.get("volume") or 0, c.get("volume") or 0),
            }
        by_hour[key] = c

    buckets: Dict[tuple, List[dict]] = {}
    order: List[tuple] = []
    for hour_key in hour_order:
        c = by_hour[hour_key]
        day, hour = hour_key
        key = (day, hour // factor)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(c)

    out: List[dict] = []
    for key in order:
        group = buckets[key]
        out.append({
            "date": f"{key[0]} {key[1] * factor:02d}:00",
            "open": group[0]["open"],
            "high": max(c["high"] for c in group),
            "low": min(c["low"] for c in group),
            "close": group[-1]["close"],
            "volume": sum(c.get("volume") or 0 for c in group),
            "bars": len(group),
            "complete": len(group) == factor,
        })
    return out


def fetch_crypto_candles(symbol: str, timeframe: str = "4h") -> tuple:
    """Return ``(candles, source_label)`` for a crypto pair at ``timeframe``.

    Tiered by accuracy, then by coverage:

      1. **Binance klines** (futures book, then spot). Native bars at every
         timeframe, and measured against TradingView's 4h closes they sit
         within 0.01-0.05% where the Yahoo path was 0.10-0.11% out.
      2. **Yahoo**, resampled. Only 326 of KuCoin's 1,006 symbols list on
         Binance at all, so the long tail still needs this — at 1h/4h/1D only,
         since those are the intervals Yahoo serves.

    The label is returned rather than assumed: the tools used to report
    "Yahoo Finance" unconditionally, which would now be wrong for every
    liquid pair.

    Raises ``ValueError`` when no provider can serve the timeframe, and
    ``RuntimeError`` when every provider fails.
    """
    bn_symbol = binance_symbol_for(symbol)
    if bn.kline_interval_for(timeframe):
        try:
            candles, label = bn.fetch_candles(bn_symbol, timeframe, limit=_KLINE_LIMIT)
            if candles:
                return candles, label
        except Exception:
            pass  # not on Binance, or Binance is unreachable — fall through

    if timeframe not in _YF_PLAN:
        raise ValueError(
            f"{bn_symbol} is not on Binance, and the fallback provider serves "
            f"only {', '.join(_CANDLE_TIMEFRAMES)} — {timeframe!r} is unavailable "
            "for this symbol"
        )
    interval, period, factor = _YF_PLAN[timeframe]
    # The package's only other OHLC provider. Private by name, but duplicating
    # its Yahoo handling (proxy fallback, split/dividend adjustment) would be
    # worse than importing it.
    from tradingview_mcp.core.services.backtest_service import _fetch_ohlcv

    candles = _resample(_fetch_ohlcv(yahoo_crypto_symbol(symbol), period, interval), factor)
    suffix = " (1h resampled to 4h)" if timeframe == "4h" else ""
    return candles, f"Yahoo Finance{suffix}"


def _ohlc(candles: Sequence[dict]) -> tuple:
    return (
        [c["high"] for c in candles],
        [c["low"] for c in candles],
        [c["close"] for c in candles],
    )


def _last_not_none(values: Sequence[Optional[float]]) -> Optional[float]:
    for v in reversed(values):
        if v is not None:
            return v
    return None


# ── 1. Fibonacci retracement ──────────────────────────────────────────────────

def analyze_crypto_fibonacci(
    symbol: str,
    exchange: str = "KUCOIN",
    lookback: str = "52W",
    timeframe: str = "1D",
) -> dict:
    """Fibonacci retracement analysis for a crypto pair.

    The crypto counterpart of ``analyze_egx_fibonacci``: swing high/low come
    from scanner period-extreme columns, and the level maths is the shared
    implementation in ``indicators`` — so the two tools cannot drift apart.

    Args:
        symbol:    Pair symbol, e.g. ``BTCUSDT``.
        exchange:  Crypto exchange, e.g. ``KUCOIN``.
        lookback:  Swing window — ``1M``, ``3M``, ``6M``, ``52W`` or ``ALL``.
        timeframe: Interval for the indicator context.
    """
    if lookback not in _LOOKBACK_COLUMNS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Invalid lookback: {lookback}",
            valid_lookbacks=sorted(_LOOKBACK_COLUMNS),
        )

    full = _full_symbol(symbol, exchange)
    screener = _screener_for(exchange)
    high_col, low_col = _LOOKBACK_COLUMNS[lookback]

    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    swing_source: Optional[str] = None
    try:
        rows = resilient_scan_columns(screener, timeframe, [full], ["close", high_col, low_col])
        row = rows.get(full) or {}
        h, lo = row.get(high_col), row.get(low_col)
        if h is not None and lo is not None and float(h) > float(lo):
            swing_high, swing_low = float(h), float(lo)
            swing_source = f"screener ({lookback} period high/low)"
    except Exception:
        pass  # fall through to the pivot fallback below

    try:
        analysis = resilient_get_multiple_analysis(screener, timeframe, [full])
    except Exception as exc:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Analysis failed: {exc}", symbol=full)

    if full not in analysis or analysis[full] is None:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, f"No data found for {full}", symbol=full)

    ind = dict(analysis[full].indicators)
    if ind.get("ATR") is None:
        atr_value = fetch_atr_for_ticker(full, screener, timeframe)
        if atr_value is not None:
            ind["ATR"] = atr_value

    close = ind.get("close")
    if not close:
        return make_error(ErrorCode.NO_DATA, f"No price data for {full}", symbol=full)

    if swing_high is None or swing_low is None:
        h_candidate = ind.get("Pivot.M.Fibonacci.R3") or ind.get("Pivot.M.Classic.R3")
        l_candidate = ind.get("Pivot.M.Fibonacci.S3") or ind.get("Pivot.M.Classic.S3")
        if h_candidate and l_candidate and h_candidate > l_candidate:
            swing_high, swing_low = float(h_candidate), float(l_candidate)
            swing_source = "pivot points (R3/S3 fallback)"
        else:
            return make_error(
                ErrorCode.NO_DATA,
                "Could not determine swing high/low for Fibonacci calculation",
                symbol=full,
                hint="Period high/low data not available for this pair",
            )

    if swing_low <= 0:
        return make_error(
            ErrorCode.NO_DATA,
            "Invalid swing low (<= 0) — cannot compute Fibonacci range",
            swing_high=swing_high,
            swing_low=swing_low,
        )

    swing_range_pct = ((swing_high - swing_low) / swing_low) * 100
    if swing_range_pct < 2:
        return make_error(
            ErrorCode.NO_DATA,
            f"Swing range too narrow ({swing_range_pct:.1f}%) for meaningful Fibonacci levels",
            swing_high=swing_high,
            swing_low=swing_low,
        )

    ema50, ema200 = ind.get("EMA50"), ind.get("EMA200")
    trend, trend_reasoning = detect_trend_for_fibonacci(close, swing_high, swing_low, ema50, ema200)
    fib_levels = compute_fibonacci_levels(swing_high, swing_low, trend)
    position = analyze_fibonacci_position(close, fib_levels)

    rel_vol = ind.get(RELATIVE_VOLUME)
    open_px = ind.get("open")
    interp = [f"Price is at {position['retracement_depth_pct']}% retracement of the {trend}."]
    if position.get("key_zone"):
        interp.append(f"Currently in {position['key_zone']}.")
    if position.get("fib_supports"):
        s = position["fib_supports"][0]
        interp.append(f"Key Fib support at {s['price']} ({s['ratio']}).")
    if position.get("fib_resistances"):
        r = position["fib_resistances"][0]
        interp.append(f"Key Fib resistance at {r['price']} ({r['ratio']}).")

    return {
        "symbol": full,
        "timeframe": timeframe,
        "lookback_period": lookback,
        "price": close,
        "change_pct": round(((close - open_px) / open_px) * 100, 2) if open_px else None,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_range_pct": round(swing_range_pct, 1),
        "swing_source": swing_source,
        "trend": trend,
        "trend_reasoning": trend_reasoning,
        "retracement_levels": fib_levels["retracement_levels"],
        "extension_levels": fib_levels["extension_levels"],
        "price_position": position,
        "context": {
            "rsi": round(ind["RSI"], 1) if ind.get("RSI") else None,
            "ema50": ema50,
            "ema200": ema200,
            "atr": ind.get("ATR"),
            "volume_ratio": round(rel_vol, 2) if rel_vol is not None else None,
        },
        "interpretation": " ".join(interp),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── 2. Trade plan ─────────────────────────────────────────────────────────────

def generate_crypto_trade_plan(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    account_size: Optional[float] = None,
    risk_pct: float = 1.0,
) -> dict:
    """Entry / stop / targets for a crypto pair, with optional position sizing.

    Delegates the setup itself to the shared ``compute_trade_setup``, so the
    entry, stop, targets and R:R are the same coherent per-scenario plan the
    EGX trade plan returns. When ``account_size`` is given, position size comes
    from the distance to the primary scenario's stop and ``risk_pct`` of the
    account — not from a fixed fraction, so a wide stop buys less.

    Args:
        symbol:       Pair symbol, e.g. ``BTCUSDT``.
        exchange:     Crypto exchange, e.g. ``KUCOIN``.
        timeframe:    Interval to plan on (``4h`` suits swing trades).
        account_size: Account value in quote currency. Omit to skip sizing.
        risk_pct:     Percent of the account risked on the trade (0 < x <= 100).
    """
    full = _full_symbol(symbol, exchange)
    screener = _screener_for(exchange)

    if not 0 < risk_pct <= 100:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"risk_pct must be between 0 and 100, got {risk_pct}",
        )
    if account_size is not None and account_size <= 0:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"account_size must be positive, got {account_size}",
        )

    try:
        analysis = resilient_get_multiple_analysis(screener, timeframe, [full])
    except Exception as exc:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Analysis failed: {exc}", symbol=full)

    if full not in analysis or analysis[full] is None:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, f"No data found for {full}", symbol=full)

    ind = dict(analysis[full].indicators)
    if ind.get("ATR") is None:
        atr_value = fetch_atr_for_ticker(full, screener, timeframe)
        if atr_value is not None:
            ind["ATR"] = atr_value

    close = ind.get("close")
    if not close:
        return make_error(ErrorCode.NO_DATA, f"No price data for {full}", symbol=full)

    setup = compute_trade_setup(ind)
    if not setup:
        return make_error(
            ErrorCode.NO_DATA,
            "Could not build a trade setup — ATR unavailable for this pair/timeframe",
            symbol=full,
            timeframe=timeframe,
            hint="Try a higher timeframe (4h or 1D), where ATR is more reliably published",
        )

    rsi = ind.get("RSI")
    ema50, ema200 = ind.get("EMA50"), ind.get("EMA200")
    rel_vol = ind.get(RELATIVE_VOLUME)

    sizing: Optional[dict] = None
    # compute_trade_setup has no top-level "entry": the primary scenario's entry
    # is published as risk_reward.measured_from_entry, which is also what its
    # stop and targets are measured against. Reading anything else here would
    # size a position against a stop that belongs to a different entry.
    entry = (setup.get("risk_reward") or {}).get("measured_from_entry")
    if entry is None:
        primary = (setup.get("scenarios") or {}).get(setup.get("primary_scenario") or "")
        entry = (primary or {}).get("entry")
    stop = setup.get("stop_loss")
    if account_size and entry and stop and entry != stop:
        risk_amount = account_size * (risk_pct / 100.0)
        per_unit_risk = abs(entry - stop)
        units = risk_amount / per_unit_risk
        notional = units * entry
        sizing = {
            "account_size": account_size,
            "risk_pct": risk_pct,
            "risk_amount": round(risk_amount, 2),
            "stop_distance": round(per_unit_risk, 8),
            "stop_distance_pct": round((per_unit_risk / entry) * 100, 2),
            "units": round(units, 8),
            "notional": round(notional, 2),
            "leverage_vs_account": round(notional / account_size, 2),
            "note": (
                "Size risks exactly risk_pct of the account if the stop fills at its "
                "level. Slippage and funding are not modelled; a gap through the stop "
                "loses more."
            ),
        }

    trend_bits = []
    if ema50 and ema200:
        trend_bits.append("EMA50 above EMA200" if ema50 > ema200 else "EMA50 below EMA200")
    if ema50 and close:
        trend_bits.append("price above EMA50" if close > ema50 else "price below EMA50")
    if rsi is not None:
        trend_bits.append(f"RSI {rsi:.0f}")

    return {
        "symbol": full,
        "timeframe": timeframe,
        "price": close,
        "trade_setup": setup,
        "position_sizing": sizing,
        "context": {
            "rsi": round(rsi, 1) if rsi is not None else None,
            "ema20": ind.get("EMA20"),
            "ema50": ema50,
            "ema200": ema200,
            "atr": ind.get("ATR"),
            "volume_ratio": round(rel_vol, 2) if rel_vol is not None else None,
            "recommendation": ind.get("Recommend.All"),
        },
        "trend_summary": ", ".join(trend_bits) if trend_bits else None,
        # Discovery where it is actually useful. Someone holding an entry, a
        # stop and a target is exactly the person who should know the wallet
        # exists — a line in the README reaches nobody at this moment. Kept to
        # one sentence, because a pointer repeated on every call is a tax.
        "next_step": (
            "To test this setup without risking anything, paper_trade_open "
            "takes these levels and paper_wallet_status tracks it; the wallet "
            "is simulated and places no real order."
        ),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── 3. Watchlist ──────────────────────────────────────────────────────────────

_WATCHLIST_COLUMNS = [
    "close", "change", "RSI", "EMA50", "ATR",
    RELATIVE_VOLUME, AVERAGE_VOLUME, "volume", "Recommend.All",
]


def build_crypto_watchlist(
    symbols: List[str],
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
) -> dict:
    """One scored row per symbol, in a single scanner request.

    The crypto/stock counterpart of ``futures_watchlist``. Calling
    ``coin_analysis`` per symbol costs one request each; this asks the scanner
    for every symbol at once, so a 20-pair watchlist is one round trip.

    Args:
        symbols:   Pair symbols, e.g. ``["BTCUSDT", "ETHUSDT"]`` (max 30).
        exchange:  Exchange the symbols trade on.
        timeframe: Interval for the indicator values.
    """
    if not symbols:
        return make_error(ErrorCode.INVALID_PARAMETER, "symbols must not be empty")

    requested = [s for s in (sym.strip() for sym in symbols) if s][:_MAX_WATCHLIST]
    if not requested:
        return make_error(ErrorCode.INVALID_PARAMETER, "symbols must not be empty")

    full_map = {_full_symbol(s, exchange): s for s in requested}
    screener = _screener_for(exchange)

    try:
        rows = resilient_scan_columns(screener, timeframe, list(full_map), _WATCHLIST_COLUMNS)
    except Exception as exc:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Scan failed: {exc}", exchange=exchange)

    out: List[dict] = []
    missing: List[str] = []
    for full, original in full_map.items():
        row = rows.get(full)
        if not row:
            missing.append(original)
            continue
        close = row.get("close")
        ema50 = row.get("EMA50")
        rel_vol = row.get(RELATIVE_VOLUME)
        rsi = row.get("RSI")
        if close and ema50:
            trend = "above EMA50" if close > ema50 else "below EMA50"
        else:
            trend = None
        out.append({
            "symbol": original,
            "ticker": full,
            "price": close,
            "change_pct": round(row["change"], 2) if row.get("change") is not None else None,
            "rsi": round(rsi, 1) if rsi is not None else None,
            "rsi_zone": (
                None if rsi is None
                else "overbought" if rsi >= 70
                else "oversold" if rsi <= 30
                else "neutral"
            ),
            "ema50": ema50,
            "trend": trend,
            "atr": row.get("ATR"),
            "volume_ratio": round(rel_vol, 2) if rel_vol is not None else None,
            "volume": row.get("volume"),
            "avg_volume_10d": row.get(AVERAGE_VOLUME),
            "rating": row.get("Recommend.All"),
        })

    out.sort(key=lambda r: (r["change_pct"] is None, -(r["change_pct"] or 0)))
    result: Dict[str, Any] = {
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "requested": len(requested),
        "returned": len(out),
        "rows": out,
    }
    if missing:
        result["not_found"] = missing
        result["note"] = (
            "Symbols under not_found are unknown to this exchange's scanner — "
            "check the pair spelling and that it lists on this venue."
        )
    if len(symbols) > _MAX_WATCHLIST:
        result["truncated_to"] = _MAX_WATCHLIST
    return result


# ── 4. RSI divergence ─────────────────────────────────────────────────────────

def _pivot_indices(values: Sequence[Optional[float]], window: int, kind: str) -> List[int]:
    """Indices of local extremes with ``window`` bars clear on each side.

    A pivot needs both sides, so the last ``window`` bars can never qualify —
    that is deliberate: a high with nothing to its right is not yet a pivot,
    and treating it as one invents divergences on the forming bar.
    """
    out: List[int] = []
    n = len(values)
    for i in range(window, n - window):
        v = values[i]
        if v is None:
            continue
        neighbours = [
            values[j] for j in range(i - window, i + window + 1)
            if j != i and values[j] is not None
        ]
        if not neighbours:
            continue
        if kind == "high" and v > max(neighbours):
            out.append(i)
        elif kind == "low" and v < min(neighbours):
            out.append(i)
    return out


def detect_rsi_divergence(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    lookback: int = 60,
    pivot_window: int = 3,
    rsi_period: int = 14,
) -> dict:
    """Detect bullish / bearish RSI divergence on a crypto pair.

    Bearish divergence: price prints a higher pivot high while RSI prints a
    lower one. Bullish: price prints a lower pivot low while RSI does not.
    Both compare the two most recent confirmed pivots inside ``lookback`` bars.

    Args:
        symbol:       Pair symbol, e.g. ``BTCUSDT``.
        exchange:     Crypto exchange.
        timeframe:    ``1h``, ``4h`` or ``1D`` (candle-based tool).
        lookback:     Bars to search, 20-300.
        pivot_window: Bars required clear either side of a pivot (1-10).
        rsi_period:   RSI period (2-50).
    """
    if not 20 <= lookback <= 300:
        return make_error(ErrorCode.INVALID_PARAMETER, f"lookback must be 20-300, got {lookback}")
    if not 1 <= pivot_window <= 10:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"pivot_window must be 1-10, got {pivot_window}"
        )
    if not 2 <= rsi_period <= 50:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"rsi_period must be 2-50, got {rsi_period}"
        )

    try:
        candles, candle_source = fetch_crypto_candles(symbol, timeframe)
    except ValueError as exc:
        return make_error(
            ErrorCode.INVALID_TIMEFRAME, str(exc), supported=_supported_timeframes()
        )
    except Exception as exc:
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"Candle fetch failed: {exc}",
            symbol=symbol,
            binance_symbol=binance_symbol_for(symbol),
        )

    if len(candles) < rsi_period + 2 * pivot_window + 5:
        return make_error(
            ErrorCode.NO_DATA,
            f"Only {len(candles)} candles available — too few for divergence detection",
            symbol=symbol,
            candle_source=candle_source,
        )

    candles = candles[-lookback:]
    highs, lows, closes = _ohlc(candles)
    rsi = calc_rsi(closes, rsi_period)

    findings: List[dict] = []
    for kind, series in (("bearish", highs), ("bullish", lows)):
        pivots = _pivot_indices(series, pivot_window, "high" if kind == "bearish" else "low")
        pivots = [i for i in pivots if rsi[i] is not None]
        if len(pivots) < 2:
            continue
        prev_i, last_i = pivots[-2], pivots[-1]
        p_prev, p_last = series[prev_i], series[last_i]
        r_prev, r_last = rsi[prev_i], rsi[last_i]
        if kind == "bearish":
            diverged = p_last > p_prev and r_last < r_prev
        else:
            diverged = p_last < p_prev and r_last > r_prev
        if not diverged:
            continue
        findings.append({
            "type": kind,
            "price_pivots": [
                {"date": candles[prev_i]["date"], "price": p_prev, "rsi": round(r_prev, 1)},
                {"date": candles[last_i]["date"], "price": p_last, "rsi": round(r_last, 1)},
            ],
            "price_change_pct": round(((p_last - p_prev) / p_prev) * 100, 2),
            "rsi_change": round(r_last - r_prev, 1),
            "strength": round(abs(r_last - r_prev), 1),
            "bars_since": len(candles) - 1 - last_i,
        })

    findings.sort(key=lambda f: -f["strength"])
    last_rsi = _last_not_none(rsi)
    return {
        "symbol": binance_symbol_for(symbol),
        "timeframe": timeframe,
        "candles_analyzed": len(candles),
        "candle_source": candle_source,
        "rsi_period": rsi_period,
        "pivot_window": pivot_window,
        "current_price": closes[-1],
        "current_rsi": round(last_rsi, 1) if last_rsi is not None else None,
        "divergences": findings,
        "divergence_found": bool(findings),
        "interpretation": (
            f"{findings[0]['type'].capitalize()} divergence {findings[0]['bars_since']} bars ago: "
            f"price {findings[0]['price_change_pct']:+}% against an RSI move of "
            f"{findings[0]['rsi_change']:+}."
            if findings else
            "No divergence between the two most recent confirmed pivots."
        ),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── 5 & 6. Candle-based scans ─────────────────────────────────────────────────

def _scan_universe(exchange: str, timeframe: str, limit: int) -> tuple:
    """Pick the most liquid symbols on ``exchange`` in one scanner request.

    Candle-based indicators need one Yahoo request per symbol, so scanning a
    whole exchange bar by bar is off the table. This narrows the venue to its
    highest-turnover names first — one scanner call — and only those get
    candles. Returns ``(symbols, note)``.
    """
    universe = load_symbols(exchange)
    if not universe:
        return [], f"No symbol list shipped for {exchange.upper()}"
    screener = _screener_for(exchange)
    full = [_full_symbol(s, exchange) for s in universe]
    try:
        rows = resilient_scan_columns(screener, timeframe, full, ["close", "volume"])
    except Exception:
        # Scanner unavailable — fall back to the head of the packaged list so
        # the scan still returns something, and say so.
        return universe[:limit], (
            "Liquidity ranking unavailable (scanner request failed); "
            "scanned the first symbols in the packaged list instead."
        )
    ranked = sorted(
        (
            (r.get("close") or 0) * (r.get("volume") or 0),
            ticker.split(":")[-1],
        )
        for ticker, r in rows.items()
    )
    picked = [sym for _, sym in reversed(ranked)][:limit]
    return picked, f"Top {len(picked)} of {len(universe)} {exchange.upper()} symbols by turnover"


def _gather_candles(symbols: Sequence[str], timeframe: str) -> tuple:
    """Fetch candles for several symbols concurrently.

    Returns ``(ok, failed, sources)`` where ``sources`` counts how many
    symbols each provider served — a scan can legitimately mix Binance and
    Yahoo bars, and the result says so rather than implying one source.
    """
    ok: Dict[str, List[dict]] = {}
    failed: List[str] = []
    sources: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_crypto_candles, s, timeframe): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                candles, source = fut.result()
            except Exception:
                failed.append(sym)
                continue
            if candles:
                ok[sym] = candles
                sources[source] = sources.get(source, 0) + 1
            else:
                failed.append(sym)
    return ok, failed, sources


def _resolve_scan_symbols(
    symbols: Optional[List[str]], exchange: str, timeframe: str, max_symbols: int
) -> tuple:
    if symbols:
        picked = [s for s in (sym.strip() for sym in symbols) if s][:max_symbols]
        return picked, "Explicit symbol list"
    return _scan_universe(exchange, timeframe, max_symbols)


def scan_supertrend(
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    symbols: Optional[List[str]] = None,
    max_symbols: int = 15,
    flip_within: int = 3,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> dict:
    """True Supertrend state, and recent flips, across a set of crypto pairs.

    Supertrend is recursive — each bar's band depends on the previous bar's —
    so no scanner column can supply it and it is computed from real candles
    here. Pass ``symbols`` to scan exactly those, or omit it to scan the
    exchange's most liquid names (see ``universe_note`` in the result).

    Args:
        exchange:    Crypto exchange.
        timeframe:   ``1h``, ``4h`` or ``1D``.
        symbols:     Explicit pairs to scan. Omit for the liquidity-ranked set.
        max_symbols: Pairs to fetch candles for, 1-40.
        flip_within: Flag flips that happened within this many bars, 1-20.
        atr_period:  Supertrend ATR period (2-50).
        multiplier:  Supertrend ATR multiplier (0.5-10).
    """
    if not 1 <= max_symbols <= _MAX_SCAN_SYMBOLS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"max_symbols must be 1-{_MAX_SCAN_SYMBOLS}, got {max_symbols}",
        )
    if not 1 <= flip_within <= 20:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"flip_within must be 1-20, got {flip_within}"
        )
    if not 2 <= atr_period <= 50:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"atr_period must be 2-50, got {atr_period}"
        )
    if not 0.5 <= multiplier <= 10:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"multiplier must be 0.5-10, got {multiplier}"
        )
    if not _timeframe_has_provider(timeframe):
        return make_error(
            ErrorCode.INVALID_TIMEFRAME,
            f"timeframe {timeframe!r} has no candle source",
            supported=_supported_timeframes(),
        )

    picked, universe_note = _resolve_scan_symbols(symbols, exchange, timeframe, max_symbols)
    if not picked:
        return make_error(
            ErrorCode.NO_DATA, f"No symbols to scan for {exchange}", exchange=exchange
        )

    candle_map, failed, sources = _gather_candles(picked, timeframe)
    rows: List[dict] = []
    for sym, candles in candle_map.items():
        if len(candles) < atr_period + 5:
            failed.append(sym)
            continue
        highs, lows, closes = _ohlc(candles)
        st = calc_supertrend(highs, lows, closes, atr_period, multiplier)
        direction = st["direction"]
        known = [i for i, d in enumerate(direction) if d is not None]
        if not known:
            failed.append(sym)
            continue
        last_i = known[-1]
        current = direction[last_i]
        flip_i = None
        for i in reversed(known[:-1]):
            if direction[i] != current:
                flip_i = i + 1
                break
        bars_since_flip = None if flip_i is None else last_i - flip_i
        line = st["lower"][last_i] if current == 1 else st["upper"][last_i]
        atr_now = _last_not_none(calc_atr(highs, lows, closes, atr_period))
        rows.append({
            "symbol": sym,
            "price": closes[last_i],
            "direction": "bullish" if current == 1 else "bearish",
            "supertrend_line": None if line is None else round(line, 8),
            "distance_pct": (
                None if not line else round(((closes[last_i] - line) / line) * 100, 2)
            ),
            "atr_distance": (
                None if not (line and atr_now) else round(abs(closes[last_i] - line) / atr_now, 2)
            ),
            "bars_since_flip": bars_since_flip,
            "just_flipped": bars_since_flip is not None and bars_since_flip <= flip_within,
            # Before the first band penetration the direction is only the seed
            # guess, not a signal. Checked against pandas-ta over ~1,000 bars
            # per symbol: the two implementations agree bar-for-bar from the
            # first flip onward, and every disagreement lies in this
            # pre-flip run. Callers should treat an unconfirmed direction as
            # "no reading yet" rather than as bullish or bearish.
            "direction_confirmed": bars_since_flip is not None,
            "last_bar": candles[last_i]["date"],
        })

    rows.sort(key=lambda r: (not r["just_flipped"], r["bars_since_flip"] is None,
                             r["bars_since_flip"] or 0))
    return {
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "indicator": f"Supertrend (ATR {atr_period}, x{multiplier})",
        "candle_sources": sources or None,
        "universe_note": universe_note,
        "scanned": len(candle_map),
        "flips": [r for r in rows if r["just_flipped"]],
        "rows": rows,
        "unavailable": sorted(set(failed)) or None,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


def scan_donchian_breakout(
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    symbols: Optional[List[str]] = None,
    period: int = 20,
    max_symbols: int = 15,
) -> dict:
    """Donchian channel breakouts across a set of crypto pairs.

    The channel is measured on the bars **before** the one being tested:
    ``calc_donchian`` includes the current bar in its window, so comparing a
    close to its own channel high can never register a breakout. Here the
    channel comes from ``candles[:-1]`` and the latest close is tested against
    it — the comparison a chart reader actually makes.

    Args:
        exchange:    Crypto exchange.
        timeframe:   ``1h``, ``4h`` or ``1D``.
        symbols:     Explicit pairs to scan. Omit for the liquidity-ranked set.
        period:      Channel lookback in bars, 5-100.
        max_symbols: Pairs to fetch candles for, 1-40.
    """
    if not 5 <= period <= 100:
        return make_error(ErrorCode.INVALID_PARAMETER, f"period must be 5-100, got {period}")
    if not 1 <= max_symbols <= _MAX_SCAN_SYMBOLS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"max_symbols must be 1-{_MAX_SCAN_SYMBOLS}, got {max_symbols}",
        )
    if not _timeframe_has_provider(timeframe):
        return make_error(
            ErrorCode.INVALID_TIMEFRAME,
            f"timeframe {timeframe!r} has no candle source",
            supported=_supported_timeframes(),
        )

    picked, universe_note = _resolve_scan_symbols(symbols, exchange, timeframe, max_symbols)
    if not picked:
        return make_error(
            ErrorCode.NO_DATA, f"No symbols to scan for {exchange}", exchange=exchange
        )

    candle_map, failed, sources = _gather_candles(picked, timeframe)
    rows: List[dict] = []
    for sym, candles in candle_map.items():
        if len(candles) < period + 2:
            failed.append(sym)
            continue
        prior = candles[:-1]
        highs, lows, _ = _ohlc(prior)
        channel = calc_donchian(highs, lows, period)
        upper = _last_not_none(channel["upper"])
        lower = _last_not_none(channel["lower"])
        middle = _last_not_none(channel["middle"])
        if upper is None or lower is None or upper <= lower:
            failed.append(sym)
            continue
        close = candles[-1]["close"]
        if close > upper:
            state, breakout = "breakout_up", True
        elif close < lower:
            state, breakout = "breakout_down", True
        else:
            state, breakout = "inside_channel", False
        width = upper - lower
        rows.append({
            "symbol": sym,
            "price": close,
            "state": state,
            "breakout": breakout,
            "channel_upper": round(upper, 8),
            "channel_lower": round(lower, 8),
            "channel_middle": None if middle is None else round(middle, 8),
            "channel_width_pct": round((width / lower) * 100, 2),
            "position_in_channel_pct": round(((close - lower) / width) * 100, 1),
            "distance_to_upper_pct": round(((upper - close) / close) * 100, 2),
            "distance_to_lower_pct": round(((close - lower) / close) * 100, 2),
            "last_bar": candles[-1]["date"],
        })

    rows.sort(key=lambda r: (not r["breakout"], -r["position_in_channel_pct"]))
    return {
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "indicator": f"Donchian Channel ({period} bars, measured on prior bars)",
        "candle_sources": sources or None,
        "universe_note": universe_note,
        "scanned": len(candle_map),
        "breakouts": [r for r in rows if r["breakout"]],
        "rows": rows,
        "unavailable": sorted(set(failed)) or None,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
