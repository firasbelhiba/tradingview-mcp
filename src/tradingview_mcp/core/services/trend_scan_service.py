"""
Trend Scan Service — strength and volatility across a venue.

Two screens the server could not run before, each one scanner request:

  * **Trend strength.** ``top_gainers`` answers "what moved"; this answers
    "what is actually trending". A symbol up 9% on a single candle with ADX
    14 is noise; one up 3% with ADX 38 and +DI above -DI is a trend. The
    columns exist on the scanner (measured live: ADX 44.6, +DI 8.26,
    -DI 33.35 on BTC) and ``coin_analysis`` shows them for ONE symbol — no
    tool screened on them.
  * **Volatility ranking.** Position size follows volatility: the same
    percentage stop is a different trade on a 2%-a-day symbol than a
    15%-a-day one. Gives the ATR sizing in ``crypto_trade_plan`` the
    venue-wide context it otherwise lacks.

Shares the scanner plumbing and the throttle envelope with
``market_structure_service``, so a dead scanner reports the same way here.
"""
from __future__ import annotations

from typing import List, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.indicators import RELATIVE_VOLUME
from tradingview_mcp.core.services.market_structure_service import (
    _DAILY_ONLY_INTERVAL,
    _screener_for,
    _throttled_envelope,
    _venue_tickers,
)
from tradingview_mcp.core.services.screener_provider import resilient_scan_columns

# ── Trend strength ────────────────────────────────────────────────────────────

_TREND_COLUMNS = ["close", "ADX", "ADX+DI", "ADX-DI", "EMA50", "P.SAR",
                  "RSI", "ATR", RELATIVE_VOLUME]

_ADX_TRENDING = 25.0   # the conventional line between trend and chop
_ADX_STRONG = 40.0

TREND_DIRECTIONS = ("both", "bullish", "bearish")


def scan_trend_strength(
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    direction: str = "both",
    min_adx: float = 25.0,
    limit: int = 20,
    min_rel_volume: Optional[float] = None,
) -> dict:
    """Rank a venue by trend strength rather than by today's move.

    Args:
        exchange:       Venue to scan.
        timeframe:      Interval for the indicators.
        direction:      ``both``, ``bullish`` (+DI > -DI) or ``bearish``.
        min_adx:        Minimum ADX, 0-100. Below ~25 is conventionally chop.
        limit:          Rows to return, 1-100.
        min_rel_volume: Drop symbols below this relative volume.
    """
    if direction not in TREND_DIRECTIONS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"direction must be one of {', '.join(TREND_DIRECTIONS)}",
            valid_directions=list(TREND_DIRECTIONS),
        )
    if not 0 <= min_adx <= 100:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"min_adx must be 0-100, got {min_adx}"
        )
    if not 1 <= limit <= 100:
        return make_error(ErrorCode.INVALID_PARAMETER, f"limit must be 1-100, got {limit}")

    tickers = _venue_tickers(exchange)
    if not tickers:
        return make_error(
            ErrorCode.INVALID_EXCHANGE, f"No symbol list shipped for {exchange!r}",
            exchange=exchange,
        )
    try:
        rows = resilient_scan_columns(
            _screener_for(exchange), timeframe, tickers, _TREND_COLUMNS
        )
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Scan failed: {exc}", exchange=exchange.upper()
        )
    if not rows:
        return _throttled_envelope(exchange, len(tickers), "scan_trend_strength")

    out: List[dict] = []
    no_adx = 0
    for ticker, row in rows.items():
        adx = row.get("ADX")
        if adx is None:
            no_adx += 1
            continue
        if adx < min_adx:
            continue
        rel_vol = row.get(RELATIVE_VOLUME)
        if min_rel_volume is not None and (rel_vol or 0) < min_rel_volume:
            continue
        plus_di, minus_di = row.get("ADX+DI"), row.get("ADX-DI")
        if plus_di is None or minus_di is None:
            continue
        bullish = plus_di > minus_di
        if direction == "bullish" and not bullish:
            continue
        if direction == "bearish" and bullish:
            continue
        close, sar, ema50 = row.get("close"), row.get("P.SAR"), row.get("EMA50")
        out.append({
            "symbol": ticker.split(":")[-1],
            "ticker": ticker,
            "price": close,
            "adx": round(adx, 1),
            "trend_strength": (
                "very strong" if adx >= _ADX_STRONG
                else "strong" if adx >= _ADX_TRENDING else "weak"
            ),
            "direction": "bullish" if bullish else "bearish",
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "di_spread": round(abs(plus_di - minus_di), 1),
            "sar_side": (
                None if not (close and sar)
                else "below price (supporting)" if sar < close
                else "above price (resisting)"
            ),
            "above_ema50": None if not (close and ema50) else close > ema50,
            "rsi": round(row["RSI"], 1) if row.get("RSI") is not None else None,
            "atr": row.get("ATR"),
            "volume_ratio": round(rel_vol, 2) if rel_vol is not None else None,
        })

    # Strongest trend first, then the widest directional spread.
    out.sort(key=lambda r: (-r["adx"], -r["di_spread"]))
    return {
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "indicator": "ADX with +DI/-DI, Parabolic SAR side and EMA50 position",
        "universe_size": len(tickers),
        "symbols_with_data": len(rows),
        "symbols_without_adx": no_adx,
        "matched": len(out),
        "rows": out[:limit],
        "filters": {
            "direction": direction, "min_adx": min_adx,
            "min_rel_volume": min_rel_volume,
        },
        "thresholds": {"trending": _ADX_TRENDING, "very_strong": _ADX_STRONG},
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── Volatility ranking ────────────────────────────────────────────────────────

# Like the performance columns, these exist only on the daily timeframe.
_VOLATILITY_COLUMN = {"1D": "Volatility.D", "1W": "Volatility.W", "1M": "Volatility.M"}
_VOL_COLUMNS = ["close", "ATR", "RSI", RELATIVE_VOLUME] + list(_VOLATILITY_COLUMN.values())


def rank_volatility(
    exchange: str = "KUCOIN",
    window: str = "1D",
    order: str = "highest",
    limit: int = 20,
    min_rel_volume: Optional[float] = None,
) -> dict:
    """Rank a venue by realised volatility, with ATR as a price-relative check.

    Args:
        exchange:       Venue to rank.
        window:         ``1D``, ``1W`` or ``1M``.
        order:          ``highest`` or ``lowest`` volatility first.
        limit:          Rows to return, 1-100.
        min_rel_volume: Drop symbols below this relative volume.
    """
    if window not in _VOLATILITY_COLUMN:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"Invalid window: {window}",
            valid_windows=sorted(_VOLATILITY_COLUMN),
        )
    if order not in ("highest", "lowest"):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"order must be 'highest' or 'lowest', got {order!r}",
        )
    if not 1 <= limit <= 100:
        return make_error(ErrorCode.INVALID_PARAMETER, f"limit must be 1-100, got {limit}")

    column = _VOLATILITY_COLUMN[window]
    tickers = _venue_tickers(exchange)
    if not tickers:
        return make_error(
            ErrorCode.INVALID_EXCHANGE, f"No symbol list shipped for {exchange!r}",
            exchange=exchange,
        )
    try:
        rows = resilient_scan_columns(
            _screener_for(exchange), _DAILY_ONLY_INTERVAL, tickers, _VOL_COLUMNS
        )
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Scan failed: {exc}", exchange=exchange.upper()
        )
    if not rows:
        return _throttled_envelope(exchange, len(tickers), "rank_volatility")

    entries: List[dict] = []
    for ticker, row in rows.items():
        vol = row.get(column)
        if vol is None:
            continue
        rel_vol = row.get(RELATIVE_VOLUME)
        if min_rel_volume is not None and (rel_vol or 0) < min_rel_volume:
            continue
        close, atr = row.get("close"), row.get("ATR")
        entries.append({
            "symbol": ticker.split(":")[-1],
            "ticker": ticker,
            "price": close,
            f"volatility_{window.lower()}_pct": round(vol, 2),
            "atr": atr,
            "atr_pct_of_price": (
                None if not (atr and close) else round(atr / close * 100, 2)
            ),
            "rsi": round(row["RSI"], 1) if row.get("RSI") is not None else None,
            "volume_ratio": round(rel_vol, 2) if rel_vol is not None else None,
        })

    if not entries:
        return make_error(
            ErrorCode.NO_DATA,
            f"No symbols on {exchange.upper()} have {window} volatility data",
            exchange=exchange.upper(), column=column,
        )

    key = f"volatility_{window.lower()}_pct"
    entries.sort(key=lambda e: e[key], reverse=(order == "highest"))
    values = sorted(e[key] for e in entries)
    return {
        "exchange": exchange.upper(),
        "window": window,
        "volatility_column": column,
        "order": order,
        "interval_note": (
            "Volatility columns exist only on the daily timeframe, so this "
            "always queries 1D regardless of any intraday interval."
        ),
        "ranked_symbols": len(entries),
        "median_volatility_pct": values[len(values) // 2],
        "rows": entries[:limit],
        "filters": {"min_rel_volume": min_rel_volume},
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
