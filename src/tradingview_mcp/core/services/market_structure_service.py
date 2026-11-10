"""
Market Structure Service — breadth and relative strength across a whole venue.

Both tools answer questions the per-symbol tools cannot: not "how does this
pair look" but "what is the market as a whole doing, and what is leading it".
Each costs ONE scanner request for the entire exchange.

Two rules this module follows deliberately:

  * **Every percentage carries its own denominator.** A symbol the scanner has
    no EMA50 for is excluded from the above-EMA50 statistic, never counted as
    "below". Treating missing data as a value is what made the old volume
    tools report "Normal" for everything, and the ``coverage`` block in each
    result says exactly how many symbols each figure was computed over.
  * **Thresholds are written down, not implied.** The breadth label comes from
    the explicit cut-offs in ``_BREADTH_RULES`` so a caller can check the
    verdict against the numbers.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.indicators import RELATIVE_VOLUME
from tradingview_mcp.core.services.screener_provider import resilient_scan_columns
from tradingview_mcp.core.utils.validators import (
    EXCHANGE_SCREENER,
    normalize_tradingview_symbol,
)

_BREADTH_COLUMNS = ["close", "EMA50", "SMA200", "change", "RSI", RELATIVE_VOLUME]

# Perf.* and Volatility.* exist only on the daily timeframe. Requested at any
# intraday interval the scanner returns None for every symbol, so these tools
# always query 1D and say so in the result rather than silently returning a
# column full of nulls.
_PERF_COLUMN = {"1W": "Perf.W", "1M": "Perf.1M", "3M": "Perf.3M", "1Y": "Perf.Y"}
_RS_COLUMNS = ["close", "change", "RSI", RELATIVE_VOLUME] + list(_PERF_COLUMN.values())
_DAILY_ONLY_INTERVAL = "1D"

_RSI_OVERBOUGHT = 70.0
_RSI_OVERSOLD = 30.0

# (label, reasoning) chosen by the first rule whose predicate holds.
_BREADTH_RULES = (
    ("STRONG", "breadth confirms the trend",
     lambda ema, ad: ema is not None and ema >= 60 and ad is not None and ad >= 1.5),
    ("WEAK", "broad decline, not an isolated pullback",
     lambda ema, ad: ema is not None and ema <= 25 and ad is not None and ad <= 0.67),
    ("IMPROVING", "more names advancing than the trend position suggests",
     lambda ema, ad: ema is not None and ema < 50 and ad is not None and ad >= 1.5),
    ("DETERIORATING", "names rolling over while most still sit above trend",
     lambda ema, ad: ema is not None and ema > 50 and ad is not None and ad <= 0.67),
)


def _screener_for(exchange: str) -> str:
    return EXCHANGE_SCREENER.get((exchange or "").strip().lower(), "crypto")


def _pct(count: int, total: int) -> Optional[float]:
    return None if not total else round(100.0 * count / total, 1)


def _venue_tickers(exchange: str) -> List[str]:
    return [normalize_tradingview_symbol(s, exchange) for s in load_symbols(exchange)]


def _throttled_envelope(exchange: str, requested: int, context: str) -> dict:
    """The scanner answered with no rows at all.

    Distinct from "this venue has no symbols": the request was built and sent,
    and TradingView returned nothing, which in practice means its throttle is
    active (it answers a burst with empty bodies rather than an error status).
    Saying so beats reporting a market where 0% of names are above trend.
    """
    return make_error(
        ErrorCode.UPSTREAM_ERROR,
        "Scanner returned no rows — TradingView is most likely throttling "
        "this client. Retry in a minute; a burst of requests triggers it.",
        exchange=exchange.upper(),
        symbols_requested=requested,
        context=context,
        retryable=True,
    )


# ── Market breadth ────────────────────────────────────────────────────────────

def analyze_market_breadth(
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    min_rel_volume: Optional[float] = None,
) -> dict:
    """How much of an exchange is participating, in one scanner request.

    Answers what no per-symbol tool can: whether a setup is running with the
    venue or against it. A bullish chart in a market where 18% of names are
    above their EMA50 is a different proposition from the same chart at 72%.

    Args:
        exchange:       Venue to measure, e.g. ``KUCOIN`` or ``NASDAQ``.
        timeframe:      Interval for the moving-average and RSI positions.
        min_rel_volume: Drop symbols below this relative volume (illiquid
                        listings otherwise dominate a crypto venue's counts).
    """
    tickers = _venue_tickers(exchange)
    if not tickers:
        return make_error(
            ErrorCode.INVALID_EXCHANGE,
            f"No symbol list shipped for {exchange!r}",
            exchange=exchange,
        )

    try:
        rows = resilient_scan_columns(
            _screener_for(exchange), timeframe, tickers, _BREADTH_COLUMNS
        )
    except Exception as exc:
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Scan failed: {exc}", exchange=exchange.upper()
        )

    if not rows:
        return _throttled_envelope(exchange, len(tickers), "analyze_market_breadth")

    if min_rel_volume is not None:
        rows = {
            t: r for t, r in rows.items()
            if (r.get(RELATIVE_VOLUME) or 0) >= min_rel_volume
        }
        if not rows:
            return make_error(
                ErrorCode.NO_DATA,
                f"No symbols on {exchange.upper()} have relative volume >= {min_rel_volume}",
                exchange=exchange.upper(),
            )

    # Each statistic is computed over the symbols that actually have the
    # inputs for it; the denominators are reported alongside.
    ema_pairs = [(r["close"], r["EMA50"]) for r in rows.values()
                 if r.get("close") is not None and r.get("EMA50") is not None]
    sma_pairs = [(r["close"], r["SMA200"]) for r in rows.values()
                 if r.get("close") is not None and r.get("SMA200") is not None]
    changes = [r["change"] for r in rows.values() if r.get("change") is not None]
    rsis = [r["RSI"] for r in rows.values() if r.get("RSI") is not None]

    above_ema = sum(1 for c, m in ema_pairs if c > m)
    above_sma = sum(1 for c, m in sma_pairs if c > m)
    advancing = sum(1 for c in changes if c > 0)
    declining = sum(1 for c in changes if c < 0)
    unchanged = len(changes) - advancing - declining

    above_ema_pct = _pct(above_ema, len(ema_pairs))

    # With no decliners the ratio is undefined, not "equal to the advancing
    # count" — reporting a number there would invent a scale. It stays null,
    # while labelling treats the state as unambiguously broad so a clean sweep
    # doesn't fall through to MIXED.
    ad_ratio = round(advancing / declining, 2) if declining else None
    ad_for_label = ad_ratio if declining else (float("inf") if advancing else None)

    label, reasoning = "MIXED", "no clear breadth signal"
    for name, why, predicate in _BREADTH_RULES:
        if predicate(above_ema_pct, ad_for_label):
            label, reasoning = name, why
            break

    return {
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "universe_size": len(tickers),
        "symbols_with_data": len(rows),
        "trend_participation": {
            "above_ema50_pct": above_ema_pct,
            "above_ema50": above_ema,
            "above_sma200_pct": _pct(above_sma, len(sma_pairs)),
            "above_sma200": above_sma,
        },
        "advance_decline": {
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "ratio": ad_ratio,
            "ratio_note": None if declining else "undefined — no symbol declined",
            "median_change_pct": round(median(changes), 2) if changes else None,
        },
        "momentum_distribution": {
            "median_rsi": round(median(rsis), 1) if rsis else None,
            "overbought_pct": _pct(sum(1 for v in rsis if v >= _RSI_OVERBOUGHT), len(rsis)),
            "oversold_pct": _pct(sum(1 for v in rsis if v <= _RSI_OVERSOLD), len(rsis)),
            "thresholds": {"overbought": _RSI_OVERBOUGHT, "oversold": _RSI_OVERSOLD},
        },
        "assessment": {"label": label, "reasoning": reasoning},
        "coverage": {
            "ema50": len(ema_pairs),
            "sma200": len(sma_pairs),
            "change": len(changes),
            "rsi": len(rsis),
            "note": (
                "Each percentage is computed only over symbols the scanner had "
                "that input for; the counts above are those denominators."
            ),
        },
        "filters": {"min_rel_volume": min_rel_volume},
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── Relative strength ─────────────────────────────────────────────────────────

def rank_relative_strength(
    exchange: str = "KUCOIN",
    window: str = "1M",
    benchmark: str = "BTCUSDT",
    limit: int = 15,
    min_rel_volume: Optional[float] = None,
) -> dict:
    """Rank a venue by performance against a benchmark over weeks, not minutes.

    ``top_gainers`` ranks by the current bar's change, which answers a
    different question: what is moving now, not what has been leading. This
    ranks on the scanner's period-performance columns and subtracts the
    benchmark's own move, so an alt up 4% while BTC is up 9% reads as the
    laggard it is.

    Args:
        exchange:       Venue to rank.
        window:         ``1W``, ``1M``, ``3M`` or ``1Y``.
        benchmark:      Symbol to measure against (``BTCUSDT`` by default).
        limit:          Leaders and laggards to return each, 1-50.
        min_rel_volume: Drop symbols below this relative volume.
    """
    if window not in _PERF_COLUMN:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Invalid window: {window}",
            valid_windows=sorted(_PERF_COLUMN),
        )
    if not 1 <= limit <= 50:
        return make_error(ErrorCode.INVALID_PARAMETER, f"limit must be 1-50, got {limit}")

    column = _PERF_COLUMN[window]
    tickers = _venue_tickers(exchange)
    if not tickers:
        return make_error(
            ErrorCode.INVALID_EXCHANGE,
            f"No symbol list shipped for {exchange!r}",
            exchange=exchange,
        )

    bench_ticker = normalize_tradingview_symbol(benchmark, exchange)
    if bench_ticker not in tickers:
        tickers = tickers + [bench_ticker]

    try:
        rows = resilient_scan_columns(
            _screener_for(exchange), _DAILY_ONLY_INTERVAL, tickers, _RS_COLUMNS
        )
    except Exception as exc:
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Scan failed: {exc}", exchange=exchange.upper()
        )

    if not rows:
        return _throttled_envelope(exchange, len(tickers), "rank_relative_strength")

    bench_row = rows.get(bench_ticker) or {}
    bench_perf = bench_row.get(column)
    warning = None
    if bench_perf is None:
        warning = (
            f"No {window} performance for benchmark {bench_ticker} — rows are ranked on "
            "absolute performance and relative_pct is null. The ranking is still "
            "ordered, but it is not relative strength."
        )

    entries: List[dict] = []
    for ticker, row in rows.items():
        perf = row.get(column)
        if perf is None:
            continue
        if min_rel_volume is not None and (row.get(RELATIVE_VOLUME) or 0) < min_rel_volume:
            continue
        rsi = row.get("RSI")
        entries.append({
            "symbol": ticker.split(":")[-1],
            "ticker": ticker,
            "price": row.get("close"),
            f"perf_{window.lower()}_pct": round(perf, 2),
            "relative_pct": None if bench_perf is None else round(perf - bench_perf, 2),
            "outperforming": None if bench_perf is None else perf > bench_perf,
            "change_pct": round(row["change"], 2) if row.get("change") is not None else None,
            "rsi": round(rsi, 1) if rsi is not None else None,
            "volume_ratio": (
                round(row[RELATIVE_VOLUME], 2) if row.get(RELATIVE_VOLUME) is not None else None
            ),
        })

    if not entries:
        return make_error(
            ErrorCode.NO_DATA,
            f"No symbols on {exchange.upper()} have {window} performance data",
            exchange=exchange.upper(),
            column=column,
        )

    key = f"perf_{window.lower()}_pct"
    entries.sort(key=lambda e: -e[key])
    outperforming = sum(1 for e in entries if e.get("outperforming") is True)

    result: Dict[str, Any] = {
        "exchange": exchange.upper(),
        "window": window,
        "performance_column": column,
        "interval_note": (
            "Performance columns exist only on the daily timeframe, so this "
            "always queries 1D regardless of any intraday interval."
        ),
        "benchmark": {
            "symbol": benchmark.upper(),
            "ticker": bench_ticker,
            f"perf_{window.lower()}_pct": None if bench_perf is None else round(bench_perf, 2),
            "price": bench_row.get("close"),
        },
        "ranked_symbols": len(entries),
        "outperforming_benchmark": None if bench_perf is None else outperforming,
        "underperforming_benchmark": (
            None if bench_perf is None else len(entries) - outperforming
        ),
        "leaders": entries[:limit],
        "laggards": entries[-limit:][::-1],
        "filters": {"min_rel_volume": min_rel_volume},
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
    if warning:
        result["warning"] = warning
    return result
