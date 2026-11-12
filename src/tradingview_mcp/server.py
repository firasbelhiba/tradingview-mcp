"""
TradingView MCP Server — routing layer only.

Each @mcp.tool() handler is responsible for:
  1. Validating / sanitising parameters
  2. Delegating to the appropriate service module
  3. Returning the result

No business logic lives here. All computation is in core/services/*.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Annotated, Optional

from pydantic import Field

# Eagerly import pandas on the main thread. Several tools route through
# tradingview_screener's bare Query(), whose first call imports pandas lazily —
# and under the stdio server that first import happens inside an
# anyio.to_thread worker, deadlocking against the interpreter's import lock
# while the event loop awaits the worker (issue #91: seven tools hung forever
# on a fresh stdio process). Preloading here makes the in-worker import a no-op.
import pandas  # noqa: F401

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

# ── Service imports ────────────────────────────────────────────────────────────
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.screener_service import (
    fetch_bollinger_analysis,
    fetch_trending_analysis,
    analyze_coin,
    scan_consecutive_candles,
    scan_advanced_candle_patterns_single_tf,
    fetch_multi_timeframe_patterns,
    run_multi_timeframe_analysis,
)
from tradingview_mcp.core.services.scanner_service import (
    volume_breakout_scan,
    volume_confirmation_analyze,
    smart_volume_scan,
)
from tradingview_mcp.core.services.multi_agent_service import run_multi_agent_analysis
from tradingview_mcp.core.services.egx_service import (
    get_egx_market_overview,
    scan_egx_sector,
    run_egx_sector_scanner,
    analyze_egx_index,
    screen_egx_stocks,
    generate_egx_trade_plan,
    analyze_egx_fibonacci,
)
from tradingview_mcp.core.services.marketaux_service import (
    analyze_sentiment,
    fetch_news_summary,
)
from tradingview_mcp.core.services.smart_money_service import (
    analyze_smart_money,
    scan_egx_smart_money,
)
from tradingview_mcp.core.services.yahoo_finance_service import (
    get_price,
    get_price_async,
    get_market_snapshot,
)
from tradingview_mcp.core.services.bitcoin_market_service import get_bitcoin_market_pulse
from tradingview_mcp.core.services.extended_hours_service import (
    get_extended_hours_price,
    get_extended_hours_price_async,
)
from tradingview_mcp.core.services.options_service import (
    get_options_chain,
    get_unusual_options_activity,
)
from tradingview_mcp.core.services.futures_service import (
    get_futures_overview,
    get_futures_movers,
    get_futures_category_snapshot,
    get_futures_watchlist,
)
from tradingview_mcp.core.services.stock_screener_service import (
    EXAMPLE_MARKETS,
    fetch_stock_prices,
    screen_stocks,
)
from tradingview_mcp.core.services.backtest_service import (
    run_backtest,
    compare_strategies as _compare_strategies,
    walk_forward_backtest,
)
from tradingview_mcp.core.services.positioning_service import analyze_positioning
from tradingview_mcp.core.services.positioning_scan_service import (
    scan_squeeze_risk,
    screen_funding,
)
from tradingview_mcp.core.services.alert_service import (
    check_alerts,
    create_alert,
    delete_alert as _delete_alert,
    list_alerts as _list_alerts,
    metric_catalogue,
    rearm_alert,
)
from tradingview_mcp.core.services.market_structure_service import (
    analyze_market_breadth,
    rank_relative_strength,
)
from tradingview_mcp.core.services.trend_scan_service import (
    rank_volatility,
    scan_trend_strength,
)
from tradingview_mcp.core.services.correlation_service import (
    analyze_correlation,
    analyze_seasonality,
)
from tradingview_mcp.core.services.sentiment_index_service import (
    get_fear_greed,
    get_narrative_rotation,
)
from tradingview_mcp.core.services.copytrading_service import (
    get_lead_trader,
    list_lead_traders,
)
from tradingview_mcp.core.services.copytrading_trades_service import (
    get_lead_trader_trades,
)
from mcp.types import CallToolResult, TextContent

from tradingview_mcp import apps, wallet_app
from tradingview_mcp.core.services.brief_service import market_brief as _market_brief
from tradingview_mcp.core.services.bubbles_service import crypto_bubbles as _crypto_bubbles
from tradingview_mcp.core.services.paper_copy import copy_trader as _copy_trader
from tradingview_mcp.core.services.paper_orders import (
    cancel_order as _cancel_order,
    list_orders as _list_orders,
    place_order as _place_order,
)
from tradingview_mcp.core.services.onboarding_service import (
    alert_schedule_recipe as _alert_schedule,
    get_started as _get_started,
)
from tradingview_mcp.core.services.paper_trading_service import (
    close_trade as _paper_close,
    modify_trade as _paper_modify,
    open_trade as _paper_open,
    reset_wallet as _paper_reset,
    trade_history as _paper_history,
    wallet_status as _paper_status,
    wallet_names as _paper_wallets,
)
from tradingview_mcp.core.services.crypto_analysis_service import (
    analyze_crypto_fibonacci,
    build_crypto_watchlist,
    detect_rsi_divergence,
    generate_crypto_trade_plan,
    scan_donchian_breakout,
    scan_supertrend,
)
from tradingview_mcp.core.utils.validators import (
    validate_timeframe,
    validate_exchange,
    normalize_tradingview_symbol,
    normalize_yahoo_symbol,
)
from tradingview_mcp.core.errors import (
    ErrorCode,
    exception_to_envelope,
    make_error,
)

try:
    import tradingview_screener  # noqa: F401
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


# ── MCP server instance ────────────────────────────────────────────────────────

mcp = FastMCP(
    name="TradingView Multi-Market Screener",
    instructions=(
        "Multi-market screener backed by TradingView. "
        "Supports crypto exchanges (KuCoin, Binance, Bybit, MEXC, etc.), stock markets "
        "(EGX, BIST, NASDAQ, NYSE, Bursa Malaysia, HKEX, SSE, SZSE, TWSE, TPEX), "
        "and futures markets (CME, COMEX, NYMEX, CBOT — equity index, energy, metals, "
        "agriculture, rates, forex, crypto futures). "
        "Tools: top_gainers, top_losers, bollinger_scan, coin_analysis, multi_agent_analysis, "
        "volume_breakout_scanner, futures_market_overview, futures_top_movers, "
        "futures_category_snapshot, futures_watchlist, egx_market_overview, "
        "crypto_trade_plan, crypto_fibonacci_retracement, crypto_watchlist, "
        "rsi_divergence, supertrend_scan, donchian_breakout_scan, positioning, "
        "squeeze_scan, funding_screener, market_breadth, relative_strength, "
        "alerts and a simulated paper-trading wallet, and more.\n\n"
        # Guidance to the assistant, carried on every connection.
        #
        # The server cannot draw: it can emit a ui:// resource, but a local
        # desktop host ignores it (measured 2026-11-11). The assistant
        # usually can draw, so the server asks rather than hoping the person
        # thinks to. Kept to presentation — it never tells the assistant what
        # to conclude about a market, only how to show what it found.
        "PRESENTATION — when the person is LOOKING at state rather than you "
        "reading it to reason with, render it inline as a compact visual "
        "panel if you have any way to draw one (an artifact, an inline "
        "HTML/SVG widget, a canvas), without asking first and without "
        "repeating the numbers in prose underneath. This is worth most for: "
        "every paper wallet result, not only the ones that read it — the "
        "moment after someone creates a wallet, opens a trade or closes "
        "one is when they are most certainly looking, and each result "
        "carries a display_hint saying what to draw; "
        "market_breadth and relative_strength, squeeze_scan and "
        "funding_screener, and lead_trader comparisons. A panel of live "
        "marks is a photograph: stamp it with the result's as_of, and where "
        "your surface can send a prompt back, give it a control that re-asks "
        "for the same view — and, for the wallet, one per position that "
        "closes it — rather than leaving the person to retype the question. "
        "Always surface a "
        "failure a glance could miss — unsettled_positions, "
        "unavailable_feeds, not_evaluated, price_errors — rather than only "
        "the healthy numbers. Skip all of this if you have no rendering "
        "tool, or the question is narrow enough that one number answers it."
    ),
)


# The wallet panel: a ui:// page and the capability that says views are
# served. Registered before the tools so the one that points at it can.
_WALLET_APP_REGISTERED = wallet_app.register(mcp)


# ── Screener tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Top Gainers Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def top_gainers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict] | dict:
    """Return top gainers for an exchange and timeframe using Bollinger Band analysis.

    Args:
        exchange: Exchange name — crypto: KUCOIN, BINANCE, BYBIT, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX, EPA (Euronext Paris), AMS, BRU, LIS, MIL, LSE, SIX, BME, XETRA, FWB, TSX, TSXV, NSE, BSE (India), IDX (Indonesia)
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)

    Returns:
        list[dict] on success. On ANY failure returns a structured error
        envelope ``{"error": {"code": ..., "retryable": ...}}`` — never a
        raw exception string.
    """
    limit = max(1, min(limit, 50))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        # Underlying tradingview-screener is sync (uses urllib). Push to a
        # worker thread so the event loop is free for other concurrent
        # tool calls.
        rows = await asyncio.to_thread(
            fetch_trending_analysis, exchange, timeframe=timeframe, limit=limit
        )
    except Exception as e:
        return exception_to_envelope(e, context="top_gainers")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Top Losers Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def top_losers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict] | dict:
    """Return top losers for an exchange and timeframe. Supports crypto (KUCOIN, BINANCE, MEXC) and stocks (EGX, BIST, NASDAQ).

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    limit = max(1, min(limit, 50))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        # sort="asc" makes the service sort BEFORE truncating to limit —
        # re-sorting after fetch returned the smallest of the top GAINERS.
        rows = fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit, sort="asc")
    except Exception as e:
        return exception_to_envelope(e, context="top_losers")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Bollinger Squeeze Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def bollinger_scan(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict] | dict:
    """Scan for assets with low Bollinger Band Width (squeeze detection). Works with crypto and stocks.

    This scans a whole EXCHANGE for squeezes (canonical name is exactly
    `bollinger_scan`; there is no "get_bollinger_band_analysis" tool). For
    the Bollinger read of ONE symbol, call `coin_analysis` instead.

    Example: bollinger_scan(exchange="BINANCE", timeframe="15m", bbw_threshold=0.008)

    Args:
        exchange: Exchange — crypto: KUCOIN, BINANCE, BYBIT, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX, EPA (Euronext Paris), AMS, BRU, LIS, MIL, LSE, SIX, BME, XETRA, FWB, TSX, TSXV, NSE, BSE (India), IDX (Indonesia)
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M. Typical squeeze thresholds: 15m→0.008, 1h→0.02, 4h→0.04, 1D→0.12
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    limit = max(1, min(limit, 100))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "4h")
        rows = fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    except Exception as e:
        return exception_to_envelope(e, context="bollinger_scan")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Bollinger Rating Filter", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def rating_filter(exchange: str = "KUCOIN", timeframe: str = "5m", rating: int = 2, limit: int = 25) -> list[dict] | dict:
    """Filter coins by Bollinger Band rating.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, MEXC, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "5m")
        rows = fetch_trending_analysis(exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit)
    except Exception as e:
        return exception_to_envelope(e, context="rating_filter")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


# ── Coin / asset analysis ──────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Full Technical Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def coin_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Get detailed analysis for a specific asset (coin or stock) on specified exchange and timeframe.

    This is the canonical single-symbol technical readout (there is no
    "get_technical_analysis" or "get_technical_summary" tool — use THIS one).
    Use `multi_timeframe_analysis` instead when you need trend alignment
    across several timeframes, and `combined_analysis` when you also want
    news sentiment + headlines in the same call.

    Example: coin_analysis(symbol="BTCUSDT", exchange="BINANCE", timeframe="1h")

    Args:
        symbol: Bare ticker, no exchange prefix — crypto: "BTCUSDT", "ETHUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX, EPA (Euronext Paris), AMS, BRU, LIS, MIL, LSE, SIX, BME, XETRA, FWB, TSX, TSXV, NSE, BSE (India), IDX (Indonesia). If the symbol isn't listed there, the error's `listed_on` field names exchanges that do list it.
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)

    Returns:
        Detailed analysis with all indicators and metrics
    """
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        return analyze_coin(symbol, exchange, timeframe)
    except Exception as e:
        return exception_to_envelope(e, context="coin_analysis")


# ── Candle pattern tools ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Consecutive Candles Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def consecutive_candles_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    pattern_type: str = "bullish",
    candle_count: int = 3,
    min_growth: float = 2.0,
    limit: int = 20,
) -> dict:
    """Scan for coins with consecutive growing/shrinking candles pattern.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h)
        pattern_type: "bullish" (growing candles) or "bearish" (shrinking candles)
        candle_count: Number of consecutive candles to check (2-5)
        min_growth: Minimum growth percentage for each candle
        limit: Maximum number of results to return
    """
    candle_count = max(2, min(5, candle_count))
    min_growth = max(0.5, min(20.0, min_growth))
    limit = max(1, min(50, limit))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        return scan_consecutive_candles(exchange, timeframe, pattern_type, candle_count, min_growth, limit)
    except Exception as e:
        return exception_to_envelope(e, context="consecutive_candles_scan")


@mcp.tool(annotations=ToolAnnotations(title="Candlestick Pattern Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def advanced_candle_pattern(
    exchange: str = "KUCOIN",
    base_timeframe: str = "15m",
    pattern_length: int = 3,
    min_size_increase: float = 10.0,
    limit: int = 15,
) -> dict:
    """Advanced candle pattern analysis using multi-timeframe data.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        base_timeframe: Base timeframe for analysis (5m, 15m, 1h, 4h)
        pattern_length: Number of consecutive periods to analyse (2-4)
        min_size_increase: Minimum percentage increase in candle size
        limit: Maximum number of results to return
    """
    pattern_length = max(2, min(4, pattern_length))
    min_size_increase = max(5.0, min(50.0, min_size_increase))
    limit = max(1, min(30, limit))

    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        base_timeframe = validate_timeframe(base_timeframe, "15m")

        symbols = load_symbols(exchange)
        if not symbols:
            return make_error(ErrorCode.NO_DATA, f"No symbols found for exchange: {exchange}", exchange=exchange)
        symbols = symbols[: min(limit * 2, 100)]

        fallback_reason = None
        if TRADINGVIEW_SCREENER_AVAILABLE:
            try:
                results = fetch_multi_timeframe_patterns(exchange, symbols, base_timeframe, pattern_length, min_size_increase)
                return {
                    "exchange": exchange,
                    "base_timeframe": base_timeframe,
                    "pattern_length": pattern_length,
                    "min_size_increase": min_size_increase,
                    "method": "multi-timeframe",
                    "total_found": len(results),
                    "data": results[:limit],
                }
            except Exception as exc:
                # Fall through to the single-timeframe fallback — but say so,
                # instead of silently returning a different analysis method.
                fallback_reason = repr(exc)
        else:
            fallback_reason = "tradingview-screener not installed"

        result = scan_advanced_candle_patterns_single_tf(exchange, symbols, base_timeframe, pattern_length, min_size_increase, limit)
        if isinstance(result, dict):
            result["fallback_reason"] = fallback_reason
        return result
    except Exception as e:
        return exception_to_envelope(e, context="advanced_candle_pattern")


# ── Volume scanner tools ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Volume Breakout Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def volume_breakout_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    volume_multiplier: float = 2.0,
    price_change_min: float = 3.0,
    limit: int = 25,
) -> list[dict] | dict:
    """Detect coins with volume breakout + price breakout.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, MEXC, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        volume_multiplier: How many times the volume should be above normal level (default 2.0)
        price_change_min: Minimum price change percentage (default 3.0)
        limit: Number of rows to return (max 50)

    Returns ``list[dict]`` on success, or an error envelope on total upstream
    failure (``{"error": {"code": "ALL_BATCHES_FAILED", ...}}``). The empty
    list now strictly means "no matches today"; rate-limit cliffs surface
    explicitly.
    """
    volume_multiplier = max(1.5, min(10.0, volume_multiplier))
    price_change_min = max(1.0, min(20.0, price_change_min))
    limit = max(1, min(limit, 50))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        # volume_breakout_scan makes a blocking whole-exchange scanner
        # request. Off-load to a worker thread to keep the event loop
        # responsive for other concurrent tool calls.
        return await asyncio.to_thread(
            volume_breakout_scan,
            exchange, timeframe, volume_multiplier, price_change_min, limit,
        )
    except Exception as e:
        return exception_to_envelope(e, context="volume_breakout_scanner")


@mcp.tool(annotations=ToolAnnotations(title="Volume Confirmation Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def volume_confirmation_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Detailed volume confirmation analysis for a specific coin.

    Args:
        symbol: Coin symbol (e.g., BTCUSDT)
        exchange: Exchange name
        timeframe: Time frame for analysis
    """
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        return volume_confirmation_analyze(symbol, exchange, timeframe)
    except Exception as e:
        return exception_to_envelope(e, context="volume_confirmation_analysis")


@mcp.tool(annotations=ToolAnnotations(title="Smart Volume Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def smart_volume_scanner(
    exchange: str = "KUCOIN",
    min_volume_ratio: float = 2.0,
    min_price_change: float = 2.0,
    rsi_range: str = "any",
    limit: int = 20,
) -> list[dict] | dict:
    """Smart volume + technical analysis combination scanner.

    Args:
        exchange: Exchange name
        min_volume_ratio: Minimum volume multiplier (default 2.0)
        min_price_change: Minimum price change percentage (default 2.0)
        rsi_range: "oversold" (<30), "overbought" (>70), "neutral" (30-70), "any"
        limit: Number of results (max 30)

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    min_volume_ratio = max(1.2, min(10.0, min_volume_ratio))
    min_price_change = max(0.5, min(20.0, min_price_change))
    limit = max(1, min(limit, 30))
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        return smart_volume_scan(exchange, min_volume_ratio, min_price_change, rsi_range, limit)
    except Exception as e:
        return exception_to_envelope(e, context="smart_volume_scanner")


# ── Multi-agent analysis ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Multi-Agent Market Debate", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def multi_agent_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Run a multi-agent debate (Technical, Sentiment, Risk) for a specific symbol.

    Args:
        symbol: Symbol — crypto: "BTCUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX), "GDX" (AMEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, AMEX, NYSEARCA, PCX, SSE, SZSE, TWSE, TPEX, EPA (Euronext Paris), AMS, BRU, LIS, MIL, LSE, SIX, BME, XETRA, FWB, TSX, TSXV, NSE, BSE (India), IDX (Indonesia)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W)

    Returns:
        A structured debate between 3 AI agents culminating in a final trading decision.
    """
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        timeframe = validate_timeframe(timeframe, "15m")
        full_symbol = normalize_tradingview_symbol(symbol, exchange)
        return run_multi_agent_analysis(full_symbol, exchange, timeframe)
    except Exception as e:
        return exception_to_envelope(e, context="multi_agent_analysis")


# ── EGX market tools ───────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="EGX Market Overview", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_market_overview(timeframe: str = "1D", limit: int = 10) -> dict:
    """Get a comprehensive overview of the Egyptian Exchange (EGX) market.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D for stocks)
        limit: Number of stocks per category (max 20)
    """
    limit = max(1, min(limit, 20))
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return get_egx_market_overview(timeframe, limit)
    except Exception as e:
        return exception_to_envelope(e, context="egx_market_overview")


@mcp.tool(annotations=ToolAnnotations(title="EGX Sector Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_sector_scan(sector: str = "", timeframe: str = "1D", limit: int = 20) -> dict:
    """Scan EGX stocks by sector. Shows available sectors if none specified.

    Args:
        sector: Sector name (banks, healthcare_and_pharma, real_estate, etc.)
                Leave empty to list all sectors.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Max results per sector (max 50)
    """
    limit = max(1, min(limit, 50))
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return scan_egx_sector(sector, timeframe, limit)
    except Exception as e:
        return exception_to_envelope(e, context="egx_sector_scan")


@mcp.tool(annotations=ToolAnnotations(title="EGX Sector Rotation Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_sector_scanner(
    timeframe: str = "1D",
    top_n_sectors: int = 5,
    top_n_stocks: int = 3,
    min_stock_score: int = 60,
) -> dict:
    """Sector rotation scanner for EGX — identifies hot/cold sectors and top picks.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        top_n_sectors: Number of top sectors to show stock picks for (1-18, default 5)
        top_n_stocks: Number of top stocks per highlighted sector (1-10, default 3)
        min_stock_score: Minimum stock score for picks (0-100, default 60)
    """
    top_n_sectors = max(1, min(18, top_n_sectors))
    top_n_stocks = max(1, min(10, top_n_stocks))
    min_stock_score = max(0, min(100, min_stock_score))
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return run_egx_sector_scanner(timeframe, top_n_sectors, top_n_stocks, min_stock_score)
    except Exception as e:
        return exception_to_envelope(e, context="egx_sector_scanner")


@mcp.tool(annotations=ToolAnnotations(title="EGX Index Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_index_analysis(index: str = "EGX30", timeframe: str = "1D", limit: int = 30) -> dict:
    """Analyse an EGX index showing constituent performance with full indicators.

    Args:
        index: EGX30, EGX70, EGX100, SHARIAH33, EGX35LV, TAMAYUZ
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        limit: Number of stocks to show in detail (max 100)
    """
    limit = max(1, min(limit, 100))
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return analyze_egx_index(index, timeframe, limit)
    except Exception as e:
        return exception_to_envelope(e, context="egx_index_analysis")


@mcp.tool(annotations=ToolAnnotations(title="EGX Stock Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_stock_screener(
    timeframe: str = "1D",
    min_score: int = 55,
    index_filter: str = "",
    limit: int = 20,
) -> dict:
    """Production stock ranking engine for EGX — finds strong stocks with actionable setups.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        min_score: Minimum stock score to include (0-100, default 55)
        index_filter: Filter by index — EGX30, EGX70, EGX100, SHARIAH33, EGX35LV, TAMAYUZ
        limit: Number of results (max 50)
    """
    min_score = max(0, min(100, min_score))
    limit = max(1, min(50, limit))
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return screen_egx_stocks(timeframe, min_score, index_filter, limit)
    except Exception as e:
        return exception_to_envelope(e, context="egx_stock_screener")


@mcp.tool(annotations=ToolAnnotations(title="EGX Trade Plan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_trade_plan(symbol: str, timeframe: str = "1D") -> dict:
    """Generate a full trade plan for a specific EGX stock.

    Args:
        symbol: EGX stock symbol (e.g., "COMI", "TMGH", "FWRY")
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
    """
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return generate_egx_trade_plan(symbol, timeframe)
    except Exception as e:
        return exception_to_envelope(e, context="egx_trade_plan")


@mcp.tool(annotations=ToolAnnotations(title="EGX Fibonacci Retracement", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_fibonacci_retracement(symbol: str, lookback: str = "52W", timeframe: str = "1D") -> dict:
    """Fibonacci retracement analysis for EGX stocks.

    Args:
        symbol: EGX stock symbol (e.g., "COMI", "TMGH", "FWRY")
        lookback: Period for swing high/low — "1M", "3M", "6M", "52W", "ALL" (default 52W)
        timeframe: Analysis timeframe (5m, 15m, 1h, 4h, 1D, 1W, 1M — default 1D)
    """
    lookback = lookback.strip().upper()
    try:
        timeframe = validate_timeframe(timeframe, "1D")
        return analyze_egx_fibonacci(symbol, lookback, timeframe)
    except Exception as e:
        return exception_to_envelope(e, context="egx_fibonacci_retracement")


# ── Smart-money / banker-fund tools ───────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Smart Money Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def smart_money_analysis(
    symbol: str,
    exchange: str = "EGX",
    period: str = "6mo",
    interval: str = "1d",
) -> dict:
    """Banker/hot-money proxy read for one symbol from price+volume history.

    Three independent families, computed from Yahoo Finance OHLCV:
      - mcdx_style: RSI-horizon decomposition (banker RSI50 / hot-money RSI40
        / retail RSI14, each scaled 0-20) — sustained slow-horizon strength
        that fast horizons can't fake reads as institutional accumulation.
      - banker_fund_oscillator: EMA(13)-smoothed 34-bar stochastic of the
        weighted typical price, 0-100, with BANKER_ENTRY/EXIT states.
      - volume_flow_composite: CMF(20) + MFI(14) + OBV trend + up/down volume
        balance → 0-100 accumulation score.
    Plus a consensus vote across the three.

    These are documented, reproducible formula variants inspired by the
    public "banker fund" indicator family — NOT actual institutional-flow
    data, and not copies of any closed-source script.

    Args:
        symbol: Bare ticker — EGX: "COMI", "TMGH"; US: "AAPL"; crypto pairs
            work via Yahoo notation ("BTC-USD").
        exchange: EGX (maps to Yahoo .CA), BIST (.IS), TADAWUL (.SR); anything
            else passes the symbol to Yahoo unchanged.
        period: 1mo | 3mo | 6mo | 1y | 2y (default 6mo)
        interval: 1d (default) or 1h
    """
    try:
        return analyze_smart_money(symbol, exchange, period, interval)
    except Exception as e:
        return exception_to_envelope(e, context="smart_money_analysis")


@mcp.tool(annotations=ToolAnnotations(title="EGX Smart Money Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_smart_money_scanner(
    index: str = "EGX30",
    limit: int = 10,
    period: str = "6mo",
    min_score: float = 0.0,
) -> dict:
    """Rank EGX index constituents by smart-money accumulation evidence.

    Runs the full smart_money_analysis per constituent (Yahoo .CA history,
    bounded thread pool) and ranks by the volume-flow composite score, with
    the banker decomposition and oscillator state attached per row. Rows the
    data source can't serve are listed under `skipped`, never silently
    dropped.

    Args:
        index: EGX30 | EGX70 | EGX100 | SHARIAH33 | EGX35LV | TAMAYUZ
        limit: Max ranked rows to return (default 10, max 50)
        period: History window — 1mo | 3mo | 6mo | 1y | 2y (default 6mo)
        min_score: Only include rows with composite score >= this (0-100)
    """
    try:
        return scan_egx_smart_money(index, limit, period, min_score)
    except Exception as e:
        return exception_to_envelope(e, context="egx_smart_money_scanner")


# ── Multi-timeframe analysis ───────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Multi-Timeframe Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def multi_timeframe_analysis(symbol: str, exchange: str = "KUCOIN") -> dict:
    """Multi-timeframe alignment analysis (Weekly → Daily → 4H → 1H → 15m).

    Canonical name is exactly `multi_timeframe_analysis` (there is no
    "get_multi_timeframe_analysis" tool). Use this for cross-timeframe trend
    alignment on ONE symbol; for a single-timeframe deep dive use
    `coin_analysis`; for TA + sentiment + news use `combined_analysis`.

    Example: multi_timeframe_analysis(symbol="SOLUSDT", exchange="BINANCE")

    Args:
        symbol: Bare ticker, no exchange prefix — crypto: "BTCUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX), "GDX" (AMEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, AMEX, NYSEARCA, PCX, SSE, SZSE, TWSE, TPEX, EPA (Euronext Paris), AMS, BRU, LIS, MIL, LSE, SIX, BME, XETRA, FWB, TSX, TSXV, NSE, BSE (India), IDX (Indonesia)
    """
    try:
        exchange = validate_exchange(exchange, "KUCOIN")
        full_symbol = normalize_tradingview_symbol(symbol, exchange)
        # tradingview_ta backs this with a single threading.Semaphore-gated
        # request; pushing the whole call to a thread keeps the event loop
        # free while we wait on the upstream HTTP.
        return await asyncio.to_thread(run_multi_timeframe_analysis, full_symbol, exchange)
    except Exception as e:
        return exception_to_envelope(e, context="multi_timeframe_analysis")


# ── Sentiment & news tools ─────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Market News Sentiment", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def market_sentiment(symbol: str, category: str = "all", limit: int = 20) -> dict:
    """News sentiment for stocks and crypto (licensed Marketaux entity sentiment).

    Args:
        symbol: Asset symbol ("AAPL", "BTC", "ETH", "TSLA")
        category: News group to search ("crypto", "stocks", "all")
        limit: Max articles to analyse
    """
    return analyze_sentiment(symbol, category, limit)


@mcp.tool(annotations=ToolAnnotations(title="Financial News Feed", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def financial_news(symbol: Optional[str] = None, category: str = "stocks", limit: int = 10) -> dict:
    """Real-time financial news via Marketaux (licensed).

    Args:
        symbol: Optional symbol filter ("AAPL", "BTC"). None = all news.
        category: News category ("crypto", "stocks", "all")
        limit: Max number of news items
    """
    # The Marketaux fetch is sync (urllib) — offload so parallel news
    # requests don't block the event loop.
    return await asyncio.to_thread(fetch_news_summary, symbol, category, limit)


@mcp.tool(annotations=ToolAnnotations(title="Combined TA + Sentiment + News", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def combined_analysis(symbol: str, exchange: str = "NASDAQ", timeframe: str = "1D") -> dict:
    """POWER TOOL: TradingView technical analysis + news sentiment + financial news.

    Use this when you want TA AND sentiment AND news for one symbol in a
    single call. For indicators only, `coin_analysis` is faster; for
    cross-timeframe trend alignment use `multi_timeframe_analysis`.

    Example: combined_analysis(symbol="NVDA", exchange="NASDAQ", timeframe="1D")

    Args:
        symbol: Bare ticker, no exchange prefix ("AAPL", "BTCUSDT", "THYAO", "GDX")
        exchange: Exchange (NASDAQ, NYSE, AMEX, NYSEARCA, PCX, BINANCE, KUCOIN, MEXC, BIST, EGX, TWSE, TPEX)
        timeframe: Analysis timeframe (5m, 15m, 1h, 4h, 1D, 1W)
    """
    try:
        exchange_clean = validate_exchange(exchange, "NASDAQ")
        timeframe_clean = validate_timeframe(timeframe, "1D")
    except Exception as e:
        return exception_to_envelope(e, context="combined_analysis")
    cat = "crypto" if exchange_clean.upper() in ["BINANCE", "KUCOIN", "BYBIT", "MEXC"] else "stocks"

    # The three sub-calls hit independent upstreams (TradingView TA,
    # Marketaux news/sentiment) so fan them out in parallel. Pre-P4 this was
    # ~3x sequential wall-clock; now bounded by the slowest single call.
    # The TA throttle (threading.Semaphore in screener_provider) still
    # caps in-flight TV calls correctly because asyncio.to_thread runs
    # the sync call in a worker thread that respects the semaphore.
    # return_exceptions so one failing upstream degrades its own section to
    # an error envelope instead of crashing the whole tool.
    tech, sentiment, news = await asyncio.gather(
        asyncio.to_thread(analyze_coin, symbol, exchange_clean, timeframe_clean),
        asyncio.to_thread(analyze_sentiment, symbol, cat),
        asyncio.to_thread(fetch_news_summary, symbol, cat, 5),
        return_exceptions=True,
    )
    if isinstance(tech, BaseException):
        tech = exception_to_envelope(tech, context="combined_analysis.technical")
    if isinstance(sentiment, BaseException):
        sentiment = exception_to_envelope(sentiment, context="combined_analysis.sentiment")
    if isinstance(news, BaseException):
        news = exception_to_envelope(news, context="combined_analysis.news")

    tech_momentum = tech.get("market_sentiment", {}).get("momentum", "") if isinstance(tech, dict) else ""
    tech_bullish = tech_momentum == "Bullish"
    sent_bullish = sentiment.get("sentiment_score", 0) > 0.1
    signals_agree = tech_bullish == sent_bullish
    confidence = "HIGH" if signals_agree else "MIXED"
    tech_signal = tech.get("market_sentiment", {}).get("buy_sell_signal", "N/A") if isinstance(tech, dict) else "N/A"

    return {
        "symbol": symbol,
        "exchange": exchange_clean,
        "timeframe": timeframe_clean,
        "technical": tech,
        "sentiment": sentiment,
        "news": {"count": news.get("count", 0), "latest": news.get("items", [])[:3]},
        "confluence": {
            "signals_agree": signals_agree,
            "confidence": confidence,
            "recommendation": (
                f"Technical {tech_signal} "
                f"{'confirmed by' if signals_agree else 'conflicts with'} "
                f"{sentiment.get('sentiment_label', 'Neutral')} news sentiment "
                f"({sentiment.get('posts_analyzed', 0)} articles analyzed)"
            ),
        },
    }


# ── Backtest tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Strategy Backtest", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    interval: str = "1d",
    include_trade_log: bool = False,
    include_equity_curve: bool = False,
) -> dict:
    """Backtest a trading strategy on historical data with institutional-grade metrics.

    Args:
        symbol: Yahoo Finance symbol (AAPL, BTC-USD, THYAO.IS, ^GSPC)
        strategy: rsi | bollinger | macd | ema_cross | supertrend | donchian
                  | rsi_pullback | keltner_breakout | triple_ema
                  (rsi_pullback and triple_ema need period >= '1y' for SMA200 warmup)
        period: '1mo', '3mo', '6mo', '1y', '2y'
        initial_capital: Starting capital in USD (default $10,000)
        commission_pct: Per-trade commission % (default 0.1%)
        slippage_pct: Per-trade slippage % (default 0.05%)
        interval: '1d' (daily) or '1h' (hourly)
        include_trade_log: Include full per-trade log (default False)
        include_equity_curve: Include equity curve data points (default False)
    """
    return run_backtest(
        symbol, strategy, period, initial_capital,
        commission_pct, slippage_pct, interval,
        include_trade_log, include_equity_curve,
    )


@mcp.tool(annotations=ToolAnnotations(title="Strategy Comparison Race", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def compare_strategies(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    interval: str = "1d",
) -> dict:
    """Run all 9 strategies (RSI, Bollinger, MACD, EMA Cross, Supertrend, Donchian, RSI Pullback, Keltner Breakout, Triple EMA) and return a ranked leaderboard.

    Args:
        symbol: Yahoo Finance symbol (AAPL, BTC-USD, SPY…)
        period: '1mo', '3mo', '6mo', '1y', '2y'
                (period >= '1y' recommended so rsi_pullback and triple_ema can
                 complete SMA200 warmup; otherwise they contribute zero trades)
        initial_capital: Starting capital in USD (default $10,000)
        interval: '1d' (daily) or '1h' (hourly)
    """
    return _compare_strategies(symbol, period, initial_capital, interval=interval)


@mcp.tool(annotations=ToolAnnotations(title="Walk-Forward Backtest", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def walk_forward_backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "2y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    interval: str = "1d",
) -> dict:
    """Walk-forward backtest to detect overfitting — validates strategy on unseen data.

    Args:
        symbol: Yahoo Finance symbol (AAPL, BTC-USD, SPY…)
        strategy: rsi | bollinger | macd | ema_cross | supertrend | donchian
                  | keltner_breakout
                  (rsi_pullback and triple_ema not supported here — SMA200 warmup
                   exceeds typical fold size; use run_backtest with period='2y')
        period: '1mo', '3mo', '6mo', '1y', '2y' (recommend '2y')
        initial_capital: Starting capital per fold in USD (default $10,000)
        commission_pct: Per-trade commission % (default 0.1%)
        slippage_pct: Per-trade slippage % (default 0.05%)
        n_splits: Number of walk-forward folds (default 3, max 10)
        train_ratio: Fraction of each fold used for training (default 0.7)
        interval: '1d' (daily) or '1h' (hourly)
    """
    return walk_forward_backtest(
        symbol, strategy, period, initial_capital,
        commission_pct, slippage_pct, n_splits, train_ratio, interval,
    )


# ── Yahoo Finance tools ────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Real-Time Price Quote", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def yahoo_price(symbol: str) -> dict:
    """Real-time price quote from Yahoo Finance for any stock, crypto, ETF or index.

    Args:
        symbol: Yahoo Finance symbol — e.g. AAPL, BTC-USD, SPY, ^GSPC, EURUSD=X, THYAO.IS
    """
    return await get_price_async(normalize_yahoo_symbol(symbol))


@mcp.tool(annotations=ToolAnnotations(title="Global Market Snapshot", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def market_snapshot() -> dict:
    """Global market overview: major indices, top crypto, FX rates, and key ETFs.
    Powered by Yahoo Finance.
    """
    return get_market_snapshot()


@mcp.tool(annotations=ToolAnnotations(title="Bitcoin Market Pulse", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def bitcoin_market_pulse() -> dict:
    """Single-call BTC macro context: price, dominance, total market cap + risk assessment.

    Use this WHENEVER analyzing any cryptocurrency (altcoin or BTC itself) to
    get the broader market frame in one shot. A SOL/ETH/whatever setup looks
    very different when BTC is dumping with rising dominance vs. when alts
    are leading. Calling this once gives Claude the macro context to provide
    Bitcoin-aware commentary alongside the per-coin analysis - without
    chaining 2-3 separate yahoo_price + manual reasoning calls.

    Returns:
      - bitcoin: price, 24h change %, volume, market cap
      - dominance: BTC and ETH market-cap share of total crypto
      - total_market: total crypto mcap + 24h change + active coin count
      - assessment: label (HIGH_RISK / ALT_RISK / ALT_FAVORABLE / OPPORTUNITY_WITH_CAUTION / NEUTRAL) + 1-paragraph reasoning
    """
    return get_bitcoin_market_pulse()


@mcp.tool(annotations=ToolAnnotations(title="Extended-Hours Stock Price", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_extended_hours(symbol: str) -> dict:
    """Real-time pre-market and after-hours prices for a US stock symbol.

    Use this when the user asks about a stock outside the regular 9:30am-4pm
    ET session — earnings reactions, overnight news, "what is X doing in
    after-hours?", "how did Y open in pre-market?". Returns the most recent
    valid print from each session window (pre-market, regular, post-market)
    along with computed % changes vs. the previous close and the regular
    close, respectively.

    During the regular session, post_market will be null (no data yet).
    On weekends/holidays, returns whatever's most recent in each window.

    Args:
        symbol: US stock symbol — AAPL, NVDA, TSLA, SPY, ^GSPC, etc.

    Returns:
        - pre_market: {price, as_of_utc, change_vs_previous_close_pct, reference_close} or null;
          the change is measured from reference_close, the last regular close
          (before the open that is newer than previous_close)
        - regular: {price, as_of_utc, change_pct} (consolidated tape close)
        - post_market: {price, as_of_utc, change_vs_regular_close_pct} or null
        - previous_close, currency, exchange, market_state for context
    """
    return await get_extended_hours_price_async(symbol)


@mcp.tool(annotations=ToolAnnotations(title="Options Chain", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def stock_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Full options chain (calls + puts) for a US stock symbol and one expiry.

    Use this when the user asks "what's the options chain for X?", "show me
    AAPL puts expiring next Friday", or wants to inspect bid/ask/IV/volume on
    a specific strike. If no expiry is provided, returns the nearest expiry
    so Claude can quote it back and ask "want a different one?".

    Args:
        symbol: US stock symbol — AAPL, NVDA, TSLA, SPY, etc.
        expiry: Optional ISO date (YYYY-MM-DD). Must match one of the
            `available_expiries` Yahoo returns; otherwise returns an error
            with the list of valid dates.

    Returns:
        - underlying_price, underlying_change_pct
        - requested_expiry, available_expiries (list of YYYY-MM-DD)
        - call_count, put_count
        - calls: list of {strike, last_price, bid, ask, volume,
          open_interest, implied_volatility, in_the_money, expiration}
        - puts: same shape as calls
    """
    try:
        return get_options_chain(symbol, expiry)
    except Exception as e:
        return exception_to_envelope(e, context="stock_options_chain")


@mcp.tool(annotations=ToolAnnotations(title="Unusual Options Activity", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def stock_options_unusual_activity(
    symbol: str,
    top_n: int = 10,
    min_volume: int = 100,
    expiries: int = 4,
) -> dict:
    """Top strikes by volume / open-interest ratio — institutional positioning signal.

    Use this when the user asks "any unusual options activity on X?", "where
    is the smart money positioned on NVDA before earnings?", or wants a
    V/OI screener for a ticker. A V/OI ratio > 1 means today's volume already
    exceeds standing open interest, which classically flags fresh institutional
    positioning on a specific strike in a specific direction (call vs put).

    Scans the soonest few expirations, filters out illiquid strikes (under
    `min_volume`), and returns the top-N sorted by V/OI descending. Also
    returns aggregate call vs put volume so Claude can comment on the
    overall directional bias.

    Args:
        symbol: US stock symbol — AAPL, NVDA, TSLA, SPY, META, etc.
        top_n: How many strikes to return. Default 10.
        min_volume: Filter floor for today's volume — prevents noise from
            illiquid strikes with high V/OI ratios. Default 100.
        expiries: Number of soonest expirations to scan. Default 4
            (typically covers ~1 month of weeklies + monthlies).

    Returns:
        - underlying_price
        - expiries_scanned (list of YYYY-MM-DD)
        - total_call_volume, total_put_volume, put_call_volume_ratio
        - unusual: list of top-N contracts sorted by V/OI desc, each with
          {strike, side (call|put), expiration, volume, open_interest,
          v_oi_ratio, last_price, implied_volatility, in_the_money,
          strike_vs_spot_pct (moneyness)}
    """
    try:
        return get_unusual_options_activity(symbol, top_n, min_volume, expiries)
    except Exception as e:
        return exception_to_envelope(e, context="stock_options_unusual_activity")


# ── Futures tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Futures Market Overview", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_market_overview(
    category: str = "all",
    exchanges: str = "us",
    limit: int = 30,
    volume_min: int = 0,
) -> dict:
    """Top futures contracts sorted by trading volume.

    Args:
        category:   all | equity_index | energy | metals | agriculture | rates | forex | crypto_futures
        exchanges:  us (CME, COMEX, NYMEX, CBOT) | global (adds ICE, EUREX)
        limit:      max contracts to return (default 30)
        volume_min: minimum volume filter (0 = no filter)

    Returns:
        Dict with total_available count and list of contracts with OHLCV + % change.
    """
    try:
        return get_futures_overview(
            category=category,
            exchanges=exchanges,
            limit=limit,
            volume_min=volume_min,
        )
    except Exception as exc:
        # NOTE: was ErrorCode.SERVICE_ERROR — a member that never existed, so
        # this handler itself raised AttributeError instead of returning.
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Futures overview failed: {exc}", retryable=True)


@mcp.tool(annotations=ToolAnnotations(title="Futures Top Movers", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_top_movers(
    direction: str = "gainers",
    exchanges: str = "us",
    limit: int = 20,
    volume_min: int = 10,
) -> dict:
    """Futures contracts with the biggest percentage moves today.

    Args:
        direction:  gainers | losers
        exchanges:  us | global
        limit:      max results
        volume_min: minimum volume filter (default 10, filters illiquid contracts)

    Returns:
        List of futures ranked by % change with OHLCV data.
    """
    direction = direction.lower()
    if direction not in ("gainers", "losers"):
        direction = "gainers"
    try:
        return get_futures_movers(
            direction=direction,
            exchanges=exchanges,
            limit=limit,
            volume_min=volume_min,
        )
    except Exception as exc:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Futures movers failed: {exc}", retryable=True)


@mcp.tool(annotations=ToolAnnotations(title="Futures Category Snapshot", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_category_snapshot(category: str = "energy") -> dict:
    """Quote all major front-month contracts in a specific futures category.

    Args:
        category: equity_index | energy | metals | agriculture | rates | forex | crypto_futures

    Returns:
        OHLCV quotes for the standard watchlist of contracts in that category.
        Example symbols: ES1! NQ1! (equity_index), CL1! NG1! (energy), GC1! SI1! (metals).
    """
    return get_futures_category_snapshot(category)


@mcp.tool(annotations=ToolAnnotations(title="Futures Watchlist", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_watchlist() -> dict:
    """Return the full categorized list of well-known front-month futures symbols.

    Categories: equity_index, energy, metals, agriculture, rates, forex, crypto_futures.
    Use these symbols with futures_category_snapshot or coin_analysis for deeper analysis.
    """
    return get_futures_watchlist()


@mcp.tool(annotations=ToolAnnotations(title="US Stock Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_screener(
    country: str = "america",
    stock_type: str = "common",
    limit: int = 50,
    exclude_otc: bool = True,
    compact: bool = False,
    sort_by: str = "market_cap",
) -> dict:
    """Screen stocks by share type — the API twin of TradingView's
    "Common stock" / "Preferred stock" symbol-search filter.

    Args:
        country: TradingView market name — e.g. america, korea, germany,
            brazil, japan, uk, india, turkey, canada, australia, france, hongkong
        stock_type: common | preferred
        limit: rows to return (max 2000, single upstream request), ranked by market cap descending
        exclude_otc: default True — drop OTC listings (foreign companies traded
            over-the-counter); "america" otherwise means "US venue", not "US company"
        compact: default False — True returns only ticker/symbol/price/currency/
            change_percent per row (light payload for bulk price feeds)
        sort_by: market_cap (default) | dividend_yield | change | price —
            server-side descending sort over the WHOLE market, so e.g.
            sort_by=dividend_yield with limit=20 is the market's true top-20
            dividend payers, not just the biggest companies re-sorted

    Returns:
        Envelope dict: total_matches (market-wide count), returned, and rows
        of {ticker, symbol, description, exchange, price, open, high, low,
        currency, change_percent, dividend_yield, market_cap} — price is the
        current/last close; open/high/low are the current session's daily bar. Prices are in the
        market's local currency (e.g. KRW for korea).
    """
    try:
        # tradingview-screener is sync (urllib) — off-load to a worker thread
        # so the event loop stays free for concurrent tool calls.
        return await asyncio.to_thread(
            screen_stocks, country, stock_type, limit, exclude_otc, compact, sort_by
        )
    except ValueError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, str(e))
    except Exception as e:
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"scan failed for market {country!r}: {e}",
            known_good_markets=list(EXAMPLE_MARKETS),
        )


@mcp.tool(annotations=ToolAnnotations(title="Multi-Symbol Stock Prices", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_prices(tickers: str) -> dict:
    """Current price + daily % change for specific stock symbols.

    Args:
        tickers: comma-separated EXCHANGE:SYMBOL list (max 2000 — one
            upstream request even at full size), e.g.
            "NASDAQ:NVDA, NASDAQ:TSLA, KRX:005930". The exchange prefix is
            required — the scanner's direct-ticker lookup is exchange-scoped.

    Returns:
        Envelope dict: rows of {ticker, symbol, description, exchange, price,
        open, high, low, currency, change_percent} plus a not_found list naming any requested
        ticker the scanner didn't recognize.
    """
    try:
        return await asyncio.to_thread(fetch_stock_prices, tickers)
    except ValueError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, str(e))
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"price lookup failed: {e}")


# ── Crypto swing-trading tools ─────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Crypto Trade Plan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def crypto_trade_plan(
    symbol: Annotated[
        str,
        Field(description="Pair symbol, e.g. BTCUSDT"),
    ],
    exchange: Annotated[
        str,
        Field(description="Venue: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, "
            "COINBASE, GATEIO"),
    ] = "KUCOIN",
    timeframe: Annotated[
        str,
        Field(description="One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (4h suits swing trades)"),
    ] = "4h",
    account_size: Annotated[
        Optional[float],
        Field(description="Account value in quote currency; omit to skip position "
            "sizing"),
    ] = None,
    risk_pct: Annotated[
        float,
        Field(description="Percent of the account risked on this trade, e.g. 1.0 for "
            "1%"),
    ] = 1.0,
) -> dict:
    """Entry, stop, targets and R:R for a crypto pair, with optional sizing.

    The crypto counterpart of egx_trade_plan. Entry/stop/targets come from the
    same shared setup builder, so they always describe one coherent trade
    (stop < entry < target_1 for longs) with per-scenario plans under
    ``scenarios``.

    Args:
        symbol: Pair symbol, e.g. BTCUSDT
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (4h suits swing trades)
        account_size: Account value in quote currency; omit to skip sizing
        risk_pct: Percent of the account risked on the trade (default 1.0)

    Returns:
        dict with trade_setup, position_sizing (when account_size is given)
        and an indicator context block. On ANY failure returns a structured
        error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        timeframe = validate_timeframe(timeframe, "4h")
        return await asyncio.to_thread(
            generate_crypto_trade_plan, symbol, exchange, timeframe, account_size, risk_pct
        )
    except Exception as e:
        return exception_to_envelope(e, context="crypto_trade_plan")


@mcp.tool(annotations=ToolAnnotations(title="Crypto Fibonacci Retracement", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def crypto_fibonacci_retracement(
    symbol: str,
    exchange: str = "KUCOIN",
    lookback: str = "52W",
    timeframe: str = "1D",
) -> dict:
    """Fibonacci retracement and extension levels for a crypto pair.

    Swing high/low come from the scanner's period-extreme columns (one
    request, no candle download); the level maths is shared with
    egx_fibonacci_retracement.

    Args:
        symbol: Pair symbol, e.g. BTCUSDT
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO
        lookback: Swing window — 1M, 3M, 6M, 52W or ALL
        timeframe: Interval for the RSI/EMA/ATR context (default 1D)

    Returns:
        dict with retracement_levels, extension_levels, price_position and
        context. On ANY failure returns a structured error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        timeframe = validate_timeframe(timeframe, "1D")
        return await asyncio.to_thread(
            analyze_crypto_fibonacci, symbol, exchange, lookback, timeframe
        )
    except Exception as e:
        return exception_to_envelope(e, context="crypto_fibonacci_retracement")


@mcp.tool(annotations=ToolAnnotations(title="Crypto Watchlist", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def crypto_watchlist(
    symbols: list[str],
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
) -> dict:
    """One scored row per symbol in a single scanner request.

    Use this instead of calling coin_analysis once per pair: a 20-pair
    watchlist costs one upstream request rather than twenty.

    Args:
        symbols: Pair symbols, e.g. ["BTCUSDT", "ETHUSDT", "SOLUSDT"] (max 30)
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M

    Returns:
        dict with rows (price, change_pct, rsi, rsi_zone, ema50, trend, atr,
        volume_ratio, rating) sorted by change, plus not_found for symbols the
        scanner doesn't know. On ANY failure returns a structured error
        envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        timeframe = validate_timeframe(timeframe, "4h")
        return await asyncio.to_thread(build_crypto_watchlist, symbols, exchange, timeframe)
    except Exception as e:
        return exception_to_envelope(e, context="crypto_watchlist")


@mcp.tool(annotations=ToolAnnotations(title="RSI Divergence Detector", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def rsi_divergence(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    lookback: int = 60,
    pivot_window: int = 3,
    rsi_period: int = 14,
) -> dict:
    """Bullish / bearish RSI divergence between the two most recent pivots.

    Bearish: price prints a higher pivot high while RSI prints a lower one.
    Bullish: price prints a lower pivot low while RSI does not. Pivots need
    ``pivot_window`` bars clear on both sides, so the forming bar is never
    treated as a pivot.

    Needs real candles, so the timeframe must be 1h, 4h or 1D (4h bars are
    resampled from 1h).

    Args:
        symbol: Pair symbol, e.g. BTCUSDT
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO
        timeframe: 1h, 4h or 1D
        lookback: Bars to search, 20-300 (default 60)
        pivot_window: Bars clear either side of a pivot, 1-10 (default 3)
        rsi_period: RSI period, 2-50 (default 14)

    Returns:
        dict with divergences (type, the two pivots compared, strength,
        bars_since) and divergence_found. On ANY failure returns a structured
        error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        return await asyncio.to_thread(
            detect_rsi_divergence, symbol, exchange, timeframe, lookback, pivot_window, rsi_period
        )
    except Exception as e:
        return exception_to_envelope(e, context="rsi_divergence")


@mcp.tool(annotations=ToolAnnotations(title="Supertrend Trend Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def supertrend_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    symbols: Optional[list[str]] = None,
    max_symbols: int = 15,
    flip_within: int = 3,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> dict:
    """True Supertrend direction, and recent flips, across crypto pairs.

    Supertrend is recursive — each bar's band depends on the previous bar's —
    so it is computed from real candles rather than a scanner column. Candles
    cost one request per symbol, so omitting ``symbols`` scans the exchange's
    most liquid names (ranked in one scanner request) rather than every pair;
    ``universe_note`` in the result says which set was used.

    Args:
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO
        timeframe: 1h, 4h or 1D
        symbols: Explicit pairs to scan; omit for the liquidity-ranked set
        max_symbols: Pairs to fetch candles for, 1-40 (default 15)
        flip_within: Flag flips within this many bars, 1-20 (default 3)
        atr_period: Supertrend ATR period, 2-50 (default 10)
        multiplier: Supertrend ATR multiplier, 0.5-10 (default 3.0)

    Returns:
        dict with flips (the just-flipped subset) and rows (direction,
        supertrend_line, distance_pct, atr_distance, bars_since_flip). On ANY
        failure returns a structured error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        return await asyncio.to_thread(
            scan_supertrend, exchange, timeframe, symbols, max_symbols,
            flip_within, atr_period, multiplier,
        )
    except Exception as e:
        return exception_to_envelope(e, context="supertrend_scan")


@mcp.tool(annotations=ToolAnnotations(title="Donchian Breakout Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def donchian_breakout_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    symbols: Optional[list[str]] = None,
    period: int = 20,
    max_symbols: int = 15,
) -> dict:
    """Donchian channel breakouts across crypto pairs.

    The channel is measured on the bars BEFORE the one being tested, so a
    close breaking out is compared against the prior window's extreme — not
    against a channel that already contains it.

    Candles cost one request per symbol, so omitting ``symbols`` scans the
    exchange's most liquid names; ``universe_note`` says which set was used.

    Args:
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO
        timeframe: 1h, 4h or 1D
        symbols: Explicit pairs to scan; omit for the liquidity-ranked set
        period: Channel lookback in bars, 5-100 (default 20)
        max_symbols: Pairs to fetch candles for, 1-40 (default 15)

    Returns:
        dict with breakouts (the breaking-out subset) and rows (state,
        channel_upper/lower/middle, channel_width_pct,
        position_in_channel_pct). On ANY failure returns a structured error
        envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        return await asyncio.to_thread(
            scan_donchian_breakout, exchange, timeframe, symbols, period, max_symbols
        )
    except Exception as e:
        return exception_to_envelope(e, context="donchian_breakout_scan")


# ── Market structure ───────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Market Breadth", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def market_breadth(
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    min_rel_volume: Optional[float] = None,
) -> dict:
    """How much of an exchange is participating — one scanner request.

    Answers what no per-symbol tool can: whether a setup runs with the venue
    or against it. A bullish chart where 18% of names are above their EMA50 is
    a different proposition from the same chart at 72%.

    Every percentage is computed only over symbols the scanner had that input
    for, and the ``coverage`` block reports those denominators — a missing
    EMA50 is excluded, never counted as "below".

    Args:
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, NASDAQ, NYSE, EGX, BIST, ...
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        min_rel_volume: Drop symbols below this relative volume (illiquid
            listings otherwise dominate a crypto venue's counts)

    Returns:
        dict with trend_participation (% above EMA50 / SMA200),
        advance_decline, momentum_distribution and an assessment label. On ANY
        failure returns a structured error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        timeframe = validate_timeframe(timeframe, "1D")
        return await asyncio.to_thread(
            analyze_market_breadth, exchange, timeframe, min_rel_volume
        )
    except Exception as e:
        return exception_to_envelope(e, context="market_breadth")


@mcp.tool(annotations=ToolAnnotations(title="Relative Strength Ranking", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def relative_strength(
    exchange: str = "KUCOIN",
    window: str = "1M",
    benchmark: str = "BTCUSDT",
    limit: int = 15,
    min_rel_volume: Optional[float] = None,
) -> dict:
    """Rank a venue by performance against a benchmark over weeks, not minutes.

    top_gainers ranks by the current bar's change, which answers a different
    question: what is moving now, not what has been leading. This ranks on
    period-performance and subtracts the benchmark's own move, so an alt up 4%
    while BTC is up 9% reads as the laggard it is.

    Performance columns exist only on the daily timeframe, so this always
    queries 1D and says so in the result.

    Args:
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, NASDAQ, NYSE, EGX, BIST, ...
        window: 1W, 1M, 3M or 1Y
        benchmark: Symbol to measure against (BTCUSDT by default)
        limit: Leaders and laggards to return each, 1-50
        min_rel_volume: Drop symbols below this relative volume

    Returns:
        dict with leaders, laggards, the benchmark's own performance and
        out/underperforming counts. On ANY failure returns a structured error
        envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        return await asyncio.to_thread(
            rank_relative_strength, exchange, window, benchmark, limit, min_rel_volume
        )
    except Exception as e:
        return exception_to_envelope(e, context="relative_strength")


@mcp.tool(annotations=ToolAnnotations(title="Perpetual Positioning", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def positioning(
    symbol: str,
    period: str = "4h",
    history_bars: int = 30,
) -> dict:
    """Who is positioned where on a perpetual, and what it costs them.

    The only tool here that does not read price or volume. Combines seven
    public Binance futures feeds: funding (current, annualised and recent
    trend), open interest (level, notional, change), the global long/short
    account split, the top traders' split, and aggressive taker flow.

    The crowd-versus-top-traders comparison is the point: a market where 75%
    of accounts are long while the largest traders sit at 68% is positioned
    differently from one where both agree, and nothing derived from price can
    tell those apart.

    Needs a Binance PERPETUAL contract. Most pairs are spot-only and return
    SYMBOL_NOT_FOUND with suggestions. No API key required.

    Args:
        symbol: Pair symbol, e.g. BTCUSDT
        period: Sampling interval for the ratio and OI histories —
            5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h or 1D
        history_bars: History points to fetch, 2-500 (default 30)

    Returns:
        dict with funding, open_interest, crowd, top_traders,
        crowd_vs_top_divergence_pts, taker_flow, an oi_price_read quadrant and
        an assessment carrying squeeze_risk plus the thresholds behind it. On
        ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(analyze_positioning, symbol, period, history_bars)
    except Exception as e:
        return exception_to_envelope(e, context="positioning")


@mcp.tool(annotations=ToolAnnotations(title="Lead Traders", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def lead_traders(
    sort: str = "track_record",
    limit: int = 10,
    instrument_type: str = "SWAP",
    min_lead_days: Optional[int] = None,
    max_avg_leverage: Optional[float] = None,
    history_limit: int = 20,
) -> dict:
    """Copy-trading lead traders from OKX, with the risk the leaderboard hides.

    The leaderboard itself publishes return, win rate, followers and AUM — and
    no leverage, no drawdown, no Sharpe. Ranked on return alone it is a list
    of whoever ran the most leverage and survived. So this fetches each
    candidate's closed trades and weekly PnL and measures what is missing:
    average and peak leverage, the worst single trade, losing weeks and a
    drawdown floor, then attaches risk_flags.

    Default sort is track_record, not return, deliberately.

    Reports published statistics only — it places, follows and sizes nothing,
    and is not a recommendation.

    Args:
        sort: "track_record", "return", "lowest_leverage", "win_rate" or
            "followers"
        limit: Traders to return, 1-15 (each costs two extra requests)
        instrument_type: "SWAP" or "SPOT"
        min_lead_days: Drop traders with a shorter track record
        max_avg_leverage: Drop traders whose measured average leverage is above
            this
        history_limit: Closed trades to sample per trader, 5-100

    Returns:
        dict with traders carrying advertised vs measured figures, weekly
        stats, risk_flags, and a what_this_cannot_tell_you block. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            list_lead_traders, sort, limit, instrument_type,
            min_lead_days, max_avg_leverage, history_limit,
        )
    except Exception as e:
        return exception_to_envelope(e, context="lead_traders")


@mcp.tool(annotations=ToolAnnotations(title="Lead Trader Profile", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def lead_trader_profile(
    unique_code: str,
    instrument_type: str = "SWAP",
    history_limit: int = 50,
) -> dict:
    """Full risk profile for one lead trader, including live open positions.

    Shows what they are holding right now with the leverage on each position,
    alongside measured trade statistics and instrument preferences.

    Reports published statistics only — it places, follows and sizes nothing,
    and is not a recommendation.

    Args:
        unique_code: The trader's uniqueCode from lead_traders
        instrument_type: "SWAP" or "SPOT"
        history_limit: Closed trades to sample, 5-100

    Returns:
        dict with advertised and measured figures, open_positions carrying
        per-position leverage, weekly stats and risk_flags. On ANY failure
        returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            get_lead_trader, unique_code, instrument_type, history_limit
        )
    except Exception as e:
        return exception_to_envelope(e, context="lead_trader_profile")


@mcp.tool(annotations=ToolAnnotations(title="Lead Trader Trade History", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def lead_trader_trades(
    unique_code: str,
    instrument_type: str = "SWAP",
    max_trades: int = 300,
    instrument: Optional[str] = None,
    include_trades: bool = True,
    include_equity_curve: bool = False,
    trade_rows: int = 40,
) -> dict:
    """Every closed trade for one lead trader, with full detail and statistics.

    Statistics always cover every trade fetched (max_trades). The rows
    themselves are capped at trade_rows, newest first, because 300 rows
    measured 95-150 KB in one answer; raise trade_rows when the
    individual trades are what is wanted.

    Pages back through the whole history (OKX caps a page at 100), so this is
    the complete log rather than a recent slice, and reports
    complete_history so you know which you got.

    Per trade: instrument, side, leverage, margin, size, open and close price,
    open and close time, hours held, PnL and PnL as a percent of margin.

    Computed over the set: win rate, profit factor, expectancy per trade,
    average win vs average loss, hold times, per-instrument and per-side
    breakdowns, and a realised drawdown sequenced by close time — which
    catches declines the weekly figure in lead_trader_profile cannot see.

    Note that pnl_pct_of_margin is return on MARGIN, not notional: -262 means
    the position lost 2.62x the margin committed, which leverage allows.

    Published statistics only — places, follows and sizes nothing, and is not
    a recommendation.

    Args:
        unique_code: The trader's uniqueCode from lead_traders
        instrument_type: "SWAP" only. OKX publishes trade history for futures
            lead traders alone; SPOT is rejected with an explanation
        max_trades: Upper bound on trades fetched, 10-2000 (each 100 is one
            request)
        instrument: Keep only this instrument, e.g. "BTC-USDT-SWAP"
        include_trades: Return the individual rows; false for statistics only
        include_equity_curve: Return cumulative realised PnL per trade

    Returns:
        dict with trades, statistics, realised_drawdown, by_instrument,
        by_side and field_notes. On ANY failure returns a structured error
        envelope.
    """
    try:
        return await asyncio.to_thread(
            get_lead_trader_trades, unique_code, instrument_type, max_trades,
            instrument, include_trades, include_equity_curve, trade_rows,
        )
    except Exception as e:
        return exception_to_envelope(e, context="lead_trader_trades")


@mcp.tool(annotations=ToolAnnotations(title="Fear and Greed Index", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def fear_greed(history_days: int = 30) -> dict:
    """The crypto Fear & Greed index, with history and direction of travel.

    Needs NO API key — unlike market_sentiment and financial_news, which
    return "Unavailable" until MARKETAUX_API_TOKEN is configured.

    A level alone says little: 32 after 65 is a different market from 32 after
    20, so the result carries change_over_window and a direction.

    Args:
        history_days: Days of history to include, 1-365

    Returns:
        dict with current reading, range, mean, direction and the series. On
        ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(get_fear_greed, history_days)
    except Exception as e:
        return exception_to_envelope(e, context="fear_greed")


@mcp.tool(annotations=ToolAnnotations(title="Narrative Rotation", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def narrative_rotation(
    min_market_cap_usd: Optional[float] = 100000000.0,
    limit: int = 15,
    order: str = "gainers",
) -> dict:
    """Which crypto sectors are moving — the crypto counterpart of egx_sector_scan.

    The market-cap floor is ON by default and it matters: CoinGecko lists 778
    categories, and sorting the raw list by 24h change opens with dust like
    "Decentralized Lottery" up 1331% on a market cap that rounds to $0.0B.
    Pass null only if you want the dust.

    Args:
        min_market_cap_usd: Category market-cap floor; null disables it
        limit: Rows to return, 1-50
        order: "gainers" or "losers" first

    Returns:
        dict with ranked categories, their top coins, and advancing/declining
        counts. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            get_narrative_rotation, min_market_cap_usd, limit, order
        )
    except Exception as e:
        return exception_to_envelope(e, context="narrative_rotation")


@mcp.tool(annotations=ToolAnnotations(title="BTC Correlation and Beta", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def correlation(
    symbols: list[str],
    benchmark: str = "BTCUSDT",
    timeframe: str = "4h",
    lookback: int = 180,
) -> dict:
    """Is this its own trade, or leveraged BTC?

    The macro tools keep reporting ALT_RISK and high BTC dominance; this makes
    the implication concrete per symbol. An alt with beta 1.6 and r-squared
    0.9 is not a diversifying position — it is a leveraged bet on the
    benchmark, and five of them is one trade in five tickets.

    Returns are paired by bar timestamp, not by position, so a gap or a
    different history start cannot mis-pair two symbols.

    Args:
        symbols: Pairs to measure, max 20
        benchmark: Pair to measure against (BTCUSDT by default)
        timeframe: Candle interval
        lookback: Bars to use, 40-1000

    Returns:
        dict with correlation, r_squared, beta, observation count and a plain
        reading per symbol. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            analyze_correlation, symbols, benchmark, timeframe, lookback
        )
    except Exception as e:
        return exception_to_envelope(e, context="correlation")


@mcp.tool(annotations=ToolAnnotations(title="Seasonality", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def seasonality(
    symbol: str,
    timeframe: str = "4h",
    lookback: int = 500,
) -> dict:
    """Average return by weekday and by hour of day (UTC).

    Crypto trades 24/7, so an hour-of-day distribution genuinely exists —
    unlike equities, where it would just describe the session.

    Every bucket carries its sample count and a thin_sample flag, and thin
    buckets are excluded from best/worst: a weekday with six observations is
    not evidence.

    Args:
        symbol: Pair to analyse
        timeframe: Candle interval — hourly buckets need an intraday interval
        lookback: Bars to use, 50-1500

    Returns:
        dict with by_weekday, by_hour_utc, and best/worst buckets. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            analyze_seasonality, symbol, timeframe, lookback
        )
    except Exception as e:
        return exception_to_envelope(e, context="seasonality")


@mcp.tool(annotations=ToolAnnotations(title="Trend Strength Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def trend_strength_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "4h",
    direction: str = "both",
    min_adx: float = 25.0,
    limit: int = 20,
    min_rel_volume: Optional[float] = None,
) -> dict:
    """Rank a venue by trend strength rather than by today's move.

    top_gainers answers "what moved". This answers "what is actually
    trending": a symbol up 9% on one candle with ADX 14 is noise, one up 3%
    with ADX 38 and +DI above -DI is a trend. coin_analysis shows these for
    one symbol; nothing screened on them. One scanner request.

    Args:
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, NASDAQ, NYSE, EGX, BIST, ...
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        direction: "both", "bullish" (+DI > -DI) or "bearish"
        min_adx: Minimum ADX, 0-100 — below ~25 is conventionally chop
        limit: Rows to return, 1-100
        min_rel_volume: Drop symbols below this relative volume

    Returns:
        dict with rows carrying adx, trend_strength, +DI/-DI and their spread,
        the Parabolic SAR side and EMA50 position. On ANY failure returns a
        structured error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        timeframe = validate_timeframe(timeframe, "4h")
        return await asyncio.to_thread(
            scan_trend_strength, exchange, timeframe, direction, min_adx,
            limit, min_rel_volume,
        )
    except Exception as e:
        return exception_to_envelope(e, context="trend_strength_scan")


@mcp.tool(annotations=ToolAnnotations(title="Volatility Ranking", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def volatility_rank(
    exchange: str = "KUCOIN",
    window: str = "1D",
    order: str = "highest",
    limit: int = 20,
    min_rel_volume: Optional[float] = None,
) -> dict:
    """Rank a venue by realised volatility, with ATR as a price-relative check.

    Position size follows volatility: the same percentage stop is a different
    trade on a 2%-a-day symbol than on a 15%-a-day one. Gives the ATR sizing
    in crypto_trade_plan the venue-wide context it otherwise lacks.

    Volatility columns exist only on the daily timeframe, so this always
    queries 1D and says so.

    Args:
        exchange: KUCOIN, BINANCE, BYBIT, MEXC, NASDAQ, NYSE, EGX, BIST, ...
        window: 1D, 1W or 1M
        order: "highest" or "lowest" volatility first
        limit: Rows to return, 1-100
        min_rel_volume: Drop symbols below this relative volume

    Returns:
        dict with ranked rows (volatility, ATR, ATR as a percent of price) and
        the venue median. On ANY failure returns a structured error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        return await asyncio.to_thread(
            rank_volatility, exchange, window, order, limit, min_rel_volume
        )
    except Exception as e:
        return exception_to_envelope(e, context="volatility_rank")


@mcp.tool(annotations=ToolAnnotations(title="Funding Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def funding_screener(
    limit: int = 20,
    direction: str = "both",
    min_quote_volume_usd: Optional[float] = 10000000.0,
) -> dict:
    """Rank the whole perpetual board by cost of carry — three requests.

    Binance returns every perp's funding in one call, so this costs the same
    whether the board has 10 contracts or 950.

    The funding interval is per symbol and it matters: 820 of ~945 perps
    settle every 4 hours, not 8, so annualising uniformly at 8h understates
    the majority of the market.

    Args:
        limit: Rows per side, 1-100
        direction: "both", "longs_pay" (most positive) or "shorts_pay"
        min_quote_volume_usd: Liquidity floor — a 300%/yr rate on a market
            with no volume is noise. Pass null to disable

    Returns:
        dict with longs_pay_most / shorts_pay_most and a board summary. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            screen_funding, limit, direction, min_quote_volume_usd
        )
    except Exception as e:
        return exception_to_envelope(e, context="funding_screener")


@mcp.tool(annotations=ToolAnnotations(title="Squeeze Risk Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def squeeze_scan(
    limit: int = 12,
    candidates: int = 30,
    period: str = "4h",
    min_quote_volume_usd: Optional[float] = 50000000.0,
) -> dict:
    """Rank the perpetual board by liquidation-cascade risk.

    Where `positioning` answers "is the crowd exposed on BTC?", this answers
    "where is the crowd exposed at all".

    Two-stage, because crowding and open-interest history are per-symbol only:
    stage one narrows the board on bulk funding where funding runs AGAINST the
    price move, stage two measures crowding for the shortlist and grades it
    with the same logic the single-symbol tool uses — so the two cannot
    disagree.

    Args:
        limit: Rows returned, 1-30
        candidates: Contracts to fetch detail for, 1-30 — each costs three
            requests, so this is the cost knob
        period: Sampling interval for crowding and OI history
        min_quote_volume_usd: Liquidity floor for stage one

    Returns:
        dict with elevated (the ELEVATED subset), ranked rows carrying
        factors, crowding, funding and OI state, plus the shortlist rule and
        thresholds. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            scan_squeeze_risk, limit, candidates, period, min_quote_volume_usd
        )
    except Exception as e:
        return exception_to_envelope(e, context="squeeze_scan")


@mcp.tool(annotations=ToolAnnotations(title="Market Brief", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def market_brief(
    exchange: str = "KUCOIN",
    symbol: str = "BTCUSDT",
    include_wallet: bool = True,
    include_alerts: bool = True,
) -> dict:
    """The market in one call — the answer to "how's the market?".

    Replaces six round trips: breadth, the bitcoin pulse, perp positioning,
    sentiment, your standing alerts and your paper wallet. They run
    concurrently and come back as one object.

    A section that fails says so and the brief still returns, because losing
    the whole answer when one feed times out would be worse than the separate
    calls this replaces. Check `unavailable` and `attention` before reading
    the healthy sections.

    It reports; it does not advise. Each section carries its own assessment;
    reading across them is your job.

    Args:
        exchange: Venue for the breadth read
        symbol: Perpetual for the positioning read
        include_wallet: Add the paper wallet's standing
        include_alerts: Evaluate standing alerts and report what fired

    Returns:
        dict with headline, sections, attention (things a glance should not
        miss), fired_alerts and unavailable. On total failure returns a
        structured error envelope.
    """
    try:
        exchange = validate_exchange(exchange, "kucoin")
        return await asyncio.to_thread(
            _market_brief, exchange, symbol, include_wallet, include_alerts
        )
    except Exception as e:
        return exception_to_envelope(e, context="market_brief")


@mcp.tool(annotations=ToolAnnotations(title="Get Started", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def get_started() -> dict:
    """What to do next, based on what this installation actually has.

    Call this when someone is new to this server, asks what it can do, or
    seems unsure where to begin. It checks real state — is there a practice
    wallet, are there alerts, is the Marketaux key set, is the data layer
    answering — and returns an ordered list of next steps.

    It is not a feature tour. A list of every tool is what the README is for,
    and whoever needs this is the person who did not read one.

    Returns:
        dict with state, next_steps and a few things worth asking. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_get_started)
    except Exception as e:
        return exception_to_envelope(e, context="get_started")


@mcp.tool(annotations=ToolAnnotations(title="Alert Schedule Recipe", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
async def alert_schedule(every_minutes: int = 15) -> dict:
    """How to make standing alerts actually fire, via a scheduled task.

    Alerts are stored and evaluated on demand, so nothing fires them on its
    own and a condition met overnight is noticed in the morning. MCP gives a
    server no way to wake anyone, so this returns the exact prompt and
    cadence to hand a scheduler — the honest answer rather than pretending
    the server can push.

    Read the caveats: a check reads the price AT CHECK TIME, so a condition
    met and unmet between two checks is missed entirely.

    Args:
        every_minutes: How often to check, 5-1440

    Returns:
        dict with the schedule, the task prompt to use, how to set it up and
        what it cannot do. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_alert_schedule, every_minutes)
    except Exception as e:
        return exception_to_envelope(e, context="alert_schedule")


@mcp.tool(annotations=ToolAnnotations(title="Crypto Bubbles", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def crypto_bubbles(
    limit: Annotated[
        int,
        Field(description="How many coins by market cap, 1-100. 10, 30 and "
            "50 are the usual views"),
    ] = 30,
    window: Annotated[
        str,
        Field(description="Which change to lead with: 1h, 24h, 7d, 30d or "
            "1y. Every window comes back regardless"),
    ] = "24h",
    include_stablecoins: Annotated[
        bool,
        Field(description="Keep the coins built not to move. Off by default, "
            "since they take the biggest bubbles to say nothing"),
    ] = False,
) -> dict:
    """The top of the market by size, with how far each coin has moved.

    The data behind a bubble chart: rank, market cap, price, volume and the
    change over 1h, 24h, 7d, 30d and 1y for every coin in one call — so a
    rendered view can switch window without asking again.

    Answers in one glance what a table does not: whether a move is broad or
    a few names, and whether money is leaving the large caps or the small
    ones. Breadth comes with it — how many up, how many down, the median,
    and the extremes.

    change_pct is PRICE change, not market-cap change: a coin that unlocked
    supply can fall in price while its cap holds. Ranking and caps are
    CoinGecko's own.

    Returns:
        dict with coins, breadth, the total cap of the set and which
        stablecoins were removed. On ANY failure returns a structured error
        envelope.
    """
    try:
        return await asyncio.to_thread(
            _crypto_bubbles, limit, window, include_stablecoins
        )
    except Exception as e:
        return exception_to_envelope(e, context="crypto_bubbles")


# -- Paper trading -------------------------------------------------------------

@mcp.tool(annotations=ToolAnnotations(title="Copy Trader To Paper Wallet", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
async def paper_copy_trader(
    unique_code: Annotated[
        str,
        Field(description="The trader's uniqueCode from lead_traders"),
    ],
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet to copy into. A separate one is "
            "worth using, so a copied strategy keeps its own record"),
    ] = None,
    margin_pct: Annotated[
        float,
        Field(description="Margin per copied position, as a percent of the "
            "wallet's starting balance. Default 2"),
    ] = 2.0,
    max_positions: Annotated[
        int,
        Field(description="Most positions to copy, 1-10"),
    ] = 5,
    instrument_type: Annotated[
        str,
        Field(description='"SWAP" or "SPOT"'),
    ] = "SWAP",
    confirm: Annotated[
        bool,
        Field(description="False previews the plan and writes nothing. True "
            "opens the positions"),
    ] = False,
) -> dict:
    """Open what a lead trader is holding, in a simulated wallet.

    SIMULATED: nothing reaches an exchange, and no money moves.

    Previews by default — opening several positions from one sentence should
    be read first, so nothing is written until confirm=true.

    This is NOT copy trading, and the difference is the whole risk. You enter
    at today's price, not theirs: a leaderboard shows trades already running,
    so a position most of the way to its target is one where you buy what is
    left of the move while carrying the full distance to the stop. Every row
    reports already_moved_pct, and the result leads with the furthest along.

    Copied positions have NO STOP, because OKX publishes what a trader holds
    and never where they would get out. Set one per position afterwards, or
    decide deliberately not to.

    Sizing is yours, not theirs: each position takes margin_pct of this
    wallet's starting balance, at their leverage capped to the wallet's
    maximum. A capped position is a different trade from theirs, not a
    smaller one, and it says so.

    Returns:
        dict with the plan, what was skipped and why, the cash it needs and
        what was opened. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            _copy_trader, unique_code, wallet, margin_pct, max_positions,
            instrument_type, confirm,
        )
    except Exception as e:
        return exception_to_envelope(e, context="paper_copy_trader")



@mcp.tool(annotations=ToolAnnotations(title="List Paper Wallets", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
async def paper_wallet_list() -> dict:
    """Every simulated wallet that exists, by name.

    Each wallet is a separate balance with its own positions, resting orders
    and history — one for the strategy being traded, another for the idea
    being tested, so a high-leverage experiment cannot contaminate the record
    of what actually works.

    Names are separation, not privacy: anything that can reach this server
    can name any wallet.

    Returns:
        dict with the wallet names and which one is the default. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_paper_wallets)
    except Exception as e:
        return exception_to_envelope(e, context="paper_wallet_list")



@mcp.tool(annotations=ToolAnnotations(title="Place Paper Order", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
async def paper_order_place(
    symbol: Annotated[
        str,
        Field(description="Pair, e.g. BTCUSDT"),
    ],
    side: Annotated[
        str,
        Field(description="\"long\" to rest a buy below the market, \"short\" "
            "to rest a sell above it"),
    ],
    notional: Annotated[
        float,
        Field(description="Position value in the wallet's currency if it "
            "fills. Margin is notional / leverage, and it is held from now"),
    ],
    limit_price: Annotated[
        float,
        Field(description="The level to wait at. A buy must sit below the "
            "current price and a sell above it"),
    ],
    market: Annotated[
        str,
        Field(description="\"futures\" or \"spot\" (spot is long-only, "
            "unlevered)"),
    ] = "futures",
    leverage: Annotated[
        float,
        Field(description="1 to the wallet's maximum; futures only"),
    ] = 1.0,
    stop_loss: Annotated[
        Optional[float],
        Field(description="Exit if it goes wrong, checked against the LIMIT "
            "price rather than today's price"),
    ] = None,
    take_profit: Annotated[
        Optional[float],
        Field(description="Exit if it goes right, checked against the LIMIT "
            "price"),
    ] = None,
    good_for_hours: Annotated[
        Optional[float],
        Field(description="Cancel it automatically after this long. Omit and "
            "it rests until cancelled"),
    ] = None,
    note: Annotated[
        Optional[str],
        Field(description="Why this level, so the history is reviewable"),
    ] = None,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Leave a simulated order resting at a level instead of buying now.

    SIMULATED: nothing reaches an exchange.

    This is the tool for "buy at support" and "sell at resistance". Nothing
    watches the market — the next time the wallet is checked, the bars since
    you placed it are scanned, and if one reached your level the order fills
    at that bar's time, with its stop and target settling from there. An
    order placed Monday and checked Friday resolves Monday's fill correctly.

    Margin is held the moment it is placed, as an exchange holds it, so two
    orders cannot both fill on the same cash.

    A fill at a touched level is a BEST CASE — in a real book you queue
    behind whoever is already resting there, and a level merely tagged often
    fills nobody.

    An order that would fill instantly is refused: use paper_trade_open to
    trade at the current price.

    Returns:
        dict with the resting order, the distance left to run, the risk if
        it fills, and how filling works. On ANY failure returns a structured
        error envelope.
    """
    try:
        return await asyncio.to_thread(
            _place_order, symbol, side, notional, limit_price, market,
            leverage, stop_loss, take_profit, good_for_hours, note,
            wallet,
        )
    except Exception as e:
        return exception_to_envelope(e, context="paper_order_place")


@mcp.tool(annotations=ToolAnnotations(title="List Paper Orders", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def paper_order_list(
    status: Annotated[
        Optional[str],
        Field(description="\"pending\", \"filled\", \"cancelled\", "
            "\"expired\", or omit for all"),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Orders to return, 1-500"),
    ] = 50,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Resting simulated orders, and what became of the others.

    Settles first, so an order reached since the last look is already shown
    as filled rather than still waiting. Pending orders carry the distance
    left to run, and the total margin they are holding — cash that has left
    `cash` without yet being in any position.

    Returns:
        dict with orders, counts, the margin they hold and the nearest one
        to filling. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_list_orders, status, limit, wallet)
    except Exception as e:
        return exception_to_envelope(e, context="paper_order_list")


@mcp.tool(annotations=ToolAnnotations(title="Cancel Paper Order", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
async def paper_order_cancel(
    order_id: Annotated[
        int,
        Field(description="From paper_order_place or paper_order_list"),
    ],
    note: Annotated[
        Optional[str],
        Field(description="Why it was pulled"),
    ] = None,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Take a resting simulated order off the book and release its margin.

    Only a pending order can be cancelled. One that already filled is a
    position now — close that with paper_trade_close.

    Returns:
        dict with the cancelled order and the margin released. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_cancel_order, order_id, note, wallet)
    except Exception as e:
        return exception_to_envelope(e, context="paper_order_cancel")


#
# A SIMULATOR. This package holds no exchange credentials and calls no
# authenticated endpoint, so these tools cannot place a real order or move real
# money however they are asked. A position is a row in a local SQLite file,
# priced against live market data.
#
# They write state, so their annotations say so and they are named in
# test_tool_annotations' stateful allowlist.

@mcp.tool(annotations=ToolAnnotations(title="Reset Paper Wallet", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
async def paper_wallet_reset(
    balance: Annotated[
        float,
        Field(description="Starting balance, 10 to 100,000,000. Ask the person for "
            "it; it silently defines what every later percentage means"),
    ],
    currency: Annotated[
        str,
        Field(description="Quote currency label, e.g. USDT"),
    ] = "USDT",
    fee_pct: Annotated[
        float,
        Field(description="Taker fee charged per side, as a PERCENT: 0.05 means "
            "0.05%, Binance-like, not 5%"),
    ] = 0.05,
    max_leverage: Annotated[
        float,
        Field(description="Highest leverage this wallet will allow, 1 to 125"),
    ] = 20.0,
    note: Annotated[
        Optional[str],
        Field(description="Free text, e.g. which strategy this wallet is testing"),
    ] = None,
    confirm_wipe: Annotated[
        bool,
        Field(description="Must be true to destroy an existing wallet and its "
            "positions"),
    ] = False,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Create a simulated wallet, or wipe it and start again.

    SIMULATED: no real order, no real money, no exchange key. The wallet is a
    local database marked against live prices.

    ASK THE PERSON for the starting balance before calling this; do not pick
    one for them. It is deliberately required, because it silently defines
    every later number — a 2% risk rule is a different trade on 500 than on
    100,000. If they genuinely do not mind, say which figure you used.

    Resetting over a wallet that already holds positions needs
    confirm_wipe=true, and that confirmation is the person's to give, not
    yours: a practice history is the record of what their strategy actually
    did, and deleting it cannot be undone.

    Args:
        balance: Starting balance, 10 to 100,000,000 — ask, do not assume
        currency: Quote currency label, e.g. USDT
        fee_pct: Taker fee charged per side, percent (0.05 is Binance-like)
        max_leverage: Ceiling this wallet will allow, 1-125
        note: Free text, e.g. what strategy this wallet is testing
        confirm_wipe: Required to destroy an existing wallet's positions

    Returns:
        dict with the new wallet and how many positions were discarded. On ANY
        failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            _paper_reset, balance, currency, fee_pct, max_leverage, note,
            confirm_wipe, wallet,
        )
    except Exception as e:
        return exception_to_envelope(e, context="paper_wallet_reset")


@mcp.tool(annotations=ToolAnnotations(title="Open Paper Trade", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
async def paper_trade_open(
    symbol: Annotated[
        str,
        Field(description="Pair, e.g. BTCUSDT"),
    ],
    side: Annotated[
        str,
        Field(description="\"long\" or \"short\" (short requires market=\"futures\")"),
    ],
    notional: Annotated[
        float,
        Field(description="Position value in the wallet's currency, e.g. 500 for a "
            "$500 position. Margin posted is notional / leverage"),
    ],
    market: Annotated[
        str,
        Field(description="\"futures\" (shorts, leverage, funding, liquidation) or "
            "\"spot\" (long-only, unlevered)"),
    ] = "futures",
    leverage: Annotated[
        float,
        Field(description="1 to the wallet's maximum; futures only"),
    ] = 1.0,
    stop_loss: Annotated[
        Optional[float],
        Field(description="Price to exit at a loss - below entry for a long, above "
            "for a short"),
    ] = None,
    take_profit: Annotated[
        Optional[float],
        Field(description="Price to exit at a gain - above entry for a long, below "
            "for a short"),
    ] = None,
    note: Annotated[
        Optional[str],
        Field(description="Why this trade, so the history is reviewable later"),
    ] = None,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Open a simulated position at the current real market price.

    SIMULATED: nothing is sent to an exchange.

    Spot is long-only and unlevered. Futures allow shorts and leverage, post
    margin, pay real funding and can be liquidated.

    A stop or target on the wrong side of entry is refused rather than filled
    on the next check - that is a typo, not a strategy.

    Args:
        symbol: Pair, e.g. BTCUSDT
        side: "long" or "short" (short requires market="futures")
        notional: Position value in the wallet's currency; margin is
            notional / leverage
        market: "futures" or "spot"
        leverage: 1 to the wallet's max (futures only)
        stop_loss: Price to exit at a loss - below entry for a long
        take_profit: Price to exit at a gain - above entry for a long
        note: Why this trade, so the history is reviewable later

    Returns:
        dict with the opened position, a risk block (loss at stop, percent of
        wallet, R:R) and the liquidation price. On ANY failure returns a
        structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            _paper_open, symbol, side, notional, market, leverage,
            stop_loss, take_profit, note, wallet,
        )
    except Exception as e:
        return exception_to_envelope(e, context="paper_trade_open")


@mcp.tool(annotations=ToolAnnotations(title="Close Paper Trade", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
async def paper_trade_close(
    position_id: Annotated[
        int,
        Field(description="From paper_trade_open or paper_wallet_status"),
    ],
    note: Annotated[
        Optional[str],
        Field(description="Why it was closed"),
    ] = None,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Close a simulated position at the current real market price.

    Settles first: if the stop, target or liquidation price was already
    reached, the recorded exit is that level - not the current price.

    Args:
        position_id: From paper_trade_open or paper_wallet_status
        note: Why it was closed

    Returns:
        dict with entry and exit, fees, funding paid, realised PnL and return
        on margin. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_paper_close, position_id, note, wallet)
    except Exception as e:
        return exception_to_envelope(e, context="paper_trade_close")


@mcp.tool(annotations=ToolAnnotations(title="Modify Paper Trade", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True))
async def paper_trade_modify(
    position_id: Annotated[
        int,
        Field(description="From paper_trade_open or paper_wallet_status"),
    ],
    stop_loss: Annotated[
        Optional[float],
        Field(description="New stop price; omit to leave it unchanged"),
    ] = None,
    take_profit: Annotated[
        Optional[float],
        Field(description="New target price; omit to leave it unchanged"),
    ] = None,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Set or move the stop and target on an open simulated position.

    Useful for trailing a stop as a trade works. A level that would fill
    immediately against the current price is refused.

    Args:
        position_id: From paper_trade_open or paper_wallet_status
        stop_loss: New stop price
        take_profit: New target price

    Returns:
        dict with the updated levels and the current price. On ANY failure
        returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(
            _paper_modify, position_id, stop_loss, take_profit, wallet
        )
    except Exception as e:
        return exception_to_envelope(e, context="paper_trade_modify")


@mcp.tool(annotations=ToolAnnotations(title="Paper Wallet Status", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def paper_wallet_status(
    include_closed: Annotated[
        int,
        Field(description="Also return this many recent closed positions, 0 for none"),
    ] = 0,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """The simulated wallet: cash, open positions marked live, and equity.

    Every call settles first, so stops and targets reached since the last look
    are applied against the bar highs and lows that reached them - a stop-out
    price later recovered from still counts.

    Args:
        include_closed: Also return this many recent closed positions

    Returns:
        dict with the wallet (cash, margin posted, unrealised PnL, equity,
        total return), open positions with liquidation prices, and whatever
        settled on this check. On ANY failure returns a structured error
        envelope.
    """
    try:
        return await asyncio.to_thread(_paper_status, include_closed, wallet)
    except Exception as e:
        return exception_to_envelope(e, context="paper_wallet_status")


@mcp.tool(annotations=ToolAnnotations(title="Paper Trade History", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def paper_trade_history(
    limit: Annotated[
        int,
        Field(description="Closed trades to return, 1 to 500"),
    ] = 50,
    include_equity_curve: Annotated[
        bool,
        Field(description="Add the equity after every trade, with the drawdown at "
            "each point"),
    ] = False,
    wallet: Annotated[
        Optional[str],
        Field(description="Which wallet, e.g. \"swing\" or \"risky\". Omit "
            "for the default one. Each name is a separate balance, with its "
            "own positions, orders and history"),
    ] = None,
) -> dict:
    """Closed simulated trades and the performance they add up to.

    Win rate, profit factor, expectancy, realised drawdown sequenced by close
    time, and what fees and funding consumed - so a strategy that only works
    before costs is visible as such.

    Set include_equity_curve when the person wants to SEE how the account
    developed: statistics describe a record, a curve shows its shape, and two
    strategies with the same win rate can look nothing alike. The curve also
    carries gross_equity — the same trades with fees and funding added back —
    so the gap between the lines is what trading cost.

    Args:
        limit: Trades to return, 1-500
        include_equity_curve: Add equity after each trade, with drawdown

    Returns:
        dict with trades, a statistics block and optionally equity_curve. On
        ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_paper_history, limit, include_equity_curve, wallet)
    except Exception as e:
        return exception_to_envelope(e, context="paper_trade_history")


@mcp.tool(
    structured_output=False,
    annotations=ToolAnnotations(
        title="Paper Wallet Panel", readOnlyHint=True, destructiveHint=False,
        openWorldHint=True,
    ),
    **apps.tool_kwargs(wallet_app.WALLET_URI),
)
async def paper_wallet_widget(include_closed: int = 0, wallet: Optional[str] = None,
                              ctx: Context = None) -> CallToolResult:
    """The practice wallet as a live panel, on apps that can draw one.

    Same figures as paper_wallet_status. On a client that supports MCP Apps
    the result is drawn as a panel with its own refresh button, and you are
    handed a short summary and told the panel is on screen - do not repeat
    the numbers or draw another. On any other client this returns exactly
    what paper_wallet_status returns, display_hint included, so nothing is
    lost by calling it.

    SIMULATED: nothing reaches an exchange.

    Args:
        include_closed: Also include this many recent closed positions
        wallet: Which wallet. Omit for the default one.

    Returns:
        The wallet status. On ANY failure the text carries the error
        envelope.
    """
    try:
        status = await asyncio.to_thread(_paper_status, include_closed, wallet)
    except Exception as e:
        status = exception_to_envelope(e, context="paper_wallet_widget")
    name = wallet_app.wallet_label(wallet)
    draws = apps.client_supports_ui(ctx) if _WALLET_APP_REGISTERED else False
    shown = dict(status)
    if "error" not in shown:
        shown["wallet_name"] = name
        shown["wallet_is_default"] = name == wallet_app.DEFAULT_WALLET
    return CallToolResult(
        content=[TextContent(
            type="text", text=wallet_app.text_for_model(status, name, draws),
        )],
        structuredContent=shown,
    )


# ── Alerts ─────────────────────────────────────────────────────────────────────
#
# The only tools here that keep state. They write rows to a local SQLite file
# (see alert_store) and nothing else: no order, no transfer, no external
# service, no network side effect. Their annotations say so honestly —
# readOnlyHint is False because they do persist something — and
# test_tool_annotations keeps every OTHER tool read-only.

@mcp.tool(annotations=ToolAnnotations(title="Create Alert", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
async def alert_create(
    symbol: Annotated[
        str,
        Field(description="Pair or ticker, e.g. BTCUSDT. Ignored for market-wide "
            "metrics"),
    ],
    metric: Annotated[
        str,
        Field(description="What to watch, e.g. price, rsi, funding_annualised_pct, "
            "crowd_long_pct, breadth_above_ema50_pct. Call "
            "alert_metrics for the full list with units"),
    ],
    operator: Annotated[
        str,
        Field(description="\"above\" or \"below\" - tested at check time, so it is a "
            "level, not a crossing"),
    ],
    threshold: Annotated[
        float,
        Field(description="The number to compare the metric against, in the metric's "
            "own units"),
    ],
    exchange: Annotated[
        str,
        Field(description="Venue, e.g. KUCOIN, BINANCE, NASDAQ"),
    ] = "KUCOIN",
    timeframe: Annotated[
        str,
        Field(description="Timeframe the metric is read on, e.g. 4h, 1D"),
    ] = "4h",
    note: Annotated[
        Optional[str],
        Field(description="Free text stored with the alert and echoed when it fires"),
    ] = None,
    repeat_after_fire: Annotated[
        bool,
        Field(description="Keep checking after it fires instead of disarming itself"),
    ] = False,
) -> dict:
    """Store a standing condition to be evaluated later by alert_check.

    Nothing is pushed anywhere — MCP is request/response. Pair this with a
    scheduler that calls alert_check and relays what fired.

    Conditions are a fixed vocabulary, not expressions: a metric, an operator
    and a threshold. Call alert_metrics for the list with units.

    Args:
        symbol: Pair or ticker, e.g. BTCUSDT. Ignored for market-wide metrics
        metric: e.g. price, rsi, funding_annualised_pct, crowd_long_pct,
            breadth_above_ema50_pct — see alert_metrics
        operator: "above" or "below" (tested at check time, not a crossing)
        threshold: The number to compare against
        exchange: Venue, e.g. KUCOIN, BINANCE, NASDAQ
        timeframe: Timeframe for the metric, e.g. 4h, 1D
        note: Free text stored with the alert and echoed when it fires
        repeat_after_fire: Keep checking after it fires instead of disarming

    Returns:
        The stored alert including its id. On ANY failure returns a structured
        error envelope.
    """
    try:
        return await asyncio.to_thread(
            create_alert, symbol, metric, operator, threshold,
            exchange, timeframe, note, repeat_after_fire,
        )
    except Exception as e:
        return exception_to_envelope(e, context="alert_create")


@mcp.tool(annotations=ToolAnnotations(title="Check Alerts", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def alert_check(
    include_triggered: Annotated[
        bool,
        Field(description="Also re-evaluate alerts that have already fired"),
    ] = False,
) -> dict:
    """Evaluate armed alerts and report which conditions are now met.

    Scanner-backed metrics are batched: fifty price alerts on one exchange
    cost ONE upstream request, not fifty.

    Alerts whose data could not be fetched come back under not_evaluated
    rather than counted as "condition not met" — a dead feed must not read as
    a quiet market.

    Args:
        include_triggered: Also re-evaluate alerts that already fired

    Returns:
        dict with fired, still_waiting (each carrying distance_to_trigger) and
        not_evaluated. On ANY failure returns a structured error envelope.
    """
    try:
        return await asyncio.to_thread(check_alerts, include_triggered)
    except Exception as e:
        return exception_to_envelope(e, context="alert_check")


@mcp.tool(annotations=ToolAnnotations(title="List Alerts", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
async def alert_list(
    state: Annotated[
        Optional[str],
        Field(description="\"armed\", \"triggered\", or omit for all of them"),
    ] = None,
) -> dict:
    """List stored alerts and their state.

    Args:
        state: "armed", "triggered", or omit for all

    Returns:
        dict with alerts, counts and the database path. On ANY failure returns
        a structured error envelope.
    """
    try:
        return await asyncio.to_thread(_list_alerts, state)
    except Exception as e:
        return exception_to_envelope(e, context="alert_list")


@mcp.tool(annotations=ToolAnnotations(title="Manage Alert", readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
async def alert_manage(
    alert_id: Annotated[
        int,
        Field(description="The id returned by alert_create or alert_list"),
    ],
    action: Annotated[
        str,
        Field(description="\"delete\" or \"rearm\""),
    ] = "delete",
) -> dict:
    """Delete an alert, or re-arm one that already fired.

    Deleting is irreversible — the row is removed, not archived.

    Args:
        alert_id: The id returned by alert_create or alert_list
        action: "delete" or "rearm"

    Returns:
        dict confirming what happened. On ANY failure returns a structured
        error envelope.
    """
    try:
        act = (action or "delete").strip().lower()
        if act == "delete":
            return await asyncio.to_thread(_delete_alert, alert_id)
        if act == "rearm":
            return await asyncio.to_thread(rearm_alert, alert_id)
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"action must be 'delete' or 'rearm', got {action!r}",
        )
    except Exception as e:
        return exception_to_envelope(e, context="alert_manage")


@mcp.tool(annotations=ToolAnnotations(title="Alert Metrics Catalogue", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
async def alert_metrics() -> dict:
    """The metrics alert_create accepts, with units and data source."""
    try:
        return {"metrics": await asyncio.to_thread(metric_catalogue),
                "operators": ["above", "below"],
                "note": "operators test the value at check time, not a crossing"}
    except Exception as e:
        return exception_to_envelope(e, context="alert_metrics")


# ── Resource ───────────────────────────────────────────────────────────────────

@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """List available exchanges from the coinlist directory."""
    try:
        current_dir = os.path.dirname(__file__)
        coinlist_dir = os.path.join(current_dir, "coinlist")
        if os.path.exists(coinlist_dir):
            exchanges = [
                f[:-4].upper()
                for f in os.listdir(coinlist_dir)
                if f.endswith(".txt")
            ]
            if exchanges:
                return f"Available exchanges: {', '.join(sorted(exchanges))}"
    except Exception:
        pass
    # Fallback must only name venues sanitize/validate actually accept —
    # this list used to advertise KRAKEN and BITSTAMP, which are unsupported.
    return "Common exchanges: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, BIST, EGX, NASDAQ, NYSE, TWSE, TPEX"


# ── Health endpoint (streamable-http transport) ───────────────────────────────

@mcp.custom_route("/health", methods=["GET"])
async def _health(_request):
    """Liveness probe for the Dockerfile HEALTHCHECK and orchestrators.

    FastMCP's streamable-http transport serves only /mcp by default, so the
    container HEALTHCHECK against /health cycled every container to
    `unhealthy` — breaking compose `condition: service_healthy` and swarm
    restarts. Cheap 200, no upstream calls.
    """
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Blanket async offload for every still-synchronous tool.
#
# mcp 1.12.x calls sync tool functions BARE on the event loop
# (func_metadata.call_fn_with_arg_validation ends in `return fn(**kwargs)`),
# so one slow scan stalls every concurrent session. Seven hot paths were
# hand-converted to async (#49), which left an arbitrary split — top_gainers
# offloaded while its twin top_losers blocked (issue #77). Rather than
# maintaining that list tool-by-tool, wrap whatever is still sync at import
# time in asyncio.to_thread. Tools added later are covered automatically the
# moment this module loads. Argument validation is untouched: FastMCP
# validates against fn_metadata (built from the ORIGINAL signature via
# functools.wraps) and then calls fn(**parsed_kwargs), which the wrapper
# forwards verbatim.
# ---------------------------------------------------------------------------
def _offload_sync_tools() -> int:
    import functools

    converted = 0
    for _tool in mcp._tool_manager.list_tools():
        if _tool.is_async:
            continue
        _sync_fn = _tool.fn

        @functools.wraps(_sync_fn)
        async def _threaded(*args, __sync_fn=_sync_fn, **kwargs):
            return await asyncio.to_thread(__sync_fn, *args, **kwargs)

        _tool.fn = _threaded
        _tool.is_async = True
        converted += 1
    return converted


_OFFLOADED_TOOL_COUNT = _offload_sync_tools()


# ---------------------------------------------------------------------------
# A scan that matches nothing still has to say something.
#
# Six screeners return a bare list, and an empty list reaches the client as
# no content at all: asked "any coins breaking out right now?", the
# assistant was handed a blank and had to guess whether that meant none,
# a failure or a wrong argument. Measured by running the calls two models
# chose for 35 everyday questions. The rows are untouched when there are
# any; only the empty case becomes an object that says what was asked and
# what to loosen.
# ---------------------------------------------------------------------------
_EMPTY_ADVICE = {
    "volume_breakout_scanner": (
        "Lower volume_multiplier or price_change_min, or use a longer "
        "timeframe: on short bars little clears the default 2x volume and 3% move."
    ),
    "smart_volume_scanner": (
        "Lower min_volume_ratio or min_price_change, or set rsi_range to 'any'."
    ),
    "bollinger_scan": (
        "Raise bbw_threshold: it is the widest band still counted as a "
        "squeeze. Typical values are 0.008 on 15m, 0.02 on 1h, 0.04 on 4h, "
        "0.12 on 1D."
    ),
    "rating_filter": (
        "Try a neighbouring rating (-3 to 3) or another timeframe; strong "
        "ratings are rare on short bars."
    ),
    "top_gainers": "Check the exchange name, or try another timeframe.",
    "top_losers": "Check the exchange name, or try another timeframe.",
}


def empty_result(tool_name: str, asked: dict) -> dict:
    """What an empty scan returns instead of nothing."""
    return {
        "matches": 0,
        "rows": [],
        "meaning": (
            "The scan ran and nothing met these settings right now. That "
            "is an answer, not a failure: say so rather than retrying "
            "the same call."
        ),
        "asked": {k: v for k, v in asked.items() if v is not None},
        "to_widen": _EMPTY_ADVICE.get(
            tool_name, "Loosen the thresholds or try another timeframe."
        ),
    }


def _explain_empty_results() -> int:
    import functools
    import inspect

    wrapped = 0
    for _tool in mcp._tool_manager.list_tools():
        _origin = _tool.fn
        while hasattr(_origin, "__wrapped__"):
            _origin = _origin.__wrapped__
        _signature = inspect.signature(_origin)
        if "list" not in str(_signature.return_annotation):
            continue
        if "dict" not in str(_signature.return_annotation):
            continue  # a tool that promises only a list keeps its promise
        _inner = _tool.fn

        @functools.wraps(_inner)
        async def _explained(*args, __inner=_inner, __name=_tool.name,
                             __sig=_signature, **kwargs):
            result = await __inner(*args, **kwargs)
            if isinstance(result, list) and not result:
                try:
                    bound = __sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    asked = dict(bound.arguments)
                except TypeError:
                    asked = dict(kwargs)
                return empty_result(__name, asked)
            return result

        _tool.fn = _explained
        wrapped += 1
    return wrapped


_EXPLAINED_TOOL_COUNT = _explain_empty_results()


# ---------------------------------------------------------------------------
# A call budget and an audit line around every tool.
#
# Both are off for a plain stdio install and switched on by environment for
# a hosted one, where every caller shares one process, one disk and one
# upstream allowance. Applied here, after the offload above, so it covers
# every tool including ones added later.
# ---------------------------------------------------------------------------
def _guard_tools() -> int:
    import functools
    import time as _time

    from tradingview_mcp.core import limits
    from tradingview_mcp.core.errors import ErrorCode, make_error

    budget = limits.budget_from_env()
    audited = limits.audit_enabled()
    if budget.per_minute <= 0 and not audited:
        return 0

    guarded = 0
    for _tool in mcp._tool_manager.list_tools():
        _inner = _tool.fn

        @functools.wraps(_inner)
        async def _guarded(*args, __inner=_inner, __name=_tool.name, **kwargs):
            wait = budget.take()
            if wait is not None:
                if audited:
                    limits.audit(__name, 0.0, "refused:RATE_LIMITED")
                return make_error(
                    ErrorCode.RATE_LIMITED,
                    f"This server allows {budget.per_minute} tool calls a "
                    f"minute across everyone using it, and that is used up. "
                    f"Try again in about {wait:.0f} seconds.",
                    retry_after_s=wait, retryable=True,
                )
            started = _time.time()
            outcome = "ok"
            try:
                result = await __inner(*args, **kwargs)
                if isinstance(result, dict) and isinstance(result.get("error"), dict):
                    outcome = f"error:{result['error'].get('code')}"
                return result
            except Exception as exc:
                outcome = f"raised:{type(exc).__name__}"
                raise
            finally:
                if audited:
                    limits.audit(__name, _time.time() - started, outcome)

        _tool.fn = _guarded
        guarded += 1
    return guarded


_GUARDED_TOOL_COUNT = _guard_tools()


# ── Transport security (streamable-http) ──────────────────────────────────────

_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _host_patterns(name: str) -> list:
    """Both forms of a host, because one pattern does not cover the other.

    The SDK matches a Host header exactly, or against ``base:*`` by testing
    ``host.startswith(base + ":")`` — so ``example.com:*`` never matches a
    bare ``example.com``. A client reaching the server on 443 sends no port
    and would be refused by the wildcard alone.
    """
    name = name.strip().rstrip("/")
    if not name:
        return []
    return [name, f"{name}:*"]


def resolve_transport_security(bind_host: str, allowed_hosts: list):
    """Decide the DNS-rebinding policy for an HTTP bind. Returns (settings, warning).

    Three cases, and the middle one is the bug this exists to fix:

      * Hosts were named — protect, and accept exactly those (plus localhost,
        so a health probe from inside the container still works). Origins are
        populated too: with protection on and ``allowed_origins`` empty, the
        SDK refuses any request that carries an Origin header at all, which
        would break a browser-based client.
      * Nothing named and the bind is public — turn protection OFF and say so
        loudly. Rebinding attacks aim a browser at a server on localhost; a
        server deliberately bound to 0.0.0.0 for remote clients is a
        different threat model, and silently refusing every request is worse
        than not protecting against that one. Put a token in front of it.
      * Nothing named and the bind is local — keep the SDK's own localhost
        policy. Stated explicitly rather than left to None, because None
        means "no settings", which the middleware reads as protection off.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    local = []
    for name in _LOCAL_HOSTS:
        local.extend(_host_patterns(f"[{name}]" if name == "::1" else name))

    names = [n for n in (h.strip() for h in allowed_hosts) if n]
    if names:
        hosts = list(local)
        origins = []
        for name in names:
            hosts.extend(_host_patterns(name))
            for scheme in ("https", "http"):
                origins.extend(_host_patterns(f"{scheme}://{name}"))
        for name in _LOCAL_HOSTS:
            bare = f"[{name}]" if name == "::1" else name
            origins.extend(_host_patterns(f"http://{bare}"))
        return (
            TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=hosts,
                allowed_origins=origins,
            ),
            None,
        )

    if bind_host not in _LOCAL_HOSTS:
        return (
            TransportSecuritySettings(enable_dns_rebinding_protection=False),
            f"DNS-rebinding protection is OFF: bound to {bind_host} with no "
            "--allowed-host given, and refusing every request instead would "
            "be worse. Pass --allowed-host <public-hostname> (repeatable) or "
            "set TRADINGVIEW_MCP_ALLOWED_HOSTS to turn it back on. This "
            "server has no authentication of its own either — put a reverse "
            "proxy that checks a token in front of it.",
        )

    return (
        TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(local),
            allowed_origins=[
                p for name in _LOCAL_HOSTS
                for p in _host_patterns(
                    f"http://{'[::1]' if name == '::1' else name}"
                )
            ],
        ),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView Screener MCP server")
    parser.add_argument(
        "transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport (default stdio)",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        metavar="HOSTNAME",
        help=(
            "Host header to accept over streamable-http, e.g. mcp.example.com. "
            "Repeatable. Defaults to TRADINGVIEW_MCP_ALLOWED_HOSTS (comma "
            "separated). Required to keep DNS-rebinding protection on when "
            "binding to anything other than localhost."
        ),
    )
    args = parser.parse_args()

    if os.environ.get("DEBUG_MCP"):
        import sys
        print(f"[DEBUG_MCP] pkg cwd={os.getcwd()} argv={sys.argv} file={__file__}", file=sys.stderr, flush=True)

    if args.transport == "stdio":
        mcp.run()
    else:
        import sys

        try:
            mcp.settings.host = args.host
            mcp.settings.port = args.port
        except Exception:
            pass
        # Not inside the try above: a silent failure here is what made every
        # remote request fail with "Invalid Host header".
        allowed = args.allowed_host or [
            h for h in os.environ.get(
                "TRADINGVIEW_MCP_ALLOWED_HOSTS", ""
            ).split(",") if h.strip()
        ]
        security, warning = resolve_transport_security(args.host, allowed)
        mcp.settings.transport_security = security
        if warning:
            print(f"[tradingview_mcp] {warning}", file=sys.stderr, flush=True)
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
