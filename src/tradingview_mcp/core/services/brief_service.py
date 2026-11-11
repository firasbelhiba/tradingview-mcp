"""
Brief Service — "how's the market?" in one call.

That question is the most common thing anyone asks this server and, until
now, the most expensive: breadth, the bitcoin pulse, positioning, sentiment,
your alerts and your wallet were six round trips the assistant had to
remember to make, and usually made inconsistently.

This runs them concurrently and returns one object. Two rules shape it:

  * **A section that fails says so and the brief still returns.** Six feeds
    means six chances to fail, and losing the whole answer because the
    sentiment index timed out would be worse than the six separate calls
    this replaces. Failures land in ``unavailable`` with their reason.
  * **It reports, it does not advise.** The sections are assembled and their
    own assessments quoted; nothing here decides the market is bullish or
    tells anyone what to do with it. Reading across the sections is the
    assistant's job, and acting on it is the person's.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error

_WORKERS = 6


def _safe(name: str, fn: Callable[[], Any]) -> tuple:
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001
        return name, None, f"{type(exc).__name__}: {str(exc)[:90]}"
    if isinstance(out, dict) and "error" in out:
        err = out["error"]
        return name, None, f"{err.get('code')}: {str(err.get('message'))[:90]}"
    return name, out, None


def _breadth(exchange: str) -> dict:
    from tradingview_mcp.core.services.market_structure_service import (
        analyze_market_breadth,
    )
    out = analyze_market_breadth(exchange, "1D")
    tp = out["trend_participation"]
    ad = out["advance_decline"]
    return {
        "exchange": out["exchange"],
        "above_ema50_pct": tp["above_ema50_pct"],
        "above_sma200_pct": tp["above_sma200_pct"],
        "advancing": ad["advancing"],
        "declining": ad["declining"],
        "median_change_pct": ad["median_change_pct"],
        "verdict": out["assessment"]["label"],
        "reasoning": out["assessment"]["reasoning"],
        "measured_over": out["symbols_with_data"],
    }


def _bitcoin() -> dict:
    from tradingview_mcp.core.services.bitcoin_market_service import (
        get_bitcoin_market_pulse,
    )
    out = get_bitcoin_market_pulse()
    return {
        "price": out["bitcoin"]["price_usd"],
        "change_24h_pct": round(out["bitcoin"]["change_24h_pct"], 2),
        "btc_dominance_pct": round(out["dominance"]["btc_pct"], 1),
        "total_market_change_24h_pct": round(
            out["total_market"]["change_24h_pct"], 2
        ),
        "verdict": out["assessment"]["label"],
        "reasoning": out["assessment"]["summary"],
    }


def _positioning(symbol: str) -> dict:
    from tradingview_mcp.core.services.positioning_service import analyze_positioning
    out = analyze_positioning(symbol, "4h")
    a = out["assessment"]
    return {
        "symbol": out["symbol"],
        "crowd_long_pct": out["crowd"]["long_pct"],
        "top_traders_long_pct": out["top_traders"]["long_pct"],
        "divergence_pts": out["crowd_vs_top_divergence_pts"],
        "funding_annualised_pct": out["funding"]["annualised_pct"],
        "open_interest_change_pct": out["open_interest"]["change_pct"],
        "squeeze_risk": a["squeeze_risk"],
        "reasoning": a["squeeze_reasoning"],
    }


def _sentiment() -> dict:
    from tradingview_mcp.core.services.sentiment_index_service import get_fear_greed
    out = get_fear_greed(14)
    return {
        "fear_greed": out["current"]["value"],
        "classification": out["current"]["classification"],
        "change_over_14d": out["change_over_window"],
        "direction": out["direction"],
    }


def _alerts() -> dict:
    from tradingview_mcp.core.services.alert_service import check_alerts
    out = check_alerts()
    return {
        "checked": out["checked"],
        "fired": [
            {"alert_id": f["alert_id"], "condition": f["condition"],
             "value": f.get("value")}
            for f in out.get("fired", [])
        ],
        "still_waiting": len(out.get("still_waiting") or []),
        "not_evaluated": len(out.get("not_evaluated") or []),
    }


def _wallet() -> dict:
    from tradingview_mcp.core.services.paper_trading_service import wallet_status
    out = wallet_status()
    w = out["wallet"]
    return {
        "equity": w["equity"],
        "currency": w["currency"],
        "total_return_pct": w["total_return_pct"],
        "unrealised_pnl": w["unrealised_pnl"],
        "open_positions": out["open_count"],
        "settled_this_check": len(out.get("settled_this_check") or []),
        "unsettled_positions": len(out.get("unsettled_positions") or []),
    }


def market_brief(exchange: str = "KUCOIN", symbol: str = "BTCUSDT",
                 include_wallet: bool = True,
                 include_alerts: bool = True) -> dict:
    """The market in one call: regime, bitcoin, positioning, sentiment, you.

    Args:
        exchange:       Venue for breadth.
        symbol:         Perpetual for the positioning read.
        include_wallet: Add the paper wallet's standing.
        include_alerts: Evaluate standing alerts and report what fired.
    """
    jobs: List[tuple] = [
        ("regime", lambda: _breadth(exchange)),
        ("bitcoin", _bitcoin),
        ("positioning", lambda: _positioning(symbol)),
        ("sentiment", _sentiment),
    ]
    if include_alerts:
        jobs.append(("alerts", _alerts))
    if include_wallet:
        jobs.append(("wallet", _wallet))

    sections: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for name, value, error in pool.map(lambda j: _safe(j[0], j[1]), jobs):
            if error:
                unavailable[name] = error
            else:
                sections[name] = value

    if not sections:
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            "Every section of the brief failed — the upstream feeds are "
            "unreachable or throttling.",
            unavailable=unavailable,
            retryable=True,
        )

    # A one-line headline, assembled from what the sections already decided.
    # It restates; it does not judge. Anything beyond this — what it means,
    # what to do — belongs to the assistant and the person, not the server.
    bits: List[str] = []
    if "regime" in sections:
        r = sections["regime"]
        bits.append(f"breadth {r['verdict']} ({r['above_ema50_pct']}% above EMA50)")
    if "bitcoin" in sections:
        b = sections["bitcoin"]
        bits.append(f"BTC {b['change_24h_pct']:+}% at {b['price']:,.0f}")
    if "positioning" in sections:
        p = sections["positioning"]
        bits.append(f"{p['symbol']} {p['squeeze_risk']}")
    if "sentiment" in sections:
        s = sections["sentiment"]
        bits.append(f"sentiment {s['classification']} ({s['fear_greed']})")

    fired = (sections.get("alerts") or {}).get("fired") or []
    result: Dict[str, Any] = {
        "headline": " · ".join(bits) if bits else None,
        "sections": sections,
        "attention": _attention(sections, unavailable),
        "unavailable": unavailable or None,
        "fired_alerts": fired or None,
        "scope_note": (
            "Sections are gathered and their own assessments quoted. Reading "
            "across them is not done here."
        ),
        "display_hint": (
            "If the person is looking at this, draw it as a panel (see this "
            "server's PRESENTATION instructions); show attention and "
            "unavailable, not just the healthy sections."
        ),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
    return result


def _attention(sections: Dict[str, Any], unavailable: Dict[str, str]) -> List[str]:
    """Things a glance should not miss, named plainly.

    Not a verdict on the market — a list of the facts most easily lost in a
    tidy summary: a fired alert, a position nobody could check, a feed that
    did not answer.
    """
    out: List[str] = []
    alerts = sections.get("alerts") or {}
    for f in alerts.get("fired") or []:
        out.append(f"alert fired: {f['condition']} (value {f.get('value')})")
    if alerts.get("not_evaluated"):
        out.append(
            f"{alerts['not_evaluated']} alert(s) could NOT be evaluated — "
            "that is not the same as 'condition not met'"
        )
    wallet = sections.get("wallet") or {}
    if wallet.get("unsettled_positions"):
        out.append(
            f"{wallet['unsettled_positions']} wallet position(s) could not be "
            "checked against their stop or target"
        )
    if wallet.get("settled_this_check"):
        out.append(
            f"{wallet['settled_this_check']} position(s) hit a level since the "
            "last look"
        )
    for name, reason in unavailable.items():
        out.append(f"{name} unavailable: {reason}")
    return out
