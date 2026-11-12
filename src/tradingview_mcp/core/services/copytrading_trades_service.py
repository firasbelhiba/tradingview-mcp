"""
Copy Trading Trades Service — a lead trader's full closed-trade log.

``lead_trader_profile`` aggregates a sample of trades; this returns the
trades themselves, paging back through the whole history, and computes the
performance statistics the leaderboard never shows.

Pagination is real: ``public-subpositions-history`` caps ``limit`` at 100 and
accepts ``after=<subPosId>`` to walk backwards. Verified against a live
trader — page two returned 74 further rows with zero overlap — so a complete
log is reachable rather than a recent slice.

Two things about this data that will mislead a reader who is not told:

  * **``pnlRatio`` is return on MARGIN, not on notional.** A value of -2.62
    means the position lost 262% of the margin committed to it, which is
    possible because the trade was levered. Read as a price move it looks
    impossible.
  * **An equity curve built from closed trades is ordered by CLOSE time**, so
    overlapping positions are credited when each one closed, not while it was
    open. The drawdown that follows is therefore realised drawdown, not
    mark-to-market.

The drawdown here is strictly better than the weekly one in
``copytrading_service``: trade-level sequencing sees declines that opened and
closed inside a single week.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.copytrading_service import (
    HIDDEN_NOTE,
    INSTRUMENT_TYPES,
    _HIDDEN_CODE,
    _f,
    _get,
    _rows,
    trader_is_known,
    unknown_trader_error,
)

_PAGE_SIZE = 100      # OKX's ceiling; 200 is rejected outright
_MAX_PAGES = 20       # 2,000 trades, i.e. 20 requests at worst
_MS_PER_HOUR = 3_600_000.0


def _ms_to_iso(value: Any) -> Optional[str]:
    ms = _f(value)
    if ms is None:
        return None
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _trade(raw: dict) -> dict:
    open_ms, close_ms = _f(raw.get("openTime")), _f(raw.get("closeTime"))
    held_h = (
        None if open_ms is None or close_ms is None or close_ms < open_ms
        else (close_ms - open_ms) / _MS_PER_HOUR
    )
    ratio = _f(raw.get("pnlRatio"))
    return {
        "trade_id": raw.get("subPosId"),
        "instrument": raw.get("instId"),
        "side": raw.get("posSide"),
        "leverage": _f(raw.get("lever")),
        "margin": _f(raw.get("margin")),
        "size": _f(raw.get("subPos")),
        "open_price": _f(raw.get("openAvgPx")),
        "close_price": _f(raw.get("closeAvgPx")),
        "opened_at": _ms_to_iso(raw.get("openTime")),
        "closed_at": _ms_to_iso(raw.get("closeTime")),
        "held_hours": None if held_h is None else round(held_h, 2),
        "pnl": _f(raw.get("pnl")),
        "pnl_pct_of_margin": None if ratio is None else round(ratio * 100, 2),
        "currency": raw.get("ccy"),
        "margin_mode": raw.get("mgnMode"),
    }


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _equity_curve(trades: List[dict]) -> Tuple[List[dict], Optional[float], Optional[float]]:
    """Cumulative realised PnL in close-time order, with its worst drawdown."""
    ordered = [t for t in trades if t.get("closed_at") and t.get("pnl") is not None]
    ordered.sort(key=lambda t: t["closed_at"])
    curve: List[dict] = []
    cumulative, peak, worst, worst_pct = 0.0, 0.0, 0.0, 0.0
    for t in ordered:
        cumulative += t["pnl"]
        peak = max(peak, cumulative)
        decline = cumulative - peak
        if decline < worst:
            worst = decline
            worst_pct = (decline / peak * 100.0) if peak > 0 else 0.0
        curve.append({
            "closed_at": t["closed_at"],
            "pnl": round(t["pnl"], 2),
            "cumulative_pnl": round(cumulative, 2),
        })
    if not curve:
        return [], None, None
    return curve, round(worst, 2), round(worst_pct, 1)


def _statistics(trades: List[dict]) -> dict:
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    ratios = [t["pnl_pct_of_margin"] for t in trades
              if t.get("pnl_pct_of_margin") is not None]
    levers = [t["leverage"] for t in trades if t.get("leverage")]
    held = [t["held_hours"] for t in trades if t.get("held_hours") is not None]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = (gross_profit / len(wins)) if wins else None
    avg_loss = (gross_loss / len(losses)) if losses else None

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(pnls) - len(wins) - len(losses),
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
        "net_pnl": round(sum(pnls), 2) if pnls else None,
        "gross_profit": round(gross_profit, 2) if wins else None,
        "gross_loss": round(gross_loss, 2) if losses else None,
        # Gross profit over gross loss. Above 1 means the wins outweigh the
        # losses; it is undefined with no losing trade, which is reported as
        # null rather than infinity so it cannot read as a perfect record.
        "profit_factor": (
            None if not losses or not wins else round(gross_profit / gross_loss, 2)
        ),
        "avg_win": None if avg_win is None else round(avg_win, 2),
        "avg_loss": None if avg_loss is None else round(avg_loss, 2),
        "win_loss_size_ratio": (
            None if not avg_win or not avg_loss else round(avg_win / avg_loss, 2)
        ),
        # Expected PnL per trade on this sample — the win rate and the
        # win/loss sizes combined, which either alone can flatter.
        "expectancy_per_trade": round(sum(pnls) / len(pnls), 2) if pnls else None,
        "best_trade": round(max(pnls), 2) if pnls else None,
        "worst_trade": round(min(pnls), 2) if pnls else None,
        "best_trade_pct_of_margin": max(ratios) if ratios else None,
        "worst_trade_pct_of_margin": min(ratios) if ratios else None,
        "avg_leverage": round(sum(levers) / len(levers), 1) if levers else None,
        "max_leverage": round(max(levers), 0) if levers else None,
        "median_hold_hours": (
            None if not held else round(_median(held) or 0.0, 2)
        ),
        "avg_hold_hours": round(sum(held) / len(held), 2) if held else None,
        "longest_hold_hours": round(max(held), 2) if held else None,
    }


def _breakdown(trades: List[dict], key: str) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    for t in trades:
        value = t.get(key)
        if value:
            groups.setdefault(str(value), []).append(t)
    out = []
    for name, rows in groups.items():
        pnls = [r["pnl"] for r in rows if r.get("pnl") is not None]
        wins = sum(1 for p in pnls if p > 0)
        out.append({
            key: name,
            "trades": len(rows),
            "net_pnl": round(sum(pnls), 2) if pnls else None,
            "win_rate_pct": round(wins / len(pnls) * 100, 1) if pnls else None,
        })
    out.sort(key=lambda r: -(r["net_pnl"] or 0))
    return out


def get_lead_trader_trades(
    unique_code: str,
    instrument_type: str = "SWAP",
    max_trades: int = 300,
    instrument: Optional[str] = None,
    include_trades: bool = True,
    include_equity_curve: bool = False,
    trade_rows: int = 40,
) -> dict:
    """Every closed trade for one lead trader, with performance statistics.

    Args:
        unique_code:          The trader's ``uniqueCode``.
        instrument_type:      ``SWAP`` or ``SPOT``. A futures-only trader
                              returns nothing under SPOT and vice versa.
        max_trades:           Upper bound on trades fetched, 10-2000. Each
                              100 is one request.
        instrument:           Keep only this instrument, e.g. ``BTC-USDT-SWAP``.
                              Filtering happens after fetching, so statistics
                              describe the filtered set.
        include_trades:       Return the individual rows. Set false for
                              statistics only.
        include_equity_curve: Return cumulative realised PnL per trade.
    """
    if not unique_code or not str(unique_code).strip():
        return make_error(ErrorCode.INVALID_PARAMETER, "unique_code is required")
    if instrument_type not in INSTRUMENT_TYPES:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"instrument_type must be one of {', '.join(INSTRUMENT_TYPES)}",
        )
    if instrument_type != "SWAP":
        # Measured, not assumed: public-subpositions-history rejects
        # instType=SPOT outright (400, code 51000), and omitting instType
        # returns zero rows for every SPOT lead trader tried. The endpoint
        # publishes futures positions only, so SPOT traders have no public
        # trade log to list — saying so beats passing it through to a 400.
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            "Trade history is published for SWAP (futures) lead traders only. "
            "OKX rejects instType=SPOT on this endpoint and returns no rows "
            "for spot lead traders even with it omitted, so there is no spot "
            "trade log to list. lead_traders and lead_trader_profile do "
            "support SPOT.",
            requested=instrument_type,
            supported=["SWAP"],
        )
    if not 10 <= max_trades <= _PAGE_SIZE * _MAX_PAGES:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"max_trades must be 10-{_PAGE_SIZE * _MAX_PAGES}, got {max_trades}",
        )

    code = str(unique_code).strip()
    collected: List[dict] = []
    seen: set = set()
    pages = 0
    after: Optional[str] = None
    truncated = False
    error: Optional[str] = None

    while len(collected) < max_trades and pages < _MAX_PAGES:
        path = (
            f"public-subpositions-history?instType={instrument_type}"
            f"&uniqueCode={code}&limit={_PAGE_SIZE}"
        )
        if after:
            path += f"&after={after}"
        try:
            page = _rows(_get(path))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:120]
            break
        pages += 1
        if not page:
            break
        fresh = [r for r in page if r.get("subPosId") not in seen]
        if not fresh:
            break  # the API repeated a page; stop rather than loop
        for r in fresh:
            seen.add(r.get("subPosId"))
        collected.extend(fresh)
        after = page[-1].get("subPosId")
        if len(page) < _PAGE_SIZE:
            break  # a short page is the end of the history
    else:
        truncated = True

    if not collected:
        if error and _HIDDEN_CODE in error:
            # The same error covers a typo. Only say "hidden" for a
            # trader OKX otherwise admits to.
            if trader_is_known(code, instrument_type, get=_get) is False:
                return unknown_trader_error(code)
            # OKX words this "This futures lead trader doesn't exist". They
            # do: the profile tool reads their aggregates. What is withheld is
            # the trade history, and asking again will not change that.
            return make_error(
                ErrorCode.NO_DATA,
                "This trader's trade history is hidden. " + HIDDEN_NOTE,
                unique_code=code,
                trade_history_state="hidden",
                retryable=False,
            )
        if error:
            return make_error(
                ErrorCode.UPSTREAM_ERROR, f"OKX: {error}",
                unique_code=code, retryable=True,
            )
        return make_error(
            ErrorCode.NO_DATA,
            "No closed futures trades published for this trader. The usual "
            "cause is that they lead on SPOT, whose trade history OKX does "
            "not publish — lead_trader_profile still works for them.",
            unique_code=code,
            instrument_type=instrument_type,
        )

    trades = [_trade(r) for r in collected[:max_trades]]
    if instrument:
        wanted = instrument.strip().upper()
        trades = [t for t in trades if (t.get("instrument") or "").upper() == wanted]
        if not trades:
            return make_error(
                ErrorCode.NO_DATA,
                f"No closed trades on {wanted} in the {len(collected)} fetched",
                unique_code=code, instrument=wanted,
            )

    trades.sort(key=lambda t: t.get("closed_at") or "", reverse=True)
    stats = _statistics(trades)
    curve, drawdown, drawdown_pct = _equity_curve(trades)

    result: Dict[str, Any] = {
        "source": "OKX copy trading, public endpoints (no key)",
        "unique_code": code,
        "instrument_type": instrument_type,
        "pages_fetched": pages,
        "trades_fetched": len(collected),
        "trades_returned": len(trades),
        "complete_history": not truncated and not error,
        "statistics": stats,
        "realised_drawdown": drawdown,
        "realised_drawdown_pct_of_peak": drawdown_pct,
        "by_instrument": _breakdown(trades, "instrument"),
        "by_side": _breakdown(trades, "side"),
        "period": {
            "oldest_close": min((t["closed_at"] for t in trades if t["closed_at"]),
                                default=None),
            "newest_close": max((t["closed_at"] for t in trades if t["closed_at"]),
                                default=None),
        },
        "field_notes": {
            "pnl_pct_of_margin": (
                "return on the MARGIN committed, not on notional — -262 means "
                "the position lost 2.62x its margin, which leverage allows"
            ),
            "realised_drawdown": (
                "peak-to-trough of cumulative REALISED pnl, ordered by close "
                "time; it does not include open-position mark-to-market, and "
                "overlapping trades are credited when each closed"
            ),
            "profit_factor": "gross profit / gross loss; null when the sample has no loss",
        },
        "read_only_note": (
            "Published statistics only. This places, follows and sizes "
            "nothing, and is not a recommendation."
        ),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
    if truncated:
        result["truncation_note"] = (
            f"Stopped at {len(collected)} trades (max_trades={max_trades}, "
            f"page cap {_MAX_PAGES}); older trades exist beyond this."
        )
    if error:
        result["incomplete"] = f"paging stopped early: {error}"
    if include_trades:
        # Statistics above cover every trade fetched; the rows are capped,
        # because 300 of them measured 95-150 KB in a single answer.
        shown = trades[:max(0, int(trade_rows))]
        result["trades"] = shown
        result["trades_shown"] = len(shown)
        if len(shown) < len(trades):
            result["trades_omitted"] = len(trades) - len(shown)
            result["trades_note"] = (
                f"showing the {len(shown)} most recent of {len(trades)} "
                "trades; the statistics cover all of them. Raise "
                "trade_rows to see more."
            )
    if include_equity_curve:
        result["equity_curve"] = curve
    return result
