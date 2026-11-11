"""
Positioning Scan Service — the perp board, not one contract.

``positioning`` answers "is the crowd exposed on BTC?". These answer "where
is the crowd exposed at all", which is the question that follows once one
symbol comes back ELEVATED.

Both lean on the bulk endpoints: ``premiumIndex`` with no symbol returns
every perpetual's funding in ONE request (~945 symbols), ``ticker/24hr``
likewise (~804), and ``fundingInfo`` gives the per-symbol funding interval.
So a whole-market funding screen costs three requests regardless of size.

The squeeze scan is two-stage for the same reason the Supertrend scan is:
crowding and open-interest history are per-symbol only, so stage one narrows
the board on the bulk data and stage two pulls the detail for the shortlist.
Grading reuses ``positioning_service._squeeze_risk`` rather than
reimplementing it, so a single-symbol verdict and a scan verdict cannot
disagree.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import display_hints
from tradingview_mcp.core.services import binance_service as bn
from tradingview_mcp.core.services import positioning_service as ps

_DEFAULT_FUNDING_INTERVAL_H = 8.0
_MAX_SHORTLIST = 30
# Measured, not guessed. Each shortlisted symbol costs three SEQUENTIAL
# requests, so wall-clock is (candidates / _WORKERS) x 3 request-times. At 6
# the full scan of 30 candidates ran 5.2-5.6s across three samples (median
# 5.3s); at 12 the median was 3.78s, with no symbol failing to grade in any
# run. Raising it further did not help - Binance paces a single IP - and the
# tail got noisier (one 8.3s sample at 12), so this buys a faster typical
# scan and a slightly wider spread, not a uniformly faster one.
_WORKERS = 12

DIRECTIONS = ("both", "longs_pay", "shorts_pay")


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _board() -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, dict]]:
    """The three bulk reads: funding, intervals, 24h stats."""
    premium = {
        row["symbol"]: row
        for row in bn.get_premium_index_all() or []
        if row.get("symbol")
    }
    intervals = bn.funding_intervals()
    tickers = {
        row["symbol"]: row
        for row in bn.get_ticker_24h_all() or []
        if row.get("symbol")
    }
    return premium, intervals, tickers


def _rows(premium: Dict[str, dict], intervals: Dict[str, float],
          tickers: Dict[str, dict],
          min_quote_volume_usd: Optional[float]) -> List[dict]:
    rows: List[dict] = []
    for symbol, p in premium.items():
        rate = _f(p.get("lastFundingRate"))
        if rate is None:
            continue
        t = tickers.get(symbol) or {}
        quote_vol = _f(t.get("quoteVolume"))
        if min_quote_volume_usd is not None and (quote_vol or 0) < min_quote_volume_usd:
            continue
        interval_h = intervals.get(symbol, _DEFAULT_FUNDING_INTERVAL_H)
        pct = rate * 100.0
        per_day = pct * (24.0 / interval_h)
        rows.append({
            "symbol": symbol,
            "funding_pct_per_interval": round(pct, 6),
            "funding_interval_hours": interval_h,
            "funding_pct_per_day": round(per_day, 4),
            "funding_annualised_pct": round(per_day * 365.0, 1),
            "payer": "longs" if pct > 0 else "shorts" if pct < 0 else "flat",
            "mark_price": _f(p.get("markPrice")),
            "price_change_24h_pct": _f(t.get("priceChangePercent")),
            "quote_volume_24h_usd": quote_vol,
        })
    return rows


def screen_funding(limit: int = 20, direction: str = "both",
                   min_quote_volume_usd: Optional[float] = 10_000_000.0) -> dict:
    """Rank the perpetual board by cost of carry.

    Three bulk requests, whatever the board's size. The interval matters:
    most perps settle every 4 hours, so annualising uniformly at 8 would
    understate the majority — ``funding_interval_hours`` is per symbol.

    Args:
        limit:                 Rows per side, 1-100.
        direction:             ``both``, ``longs_pay`` (most positive) or
                               ``shorts_pay`` (most negative).
        min_quote_volume_usd:  Drop thin contracts; a 300%/yr funding rate on
                               a market with no volume is noise, not a signal.
                               Pass ``None`` to disable.
    """
    if not 1 <= limit <= 100:
        return make_error(ErrorCode.INVALID_PARAMETER, f"limit must be 1-100, got {limit}")
    if direction not in DIRECTIONS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"direction must be one of {', '.join(DIRECTIONS)}",
            valid_directions=list(DIRECTIONS),
        )

    try:
        premium, intervals, tickers = _board()
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Binance unreachable: {str(exc)[:120]}",
            retryable=True,
        )
    if not premium:
        return make_error(
            ErrorCode.NO_DATA, "Binance returned no perpetual funding data",
            retryable=True,
        )

    rows = _rows(premium, intervals, tickers, min_quote_volume_usd)
    if not rows:
        return make_error(
            ErrorCode.NO_DATA,
            "No contracts passed the volume filter",
            min_quote_volume_usd=min_quote_volume_usd,
            perpetuals_seen=len(premium),
        )

    by_value = sorted(rows, key=lambda r: -r["funding_pct_per_interval"])
    result: Dict[str, Any] = {
        "source": "Binance USD-M futures (3 bulk requests)",
        "perpetuals_seen": len(premium),
        "after_volume_filter": len(rows),
        "filters": {"min_quote_volume_usd": min_quote_volume_usd},
        "interval_note": (
            "funding_interval_hours is per symbol — 820 of ~945 perps settle "
            "every 4h, not 8, so a uniform annualisation understates most of them"
        ),
        "non_default_intervals_known": len(intervals),
        "display_hint": display_hints.FUNDING,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
    if direction in ("both", "longs_pay"):
        result["longs_pay_most"] = by_value[:limit]
    if direction in ("both", "shorts_pay"):
        result["shorts_pay_most"] = by_value[-limit:][::-1]
    if direction == "both":
        result["summary"] = {
            "positive_funding": sum(1 for r in rows if r["funding_pct_per_interval"] > 0),
            "negative_funding": sum(1 for r in rows if r["funding_pct_per_interval"] < 0),
            "median_annualised_pct": round(
                sorted(r["funding_annualised_pct"] for r in rows)[len(rows) // 2], 1
            ),
        }
    return result


# ── Squeeze scan ──────────────────────────────────────────────────────────────

def _detail(symbol: str, period: str) -> Tuple[str, Optional[dict]]:
    """Per-symbol crowding and OI history for one shortlisted contract."""
    try:
        crowd = (bn.get_global_long_short(symbol, period, 1) or [{}])[-1]
        tops = (bn.get_top_trader_positions(symbol, period, 1) or [{}])[-1]
        oi_hist = bn.get_open_interest_history(symbol, period, 20) or []
        series = [_f(h.get("sumOpenInterest")) for h in oi_hist]
        series = [v for v in series if v is not None]
        oi_change = (
            None if len(series) < 2 or not series[0]
            else (series[-1] - series[0]) / series[0] * 100.0
        )
        return symbol, {
            "crowd_long": _f(crowd.get("longAccount")),
            "crowd_ratio": _f(crowd.get("longShortRatio")),
            "top_long": _f(tops.get("longAccount")),
            "oi_change_pct": oi_change,
        }
    except Exception:
        return symbol, None


def scan_squeeze_risk(limit: int = 12, candidates: int = 30, period: str = "4h",
                      min_quote_volume_usd: Optional[float] = 50_000_000.0) -> dict:
    """Rank the perp board by liquidation-cascade risk.

    Stage 1 narrows the board on bulk data: contracts where funding is
    meaningful AND price is already moving against the side that is paying —
    the pairing that precedes a cascade. Stage 2 fetches crowding and
    open-interest history for the shortlist and grades each one with the same
    ``_squeeze_risk`` the single-symbol tool uses.

    Args:
        limit:                 Rows returned, 1-30.
        candidates:            Contracts to fetch detail for, 1-30. Each costs
                               three requests, so this is the cost knob.
        period:                Sampling interval for crowding and OI history.
        min_quote_volume_usd:  Liquidity floor for stage 1.
    """
    if not 1 <= limit <= _MAX_SHORTLIST:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"limit must be 1-{_MAX_SHORTLIST}, got {limit}"
        )
    if not 1 <= candidates <= _MAX_SHORTLIST:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"candidates must be 1-{_MAX_SHORTLIST}, got {candidates}",
        )
    bn_period = bn.binance_period_for(period)
    if not bn_period:
        return make_error(
            ErrorCode.INVALID_TIMEFRAME,
            f"period {period!r} has no Binance equivalent",
            supported=bn.supported_periods(),
        )

    try:
        premium, intervals, tickers = _board()
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Binance unreachable: {str(exc)[:120]}",
            retryable=True,
        )
    rows = _rows(premium, intervals, tickers, min_quote_volume_usd)
    if not rows:
        return make_error(
            ErrorCode.NO_DATA, "No contracts passed the volume filter",
            min_quote_volume_usd=min_quote_volume_usd,
        )

    # Stage 1: funding against the move. A payer losing money is the setup;
    # funding alone, or a drop alone, is not.
    def pressure(r: dict) -> float:
        funding = r["funding_pct_per_interval"]
        change = r.get("price_change_24h_pct")
        if funding is None or change is None:
            return 0.0
        against = (funding > 0 and change < 0) or (funding < 0 and change > 0)
        return abs(funding) * abs(change) if against else 0.0

    shortlist = [r for r in sorted(rows, key=lambda r: -pressure(r)) if pressure(r) > 0]
    shortlist = shortlist[:candidates]
    if not shortlist:
        return {
            "source": "Binance USD-M futures",
            "scanned": len(rows),
            "shortlisted": 0,
            "ranked": [],
            "note": (
                "no contract currently has funding running against its price "
                "move — the precondition for a cascade is absent board-wide"
            ),
            "disclaimer": "For educational/informational purposes only. Not financial advice.",
        }

    details: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = [pool.submit(_detail, r["symbol"], bn_period) for r in shortlist]
        for fut in as_completed(futures):
            symbol, data = fut.result()
            if data:
                details[symbol] = data

    graded: List[dict] = []
    unavailable: List[str] = []
    for r in shortlist:
        d = details.get(r["symbol"])
        if not d:
            unavailable.append(r["symbol"])
            continue
        crowd_ratio = d["crowd_ratio"]
        crowding = (
            "CROWDED_LONG" if crowd_ratio is not None and crowd_ratio >= ps._CROWDED_LONG_RATIO
            else "CROWDED_SHORT" if crowd_ratio is not None and crowd_ratio <= ps._CROWDED_SHORT_RATIO
            else "BALANCED"
        )
        funding_pct = r["funding_pct_per_interval"]
        funding_state = (
            "EXPENSIVE_LONGS" if funding_pct >= ps._FUNDING_RICH_PCT
            else "LONGS_PAYING" if funding_pct >= ps._FUNDING_ELEVATED_PCT
            else "SHORTS_PAYING" if funding_pct <= -ps._FUNDING_ELEVATED_PCT
            else "NEUTRAL"
        )
        oi_change = d["oi_change_pct"]
        oi_state = (
            "BUILDING" if oi_change is not None and oi_change >= ps._OI_SHIFT_PCT
            else "UNWINDING" if oi_change is not None and oi_change <= -ps._OI_SHIFT_PCT
            else "STABLE"
        )
        divergence = (
            None if d["crowd_long"] is None or d["top_long"] is None
            else round((d["crowd_long"] - d["top_long"]) * 100.0, 1)
        )
        grade, why, factors = ps._squeeze_risk(
            crowding, funding_state, oi_state, r.get("price_change_24h_pct"), divergence
        )
        graded.append({
            "symbol": r["symbol"],
            "squeeze_risk": grade,
            "reasoning": why,
            "factors": factors,
            "crowding": crowding,
            "crowd_long_pct": None if d["crowd_long"] is None else round(d["crowd_long"] * 100, 2),
            "top_trader_long_pct": None if d["top_long"] is None else round(d["top_long"] * 100, 2),
            "crowd_vs_top_divergence_pts": divergence,
            "funding_pct_per_interval": funding_pct,
            "funding_annualised_pct": r["funding_annualised_pct"],
            "funding_interval_hours": r["funding_interval_hours"],
            "funding_state": funding_state,
            "oi_change_pct": None if oi_change is None else round(oi_change, 2),
            "oi_state": oi_state,
            "price_change_24h_pct": r.get("price_change_24h_pct"),
            "mark_price": r["mark_price"],
        })

    rank = {"ELEVATED_LONG_SQUEEZE": 0, "ELEVATED_SHORT_SQUEEZE": 0,
            "WATCH_LONG_SQUEEZE": 1, "WATCH_SHORT_SQUEEZE": 1, "LOW": 2}
    graded.sort(key=lambda g: (rank.get(g["squeeze_risk"], 3), -len(g["factors"])))

    # Every elevated name, by symbol only. This used to carry whole
    # rows: at limit=20 that was 5,753 of 21,710 characters repeating
    # rows already in `ranked`, byte for byte. The list exists so an
    # elevated grade cannot hide past the limit, and a symbol does
    # that job - `ranked` holds the detail for the rows it shows, and
    # `elevated_beyond_limit` says plainly when there are more.
    ranked = graded[:limit]
    elevated = [g["symbol"] for g in graded
                if g["squeeze_risk"].startswith("ELEVATED")]
    shown = {g["symbol"] for g in ranked}
    beyond = [s for s in elevated if s not in shown]

    return {
        "source": "Binance USD-M futures (3 bulk requests + 3 per shortlisted symbol)",
        "period": period,
        "scanned": len(rows),
        "shortlisted": len(shortlist),
        "graded": len(graded),
        "elevated": elevated or None,
        "elevated_beyond_limit": (
            {"symbols": beyond,
             "note": f"graded ELEVATED but outside limit={limit} - raise "
                     "limit to see their rows"}
            if beyond else None
        ),
        "ranked": ranked,
        "unavailable": sorted(unavailable) or None,
        "shortlist_rule": (
            "funding running against the 24h price move, ranked by "
            "|funding| x |change|; crowding is then measured, not assumed"
        ),
        "reading_low": (
            "A LOW grade here does NOT mean nothing is happening — every row "
            "reached the shortlist because funding is running against the "
            "price move. LOW means the account split is not one-sided enough "
            "to call a crowd (see thresholds.crowded_long_ratio), so there is "
            "no concentrated position for a cascade to feed on. Read the row's "
            "funding and oi_change_pct regardless."
        ),
        "thresholds": {
            "crowded_long_ratio": ps._CROWDED_LONG_RATIO,
            "crowded_short_ratio": ps._CROWDED_SHORT_RATIO,
            "funding_elevated_pct": ps._FUNDING_ELEVATED_PCT,
            "oi_shift_pct": ps._OI_SHIFT_PCT,
            "squeeze_factors_for_elevated": ps._SQUEEZE_ELEVATED_FACTORS,
        },
        "filters": {"min_quote_volume_usd": min_quote_volume_usd},
        "display_hint": display_hints.SQUEEZE,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
