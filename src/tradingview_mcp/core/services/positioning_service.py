"""
Positioning Service — who is on which side of a perpetual, and what it costs.

Interpretation layer over ``binance_service``, which does the fetching.
The server's price-and-volume tools cannot see any of this: funding paid,
open interest building or unwinding, how the crowd is split, and how the
largest traders are split against it.

The crowd-versus-top-traders comparison is the point of the tool. A market
where 75% of accounts are long while the biggest traders sit at 68% is
positioned differently from one where both agree, and nothing derived from
price can tell those apart.

Every threshold used for a verdict is a named constant, reported back in the
result so a caller can check the label against the numbers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import binance_service as bn

# Written-down thresholds.
_FUNDING_RICH_PCT = 0.05      # per funding interval: longs paying a lot
_FUNDING_ELEVATED_PCT = 0.01  # per funding interval: longs paying meaningfully
_CROWDED_LONG_RATIO = 2.5     # long/short accounts
_CROWDED_SHORT_RATIO = 0.8
_OI_SHIFT_PCT = 5.0           # OI change worth calling a build or unwind
_DIVERGENCE_PTS = 5.0         # crowd vs top traders, percentage points


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _safe(fn, *args, **kwargs):
    """Run one read, returning ``(value, error)``.

    One dead feed must not sink the report — a funding rate is still worth
    having without the taker flow, and the result names what was missing.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        return None, str(exc)[:120]


def _oi_price_read(oi_change_pct: Optional[float],
                   price_change_pct: Optional[float]) -> Optional[dict]:
    """The standard open-interest/price quadrant.

    Rising open interest means contracts are being created — someone is
    opening rather than closing. Pairing that with price direction says which
    side is doing it, which a price chart alone cannot show.
    """
    if oi_change_pct is None or price_change_pct is None:
        return None
    rising_oi, rising_px = oi_change_pct > 0, price_change_pct > 0
    if rising_oi and rising_px:
        return {"state": "new_longs",
                "meaning": "open interest and price both rising — fresh longs, trend supported"}
    if rising_oi:
        return {"state": "new_shorts",
                "meaning": "open interest rising while price falls — fresh shorts, or longs adding into weakness"}
    if rising_px:
        return {"state": "short_covering",
                "meaning": "price rising while open interest falls — covering, not new buying"}
    return {"state": "long_liquidation",
            "meaning": "price and open interest both falling — longs closing or being liquidated"}


_SQUEEZE_ELEVATED_FACTORS = 3   # of the four below
_SQUEEZE_WATCH_FACTORS = 2


def _squeeze_risk(crowding: str, funding_state: str, oi_state: str,
                  price_change: Optional[float],
                  divergence: Optional[float]) -> tuple:
    """Grade liquidation-cascade risk by counting independent factors.

    For a long squeeze the factors are: the crowd is long, price is falling,
    open interest is building (new positions into weakness), longs are paying
    funding, and the crowd is more long than the largest traders. None is
    required — three or more is ELEVATED, two is WATCH.

    Counting beats gating. Requiring expensive funding would miss the common
    case where crowded longs are already underwater before the funding market
    reprices.
    """
    if crowding not in ("CROWDED_LONG", "CROWDED_SHORT"):
        return ("LOW", "positioning is not one-sided enough to fuel a cascade", [])

    long_side = crowding == "CROWDED_LONG"
    factors: List[str] = ["the crowd is long" if long_side else "the crowd is short"]

    # Price moving against the crowd is what turns crowding into pressure, so
    # it gates the ELEVATED grade: crowded longs in a RISING market are not in
    # trouble yet, however much funding they pay.
    adverse = (
        None if price_change is None
        else (price_change < 0) if long_side else (price_change > 0)
    )
    if adverse:
        factors.append("price is falling" if long_side else "price is rising")
    if oi_state == "BUILDING":
        factors.append("open interest is building into the move")
    if long_side and funding_state in ("LONGS_PAYING", "EXPENSIVE_LONGS"):
        factors.append("longs are paying funding to hold")
    if not long_side and funding_state == "SHORTS_PAYING":
        factors.append("shorts are paying funding to hold")
    if divergence is not None:
        if long_side and divergence >= _DIVERGENCE_PTS:
            factors.append("the crowd is more long than the largest traders")
        elif not long_side and divergence <= -_DIVERGENCE_PTS:
            factors.append("the crowd is more short than the largest traders")

    side = "LONG" if long_side else "SHORT"
    if adverse and len(factors) >= _SQUEEZE_ELEVATED_FACTORS:
        why = (
            "crowded longs with the move against them — a liquidation cascade "
            "setup, not a contrarian long" if long_side else
            "crowded shorts with the move against them — upside can force covering"
        )
        return (f"ELEVATED_{side}_SQUEEZE", why, factors)
    if adverse or len(factors) >= _SQUEEZE_ELEVATED_FACTORS:
        why = (
            "crowded longs; a sharp drop would force exits" if long_side else
            "crowded shorts; a sharp rally would force covering"
        )
        return (f"WATCH_{side}_SQUEEZE", why, factors)
    return ("LOW",
            f"the crowd leans {'long' if long_side else 'short'} but nothing "
            "else is pressuring it", factors)


def analyze_positioning(symbol: str, period: str = "4h", history_bars: int = 30) -> dict:
    """Positioning and funding for one perpetual contract.

    Args:
        symbol:       Pair symbol, e.g. ``BTCUSDT``. Must list as a Binance
                      perpetual — most pairs do not, and spot-only pairs
                      return a SYMBOL_NOT_FOUND envelope with suggestions.
        period:       Sampling interval for the ratio and OI histories.
        history_bars: History points to fetch, 2-500.
    """
    if not 2 <= history_bars <= 500:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"history_bars must be 2-500, got {history_bars}"
        )

    bn_period = bn.binance_period_for(period)
    if not bn_period:
        return make_error(
            ErrorCode.INVALID_TIMEFRAME,
            f"period {period!r} has no Binance equivalent",
            supported=bn.supported_periods(),
        )

    perp = bn.perp_symbol_for(symbol)
    try:
        universe = bn.perpetual_symbols()
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Binance unreachable: {str(exc)[:120]}",
            symbol=perp, retryable=True,
        )

    if perp not in universe:
        stem = perp[:-4] if perp.endswith("USDT") else perp[:3]
        return make_error(
            ErrorCode.SYMBOL_NOT_FOUND,
            f"{perp} does not list as a Binance perpetual. Positioning data "
            "exists only for perpetual contracts, not spot pairs.",
            symbol=perp,
            suggestions=[s for s in universe if s.startswith(stem[:3])][:5],
            perpetuals_available=len(universe),
        )

    premium, e_prem = _safe(bn.get_premium_index, perp)
    oi_now, e_oi = _safe(bn.get_open_interest, perp)
    oi_hist, e_oih = _safe(bn.get_open_interest_history, perp, bn_period, history_bars)
    fund_hist, e_fh = _safe(bn.get_funding_history, perp, min(history_bars, 100))
    crowd, e_crowd = _safe(bn.get_global_long_short, perp, bn_period, 1)
    tops, e_tops = _safe(bn.get_top_trader_positions, perp, bn_period, 1)
    taker, e_taker = _safe(bn.get_taker_flow, perp, bn_period, 1)
    ticker, e_tick = _safe(bn.get_ticker_24h, perp)

    if premium is None and oi_now is None:
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            "Binance returned nothing for the core feeds (funding and open interest)",
            symbol=perp, detail=e_prem or e_oi, retryable=True,
        )

    mark = _f((premium or {}).get("markPrice"))
    raw_rate = _f((premium or {}).get("lastFundingRate"))
    interval_h = bn.infer_funding_interval_hours(fund_hist or [])

    funding_pct = None if raw_rate is None else raw_rate * 100.0
    per_day = None if funding_pct is None else funding_pct * (24.0 / interval_h)
    annualised = None if per_day is None else per_day * 365.0

    recent = [
        _f(h.get("fundingRate")) for h in (fund_hist or [])[-8:]
        if _f(h.get("fundingRate")) is not None
    ]
    mean_recent_pct = None if not recent else (sum(recent) / len(recent)) * 100.0

    oi_base = _f((oi_now or {}).get("openInterest"))
    oi_notional = None if (oi_base is None or mark is None) else oi_base * mark
    oi_series = [
        _f(h.get("sumOpenInterest")) for h in (oi_hist or [])
        if _f(h.get("sumOpenInterest")) is not None
    ]
    oi_change_pct = (
        None if len(oi_series) < 2 or not oi_series[0]
        else (oi_series[-1] - oi_series[0]) / oi_series[0] * 100.0
    )

    crowd_row = (crowd or [{}])[-1] if crowd else {}
    crowd_ratio = _f(crowd_row.get("longShortRatio"))
    crowd_long = _f(crowd_row.get("longAccount"))
    top_row = (tops or [{}])[-1] if tops else {}
    top_ratio = _f(top_row.get("longShortRatio"))
    top_long = _f(top_row.get("longAccount"))
    taker_ratio = _f(((taker or [{}])[-1] if taker else {}).get("buySellRatio"))
    price_change_24h = _f((ticker or {}).get("priceChangePercent"))

    divergence = (
        None if crowd_long is None or top_long is None
        else round((crowd_long - top_long) * 100.0, 1)
    )

    reasons: List[str] = []

    crowding = "BALANCED"
    if crowd_ratio is not None:
        if crowd_ratio >= _CROWDED_LONG_RATIO:
            crowding = "CROWDED_LONG"
            reasons.append(
                f"{crowd_long * 100:.1f}% of accounts are long (ratio {crowd_ratio:.2f})"
                if crowd_long is not None else f"long/short ratio {crowd_ratio:.2f}"
            )
        elif crowd_ratio <= _CROWDED_SHORT_RATIO:
            crowding = "CROWDED_SHORT"
            reasons.append(f"long/short ratio {crowd_ratio:.2f} — the crowd is short")

    funding_state = "NEUTRAL"
    if funding_pct is not None:
        if funding_pct >= _FUNDING_RICH_PCT:
            funding_state = "EXPENSIVE_LONGS"
            reasons.append(
                f"longs pay {funding_pct:.4f}% per {interval_h:.0f}h "
                f"({annualised:.0f}%/yr annualised)"
            )
        elif funding_pct >= _FUNDING_ELEVATED_PCT:
            funding_state = "LONGS_PAYING"
            reasons.append(f"longs pay {funding_pct:.4f}% per {interval_h:.0f}h")
        elif funding_pct <= -_FUNDING_ELEVATED_PCT:
            funding_state = "SHORTS_PAYING"
            reasons.append(f"shorts pay {abs(funding_pct):.4f}% per {interval_h:.0f}h")

    oi_state = "STABLE"
    if oi_change_pct is not None:
        if oi_change_pct >= _OI_SHIFT_PCT:
            oi_state = "BUILDING"
            reasons.append(f"open interest up {oi_change_pct:.1f}% over the window")
        elif oi_change_pct <= -_OI_SHIFT_PCT:
            oi_state = "UNWINDING"
            reasons.append(f"open interest down {abs(oi_change_pct):.1f}% over the window")

    if divergence is not None and abs(divergence) >= _DIVERGENCE_PTS:
        side = "less long than" if divergence > 0 else "more long than"
        reasons.append(f"top traders are {side} the crowd by {abs(divergence):.1f} points")

    # Squeeze risk accumulates from independent factors rather than requiring
    # any single one. An earlier version made funding cost a prerequisite, so
    # a market with 75% of accounts long, open interest building and price
    # falling scored LOW purely because funding had not yet turned expensive —
    # the cascade setup was already there. Funding is an amplifier, not a gate.
    squeeze, squeeze_why, squeeze_factors = _squeeze_risk(
        crowding, funding_state, oi_state, price_change_24h, divergence
    )

    unavailable = {
        k: v for k, v in {
            "premium_index": e_prem, "open_interest": e_oi,
            "open_interest_history": e_oih, "funding_history": e_fh,
            "global_long_short": e_crowd, "top_trader_positions": e_tops,
            "taker_flow": e_taker, "ticker_24h": e_tick,
        }.items() if v
    }

    result: Dict[str, Any] = {
        "symbol": perp,
        "requested_symbol": (symbol or "").upper(),
        "period": period,
        "binance_period": bn_period,
        "source": "Binance USD-M futures (public, unauthenticated)",
        "mark_price": mark,
        "index_price": _f((premium or {}).get("indexPrice")),
        "price_change_24h_pct": price_change_24h,
        "funding": {
            "rate": raw_rate,
            "pct_per_interval": None if funding_pct is None else round(funding_pct, 6),
            "interval_hours": interval_h,
            "interval_source": (
                "measured from funding history timestamps" if fund_hist
                else "assumed 8h — no funding history available"
            ),
            "pct_per_day": None if per_day is None else round(per_day, 4),
            "annualised_pct": None if annualised is None else round(annualised, 1),
            "mean_recent_pct": None if mean_recent_pct is None else round(mean_recent_pct, 6),
            "payer": (
                None if funding_pct is None
                else "longs pay shorts" if funding_pct > 0
                else "shorts pay longs" if funding_pct < 0 else "flat"
            ),
            "next_funding_time": (premium or {}).get("nextFundingTime"),
        },
        "open_interest": {
            "base_units": oi_base,
            "notional_usd": None if oi_notional is None else round(oi_notional, 2),
            "change_pct": None if oi_change_pct is None else round(oi_change_pct, 2),
            "history_points": len(oi_series),
            "state": oi_state,
        },
        "crowd": {
            "long_short_ratio": crowd_ratio,
            "long_pct": None if crowd_long is None else round(crowd_long * 100, 2),
            "short_pct": None if crowd_long is None else round((1 - crowd_long) * 100, 2),
            "basis": "all Binance futures accounts, equally weighted",
        },
        "top_traders": {
            "long_short_ratio": top_ratio,
            "long_pct": None if top_long is None else round(top_long * 100, 2),
            "basis": "top accounts by position size",
        },
        "crowd_vs_top_divergence_pts": divergence,
        "taker_flow": {
            "buy_sell_ratio": taker_ratio,
            "reading": (
                None if taker_ratio is None
                else "aggressive buying" if taker_ratio > 1.05
                else "aggressive selling" if taker_ratio < 0.95 else "balanced"
            ),
        },
        "oi_price_read": _oi_price_read(oi_change_pct, price_change_24h),
        "assessment": {
            "crowding": crowding,
            "funding": funding_state,
            "open_interest": oi_state,
            "squeeze_risk": squeeze,
            "squeeze_reasoning": squeeze_why,
            "squeeze_factors": squeeze_factors,
            "reasons": reasons or ["no threshold was crossed"],
        },
        "thresholds": {
            "funding_elevated_pct_per_interval": _FUNDING_ELEVATED_PCT,
            "funding_rich_pct_per_interval": _FUNDING_RICH_PCT,
            "crowded_long_ratio": _CROWDED_LONG_RATIO,
            "crowded_short_ratio": _CROWDED_SHORT_RATIO,
            "oi_shift_pct": _OI_SHIFT_PCT,
            "divergence_pts": _DIVERGENCE_PTS,
            "squeeze_factors_for_elevated": _SQUEEZE_ELEVATED_FACTORS,
            "squeeze_factors_for_watch": _SQUEEZE_WATCH_FACTORS,
        },
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
    if unavailable:
        result["unavailable_feeds"] = unavailable
    return result
