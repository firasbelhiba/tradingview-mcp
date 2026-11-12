"""
Copy Trading Service — lead traders, with the risk the leaderboard hides.

OKX publishes its copy-trading leaderboard without a key. Binance's futures
leaderboard endpoint is gone (404 as of this writing), so OKX is the source
for individual traders; the *aggregate* top-trader positioning this server
already reports comes from Binance and is a different, complementary thing.

**Why this module computes its own risk metrics.** The leaderboard row gives
return, win rate, follower count and AUM — and no drawdown, no leverage and
no Sharpe. Ranked on return alone it is a list of whoever ran the most
leverage and survived. The first live run made the point: the top-ranked
trader advertised +629.6% with a 63.6% win rate, and their closed trades show
an average of 22.8x leverage, a peak of 50x, and one position that lost 262%
of its margin.

Those numbers are derivable. ``public-subpositions-history`` carries ``lever``
and ``pnlRatio`` per closed trade and ``public-weekly-pnl`` gives a weekly
series, so this module measures leverage, the worst single trade, losing
weeks and a drawdown from the weekly curve, and attaches ``risk_flags``.

**No ranking score is invented.** Choosing who to follow is a decision about
someone's money, so the tool reports measured facts and flags, offers several
explicit sort orders, and leaves the judgement to the reader. Nothing here
places, follows or sizes a trade.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import display_hints

_BASE = "https://www.okx.com/api/v5/copytrading/"
_UA = "Mozilla/5.0 (compatible; tradingview-mcp/0.9.0)"
_TIMEOUT = 20
_CACHE_TTL_S = 300.0
_WORKERS = 5

_cache: Dict[str, Tuple[float, Any]] = {}
_cache_lock = Lock()

SORT_ORDERS = ("return", "track_record", "lowest_leverage", "win_rate", "followers")
INSTRUMENT_TYPES = ("SWAP", "SPOT")

# Thresholds behind the risk flags, reported with every result.
_HIGH_LEVERAGE = 10.0        # average leverage across closed trades
_EXTREME_LEVERAGE = 25.0
_SHORT_TRACK_RECORD_DAYS = 180
_CATASTROPHIC_TRADE_PCT = -50.0   # a single closed trade losing half its margin
_MAX_ENRICH = 15             # traders to pull trade history for


def _ttl() -> float:
    try:
        return max(0.0, float(os.environ.get("TRADINGVIEW_MCP_COPYTRADING_TTL",
                                             _CACHE_TTL_S)))
    except ValueError:
        return _CACHE_TTL_S


def _get(path: str) -> Any:
    url = _BASE + path
    with _cache_lock:
        hit = _cache.get(url)
    if hit and (time.time() - hit[0]) <= _ttl():
        return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = {}
        raise RuntimeError(
            f"OKX returned HTTP {exc.code} {body.get('code', '')} "
            f"{str(body.get('msg', ''))[:80]}"
        ) from exc
    if str(payload.get("code")) not in ("0", "None", ""):
        raise RuntimeError(
            f"OKX error {payload.get('code')}: {str(payload.get('msg'))[:80]}"
        )
    with _cache_lock:
        _cache[url] = (time.time(), payload)
    return payload


def _f(value: Any) -> Optional[float]:
    """Parse a numeric field.

    OKX returns leverage as strings in scientific notation ("2E+1" for 20x),
    so this must go through float() rather than int() or a string compare.
    """
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _rows(payload: Any) -> List[dict]:
    data = payload.get("data") or []
    if data and isinstance(data[0], dict) and "ranks" in data[0]:
        return data[0].get("ranks") or []
    return [r for r in data if isinstance(r, dict)]


# ── Derived risk ──────────────────────────────────────────────────────────────

def _max_drawdown(weekly_pnl: List[float]) -> Optional[float]:
    """Peak-to-trough decline of the cumulative weekly PnL curve.

    Weekly granularity, so this understates a drawdown that opened and closed
    inside one week — it is a floor on the real figure, not the real figure,
    and the result says so.
    """
    if not weekly_pnl:
        return None
    cumulative, peak, worst = 0.0, 0.0, 0.0
    for pnl in weekly_pnl:
        cumulative += pnl
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return worst


# OKX's code for "position-level data is not available for this trader". It
# reads "This futures lead trader doesn't exist", which is not true of a
# trader sitting on OKX's own leaderboard with live aggregate statistics.
_HIDDEN_CODE = "59269"

HIDDEN_NOTE = (
    "OKX withholds this trader's position-level data (error 59269) while "
    "still publishing their aggregate statistics. Measured live, 3 of the "
    "top 20 lead traders are like this. Their open positions, per-trade "
    "leverage and worst trade cannot be seen by anyone who is not copying "
    "them - which is 'cannot be verified', not 'nothing to see'."
)


def _is_hidden(exc: Exception) -> bool:
    return _HIDDEN_CODE in str(exc)


def _okx_stats(code: str, inst_type: str) -> Optional[dict]:
    """OKX's own one-year aggregates, which answer even for hidden traders.

    These are published figures, not measurements: they cannot be checked
    against trades that OKX will not show. They are still more than nothing
    for a trader whose history is withheld.
    """
    rows = _rows(_get(
        f"public-stats?instType={inst_type}&uniqueCode={code}&lastDays=4"
    ))
    if not rows:
        return None
    r = rows[0]
    win = _f(r.get("winRatio"))
    profit, loss = _f(r.get("profitDays")), _f(r.get("lossDays"))
    return {
        "window": "last 365 days",
        "win_rate_pct": None if win is None else round(win * 100, 1),
        "profit_days": None if profit is None else int(profit),
        "loss_days": None if loss is None else int(loss),
        "avg_position_notional": (
            None if _f(r.get("avgSubPosNotional")) is None
            else round(_f(r.get("avgSubPosNotional")), 2)
        ),
        "invested_amount": (
            None if _f(r.get("investAmt")) is None
            else round(_f(r.get("investAmt")), 2)
        ),
        "copiers_pnl": (
            None if _f(r.get("curCopyTraderPnl")) is None
            else round(_f(r.get("curCopyTraderPnl")), 2)
        ),
        "source": "published by OKX, not measured from trades",
    }


def _trade_stats(history: List[dict]) -> dict:
    levers = [_f(t.get("lever")) for t in history]
    levers = [v for v in levers if v]
    ratios = [_f(t.get("pnlRatio")) for t in history]
    ratios = [v for v in ratios if v is not None]
    instruments: Dict[str, int] = {}
    for t in history:
        inst = t.get("instId")
        if inst:
            instruments[inst] = instruments.get(inst, 0) + 1
    wins = sum(1 for v in ratios if v > 0)
    return {
        "closed_trades_sampled": len(history),
        "avg_leverage": round(sum(levers) / len(levers), 1) if levers else None,
        "max_leverage": round(max(levers), 0) if levers else None,
        "measured_win_rate_pct": (
            round(wins / len(ratios) * 100, 1) if ratios else None
        ),
        "worst_trade_pct": round(min(ratios) * 100, 1) if ratios else None,
        "best_trade_pct": round(max(ratios) * 100, 1) if ratios else None,
        "most_traded": sorted(instruments.items(), key=lambda kv: -kv[1])[:3] or None,
    }


def _risk_flags(trader: dict, stats: dict, weekly: dict) -> List[str]:
    flags: List[str] = []
    avg_lev = stats.get("avg_leverage")
    if avg_lev is not None:
        if avg_lev >= _EXTREME_LEVERAGE:
            flags.append(f"extreme leverage — {avg_lev}x average across sampled trades")
        elif avg_lev >= _HIGH_LEVERAGE:
            flags.append(f"high leverage — {avg_lev}x average")
    worst = stats.get("worst_trade_pct")
    if worst is not None and worst <= _CATASTROPHIC_TRADE_PCT:
        flags.append(f"a single closed trade lost {abs(worst):.0f}% of its margin")
    days = _f(trader.get("leadDays"))
    if days is not None and days < _SHORT_TRACK_RECORD_DAYS:
        flags.append(
            f"short track record — {days:.0f} days, too little to separate "
            "skill from a favourable streak"
        )
    advertised = _f(trader.get("winRatio"))
    measured = stats.get("measured_win_rate_pct")
    if advertised is not None and measured is not None:
        gap = measured - advertised * 100
        if abs(gap) >= 15:
            flags.append(
                f"advertised win rate {advertised * 100:.0f}% vs "
                f"{measured:.0f}% over the sampled trades"
            )
    losing = weekly.get("losing_weeks")
    total = weekly.get("weeks_sampled")
    if losing is not None and total:
        if losing / total >= 0.5:
            flags.append(f"{losing} of {total} sampled weeks were losing weeks")
    return flags


def _weekly_stats(weekly: List[dict]) -> dict:
    values = [_f(w.get("pnl")) for w in weekly]
    values = [v for v in values if v is not None]
    if not values:
        # The same keys as a populated result. A block whose shape depends on
        # whether a fetch succeeded makes every consumer guess, and the first
        # live run raised KeyError on exactly that.
        return {"weeks_sampled": 0, "losing_weeks": None,
                "best_week_usd": None, "worst_week_usd": None,
                "max_drawdown_usd": None, "drawdown_note": None}
    # OKX returns newest first; a drawdown curve has to run forward in time.
    chronological = list(reversed(values))
    return {
        "weeks_sampled": len(values),
        "losing_weeks": sum(1 for v in values if v < 0),
        "best_week_usd": round(max(values), 2),
        "worst_week_usd": round(min(values), 2),
        "max_drawdown_usd": round(_max_drawdown(chronological) or 0.0, 2),
        "drawdown_note": (
            "measured on the WEEKLY curve, so a drawdown that opened and "
            "closed inside one week is invisible — treat it as a floor"
        ),
    }


def _enrich(trader: dict, inst_type: str, history_limit: int) -> dict:
    code = trader.get("uniqueCode")
    # Defaults come from the same functions that build the populated versions,
    # so the result's shape never depends on whether a fetch succeeded.
    stats: dict = _trade_stats([])
    weekly: dict = _weekly_stats([])
    errors: List[str] = []
    # measured / hidden / unknown. "hidden" is OKX withholding the data for
    # this trader; "unknown" is a request that failed and might not next time.
    history_state = "unknown"
    okx_stats = None
    if code:
        try:
            history = _rows(_get(
                f"public-subpositions-history?instType={inst_type}"
                f"&uniqueCode={code}&limit={history_limit}"
            ))
            stats = _trade_stats(history)
            history_state = "measured"
        except Exception as exc:  # noqa: BLE001
            if _is_hidden(exc):
                history_state = "hidden"
                errors.append("trade history hidden by OKX for this trader")
            else:
                errors.append(f"trade history unavailable: {str(exc)[:60]}")
        try:
            okx_stats = _okx_stats(code, inst_type)
        except Exception:  # noqa: BLE001
            okx_stats = None
        try:
            weekly = _weekly_stats(_rows(_get(
                f"public-weekly-pnl?instType={inst_type}&uniqueCode={code}"
            )))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"weekly pnl unavailable: {str(exc)[:60]}")

    pnl_ratio = _f(trader.get("pnlRatio"))
    days = _f(trader.get("leadDays"))
    out = {
        "nickname": trader.get("nickName"),
        "unique_code": code,
        # OKX's portLink is the trader's avatar image, not a profile page —
        # naming it profile_url invited a click that goes to a .png.
        "avatar_url": trader.get("portLink"),
        "advertised": {
            "return_pct": None if pnl_ratio is None else round(pnl_ratio * 100, 1),
            "win_rate_pct": (
                None if _f(trader.get("winRatio")) is None
                else round(_f(trader.get("winRatio")) * 100, 1)
            ),
            "pnl": _f(trader.get("pnl")),
            "lead_days": None if days is None else int(days),
            "followers": int(_f(trader.get("copyTraderNum")) or 0),
            "followers_all_time": int(_f(trader.get("accCopyTraderNum")) or 0),
            "aum_usd": (
                None if _f(trader.get("aum")) is None else round(_f(trader.get("aum")), 2)
            ),
            # Seats, not copyState: that field is "0" for every trader on
            # an anonymous request because it describes the CALLER's copy
            # relationship. Read as "accepting followers" it said no trader
            # on OKX accepts anyone.
            "max_followers": (
                None if _f(trader.get("maxCopyTraderNum")) is None
                else int(_f(trader.get("maxCopyTraderNum")))
            ),
            "accepting_followers": (
                None if _f(trader.get("maxCopyTraderNum")) is None
                else (_f(trader.get("copyTraderNum")) or 0)
                < _f(trader.get("maxCopyTraderNum"))
            ),
        },
        "measured": stats,
        "trade_history_state": history_state,
        "okx_published_stats": okx_stats,
        "weekly": weekly,
        # Return per day of track record: +600% over 60 days and +600% over
        # 600 days rank identically on raw return and are not the same claim.
        "return_pct_per_day": (
            None if pnl_ratio is None or not days
            else round(pnl_ratio * 100 / days, 3)
        ),
    }
    out["risk_flags"] = _risk_flags(trader, stats, weekly)
    if history_state == "hidden":
        out["hidden_note"] = HIDDEN_NOTE
        out["risk_flags"].append(
            "position-level data is hidden, so leverage and worst trade "
            "cannot be verified"
        )
    if errors:
        out["incomplete"] = errors
    return out


# ── Tools ─────────────────────────────────────────────────────────────────────

def list_lead_traders(
    sort: str = "track_record",
    limit: int = 10,
    instrument_type: str = "SWAP",
    min_lead_days: Optional[int] = None,
    max_avg_leverage: Optional[float] = None,
    history_limit: int = 20,
) -> dict:
    """Lead traders from OKX's public copy-trading leaderboard, with measured risk.

    The leaderboard alone publishes no leverage and no drawdown, so each
    candidate's closed trades and weekly PnL are fetched and the missing
    metrics computed. Default sort is ``track_record``, not return — ranking
    strangers by return is how you end up copying the most leveraged survivor.

    Args:
        sort:             ``track_record``, ``return``, ``lowest_leverage``,
                          ``win_rate`` or ``followers``.
        limit:            Traders to return, 1-15 (each is 2 extra requests).
        instrument_type:  ``SWAP`` or ``SPOT``.
        min_lead_days:    Drop traders with a shorter track record.
        max_avg_leverage: Drop traders whose measured average leverage exceeds
                          this.
        history_limit:    Closed trades to sample per trader, 5-100.
    """
    if sort not in SORT_ORDERS:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"Unknown sort {sort!r}",
            valid_sorts=list(SORT_ORDERS),
        )
    if instrument_type not in INSTRUMENT_TYPES:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"instrument_type must be one of {', '.join(INSTRUMENT_TYPES)}",
        )
    if not 1 <= limit <= _MAX_ENRICH:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"limit must be 1-{_MAX_ENRICH}, got {limit}"
        )
    if not 5 <= history_limit <= 100:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"history_limit must be 5-100, got {history_limit}",
        )

    try:
        candidates = _rows(_get(
            f"public-lead-traders?instType={instrument_type}&limit={_MAX_ENRICH}"
        ))
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"OKX unreachable: {str(exc)[:100]}",
            retryable=True,
        )
    if not candidates:
        return make_error(
            ErrorCode.NO_DATA, "OKX returned no lead traders", retryable=True
        )

    if min_lead_days is not None:
        candidates = [
            c for c in candidates if (_f(c.get("leadDays")) or 0) >= min_lead_days
        ]
        if not candidates:
            return make_error(
                ErrorCode.NO_DATA,
                f"No lead trader has a track record of {min_lead_days}+ days",
                min_lead_days=min_lead_days,
            )

    enriched: List[dict] = []
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = [
            pool.submit(_enrich, c, instrument_type, history_limit)
            for c in candidates
        ]
        for fut in as_completed(futures):
            try:
                enriched.append(fut.result())
            except Exception:  # noqa: BLE001
                continue

    filtered_out = 0
    if max_avg_leverage is not None:
        kept = []
        for row in enriched:
            lev = row["measured"].get("avg_leverage")
            if lev is not None and lev > max_avg_leverage:
                filtered_out += 1
                continue
            kept.append(row)
        enriched = kept

    if not enriched:
        return make_error(
            ErrorCode.NO_DATA,
            "No lead trader passed the filters",
            max_avg_leverage=max_avg_leverage, min_lead_days=min_lead_days,
        )

    def key(row: dict):
        adv, meas = row["advertised"], row["measured"]
        if sort == "return":
            return -(adv.get("return_pct") or -1e9)
        if sort == "win_rate":
            return -(meas.get("measured_win_rate_pct")
                     or adv.get("win_rate_pct") or -1e9)
        if sort == "followers":
            return -(adv.get("followers") or 0)
        if sort == "lowest_leverage":
            lev = meas.get("avg_leverage")
            return lev if lev is not None else 1e9
        return -(adv.get("lead_days") or 0)  # track_record

    enriched.sort(key=key)
    returned = enriched[:limit]
    # The leaderboard publishes return and hides leverage; this tool
    # exists to measure what it hides. When that measurement comes
    # back empty the row still carries the advertised figures, and a
    # null leverage beside measured ones reads as a low one. Name them.
    unmeasured = [r["nickname"] for r in returned
                  if (r.get("measured") or {}).get("avg_leverage")
                  is None]
    return {
        "source": "OKX copy trading, public endpoints (no key)",
        "instrument_type": instrument_type,
        "sort": sort,
        "candidates_seen": len(candidates),
        "returned": len(returned),
        "excluded_by_leverage_filter": filtered_out or None,
        "traders": returned,
        "unmeasured": (
            {"traders": unmeasured,
             "count": len(unmeasured),
             "of_returned": len(returned),
             "hidden_by_okx": [
                 r["nickname"] for r in returned
                 if r.get("trade_history_state") == "hidden"
             ] or None,
             "meaning": (
                 "Their closed-trade history did not come back, so "
                 "avg_leverage, max_leverage, worst_trade_pct and "
                 "measured_win_rate_pct are null for them. A blank leverage is "
                 "MISSING, not low — do not read these rows as the "
                 "conservative ones. Ask again, or call lead_trader_profile "
                 "with a larger history_limit."
             )}
            if unmeasured else None
        ),
        "thresholds": {
            "high_leverage": _HIGH_LEVERAGE,
            "extreme_leverage": _EXTREME_LEVERAGE,
            "short_track_record_days": _SHORT_TRACK_RECORD_DAYS,
            "catastrophic_trade_pct": _CATASTROPHIC_TRADE_PCT,
        },
        "what_this_cannot_tell_you": [
            "whether a record reflects skill or a favourable streak — leaderboards "
            "show survivors, and traders who blew up are absent by construction",
            "the leverage on trades older than the sampled window",
            "intra-week drawdown, since the PnL series is weekly",
            "anything about future results",
        ],
        "read_only_note": (
            "This reports published statistics. It does not place, follow or "
            "size any trade, and it is not a recommendation."
        ),
        "display_hint": display_hints.LEAD_TRADERS,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


def get_lead_trader(unique_code: str, instrument_type: str = "SWAP",
                    history_limit: int = 50) -> dict:
    """Full risk profile for one lead trader, including open positions.

    Args:
        unique_code:     The trader's ``uniqueCode`` from ``list_lead_traders``.
        instrument_type: ``SWAP`` or ``SPOT``.
        history_limit:   Closed trades to sample, 5-100.
    """
    if not unique_code or not str(unique_code).strip():
        return make_error(ErrorCode.INVALID_PARAMETER, "unique_code is required")
    if instrument_type not in INSTRUMENT_TYPES:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"instrument_type must be one of {', '.join(INSTRUMENT_TYPES)}",
        )
    if not 5 <= history_limit <= 100:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"history_limit must be 5-100, got {history_limit}",
        )

    code = str(unique_code).strip()
    try:
        candidates = _rows(_get(
            f"public-lead-traders?instType={instrument_type}&limit={_MAX_ENRICH}"
        ))
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"OKX unreachable: {str(exc)[:100]}",
            retryable=True,
        )

    trader = next((c for c in candidates if c.get("uniqueCode") == code), None)
    if trader is None:
        # Not on the current leaderboard page — still enrich what is public,
        # rather than refusing a code the caller got from elsewhere.
        trader = {"uniqueCode": code}

    profile = _enrich(trader, instrument_type, history_limit)
    if not candidates or trader.get("nickName") is None:
        profile["note"] = (
            "This code is not on the leaderboard page fetched, so the "
            "advertised figures are unavailable; the measured figures below "
            "come from the trader's own public trade history."
        )

    open_positions: List[dict] = []
    # Whether the question was answered at all, kept apart from the answer.
    # OKX returns nothing for a code that is not a futures lead trader, and
    # a count of 0 beside that silence reads as "holds nothing" when the
    # truth is "nobody knows". Same distinction the alert layer keeps
    # between "not evaluated" and "condition not met".
    open_positions_known = True
    open_positions_hidden = False
    open_positions_reason = None
    try:
        for p in _rows(_get(
            f"public-current-subpositions?instType={instrument_type}"
            f"&uniqueCode={code}&limit=50"
        )):
            open_positions.append({
                "instrument": p.get("instId") or None,
                # OKX returns side, leverage, margin and unrealised PnL
                # for some traders while blanking the instrument, the
                # entry price and the size. Measured live: 5 of the top
                # 20, on top of the 3 hidden outright.
                "instrument_hidden": not p.get("instId"),
                "side": p.get("posSide"),
                "leverage": _f(p.get("lever")),
                "margin": _f(p.get("margin")),
                "open_price": _f(p.get("openAvgPx")),
                "mark_price": _f(p.get("markPx")),
                "margin_mode": p.get("mgnMode"),
                "opened_at": p.get("openTime"),
            })
    except Exception as exc:  # noqa: BLE001
        open_positions_known = False
        open_positions_hidden = _is_hidden(exc)
        open_positions_reason = str(exc)[:90]
        profile.setdefault("incomplete", []).append(
            "open positions hidden by OKX for this trader"
            if open_positions_hidden
            else f"open positions unavailable: {open_positions_reason[:60]}"
        )

    preferences: List[dict] = []
    preferences_known = True
    try:
        preferences = [
            {"currency": r.get("ccy"), "share_pct": (
                None if _f(r.get("ratio")) is None else round(_f(r.get("ratio")) * 100, 1)
            )}
            for r in _rows(_get(
                f"public-preference-currency?instType={instrument_type}&uniqueCode={code}"
            ))
        ]
    except Exception:  # noqa: BLE001
        preferences = []
        preferences_known = False

    if (profile.get("measured") or {}).get("closed_trades_sampled") == 0:
        profile.setdefault("incomplete", []).append(
            "no closed trades came back, so every measured figure is "
            "null - missing, not low"
        )
    open_levs = [p["leverage"] for p in open_positions if p.get("leverage")]
    profile["open_positions"] = open_positions or None
    # None, not 0, when nobody could tell. A reader that only looks at the
    # number must not be able to mistake silence for an empty book.
    profile["open_position_count"] = len(open_positions) if open_positions_known else None
    blanked = sum(1 for p in open_positions if p.get("instrument_hidden"))
    profile["open_positions_state"] = (
        "hidden" if open_positions_hidden
        else "unknown" if not open_positions_known
        else "instruments_hidden" if open_positions and blanked == len(open_positions)
        else "measured"
    )
    profile["open_positions_instrument_hidden"] = blanked or None
    if blanked:
        profile["instrument_hidden_note"] = (
            f"OKX shows that {blanked} of this trader's "
            f"{len(open_positions)} open positions exist - side, leverage, "
            "margin and unrealised PnL - but blanks WHICH instrument, the "
            "entry price and the size. Measured live, 5 of the top 20 lead "
            "traders are like this. Their leverage can be read; what they "
            "are holding cannot, so they cannot be mirrored."
        )
    if open_positions_hidden:
        profile["open_positions_note"] = (
            HIDDEN_NOTE + " This is NOT 'no open trades'. Retrying will not "
            "change it; pick a trader whose positions are visible if you "
            "need to see what they hold."
        )
    elif not open_positions_known:
        profile["open_positions_note"] = (
            f"OKX returned no position data ({open_positions_reason}). This "
            "is NOT 'no open trades' - it is 'nobody could tell'. It may "
            "work on a retry."
        )
    elif not open_positions:
        profile["open_positions_note"] = (
            "measured: OKX answered and this trader is holding nothing right now"
        )
    profile["open_max_leverage"] = round(max(open_levs), 0) if open_levs else None
    profile["instrument_preferences"] = preferences or None
    if not preferences_known:
        profile.setdefault("incomplete", []).append(
            "instrument preferences unavailable, so the mix below is missing "
            "rather than empty"
        )
    profile["source"] = "OKX copy trading, public endpoints (no key)"
    profile["read_only_note"] = (
        "This reports published statistics. It does not place, follow or size "
        "any trade, and it is not a recommendation."
    )
    profile["disclaimer"] = (
        "For educational/informational purposes only. Not financial advice."
    )
    return profile
