"""
Paper Copy — mirror a lead trader's open positions into a simulated wallet.

The leaderboard tools find a trader and measure the risk the leaderboard
hides. This takes the next step and opens what they are holding, in a
practice wallet, so the idea can be watched costing nothing.

It is deliberately not called copy trading, because it is not:

  * **You enter at today's price, not theirs.** A real platform mirrors you
    in at the moment they open. A leaderboard is a snapshot of trades already
    in flight, and a position 70% of the way to its target is one where you
    buy what is left of the move while carrying the whole distance to the
    stop. Every row reports how far it has already run, and the result leads
    with the worst of them.
  * **They publish no stop.** OKX shows what a trader holds, never where
    they would get out. Copied positions therefore open without one, which
    `paper_trade_open` already flags per position — and this says it once
    more at the top, because a wallet full of stopless positions is a
    different thing from one trade without a stop.
  * **It previews by default.** Opening several positions at once from one
    sentence is the kind of thing that should be read before it happens, so
    nothing is written until ``confirm=True``.

Nothing here reaches an exchange. The positions are rows in a local SQLite
file priced against live markets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import display_hints
from tradingview_mcp.core.services import paper_pricing as px
from tradingview_mcp.core.services import paper_store as store

_DEFAULT_MARGIN_PCT = 2.0
_MAX_MARGIN_PCT = 25.0
_MAX_COPIES = 10

_STOPLESS_NOTICE = (
    "Copied positions have NO STOP. OKX publishes what a trader holds, never "
    "where they would get out, so there is nothing to copy. On futures the "
    "only automatic exit is liquidation. Set a stop on each one with "
    "paper_trade_modify, or decide deliberately not to."
)


def okx_symbol(inst_id: Optional[str]) -> Optional[str]:
    """``BTC-USDT-SWAP`` -> ``BTCUSDT``. None when it is not a pair we price.

    Only -USDT instruments convert: a coin-margined or option instrument has
    different economics and copying it into a USDT wallet would be inventing
    a trade the trader did not take.
    """
    if not inst_id:
        return None
    parts = str(inst_id).upper().split("-")
    if len(parts) < 2 or parts[1] != "USDT":
        return None
    return f"{parts[0]}USDT"


def _drift_pct(side: str, opened_at: float, now: float) -> float:
    """How far the trade has already run in the trader's favour, as a percent.

    Negative means it has gone against them since they opened — which is the
    case where copying is cheaper than it looks, not more expensive.
    """
    if not opened_at:
        return 0.0
    move = (now - opened_at) if side == px.LONG else (opened_at - now)
    return move / opened_at * 100.0


def copy_trader(unique_code: str, wallet: Optional[str] = None,
                margin_pct: float = _DEFAULT_MARGIN_PCT,
                max_positions: int = 5,
                instrument_type: str = "SWAP",
                confirm: bool = False) -> dict:
    """Open what a lead trader is holding, in a simulated wallet."""
    from tradingview_mcp.core.services.copytrading_service import get_lead_trader
    from tradingview_mcp.core.services.paper_trading_service import (
        _SIMULATION_NOTICE, _use_wallet, _wallet_or_error, open_trade,
    )

    error = _use_wallet(wallet)
    if error:
        return error
    wallet_row, error = _wallet_or_error("copying a trader")
    if error:
        return error

    try:
        margin_pct = float(margin_pct)
        max_positions = int(max_positions)
    except (TypeError, ValueError):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            "margin_pct must be a number and max_positions an integer",
        )
    if not 0 < margin_pct <= _MAX_MARGIN_PCT:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"margin_pct must be 0-{_MAX_MARGIN_PCT}, got {margin_pct}",
        )
    if not 1 <= max_positions <= _MAX_COPIES:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"max_positions must be 1-{_MAX_COPIES}, got {max_positions}",
        )

    profile = get_lead_trader(unique_code, instrument_type)
    if "error" in profile:
        return profile

    # The distinction that exists for exactly this moment: copying a trader
    # whose positions could not be read would open nothing and look like it
    # worked.
    if profile.get("open_positions_state") != "measured":
        return make_error(
            ErrorCode.NO_DATA,
            "This trader's open positions could not be read, so there is "
            "nothing to copy and an empty result would be a lie. "
            + str(profile.get("open_positions_note") or ""),
            unique_code=unique_code,
            open_positions_state=profile.get("open_positions_state"),
            retryable=True,
        )
    positions = profile.get("open_positions") or []
    if not positions:
        return make_error(
            ErrorCode.NO_DATA,
            "OKX answered and this trader is holding nothing right now, so "
            "there is nothing to copy.",
            unique_code=unique_code,
            retryable=True,
        )

    starting = wallet_row["starting_balance"]
    max_lev = wallet_row["max_leverage"]
    plan: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for position in positions[:max_positions]:
        symbol = okx_symbol(position.get("instrument"))
        if not symbol:
            skipped.append({
                "instrument": position.get("instrument"),
                "reason": "not a USDT pair this wallet can price",
            })
            continue
        side = (position.get("side") or "").strip().lower()
        if side not in (px.LONG, px.SHORT):
            skipped.append({"instrument": position.get("instrument"),
                            "reason": f"unreadable side {position.get('side')!r}"})
            continue
        try:
            mark, source = px.mark_price(symbol, store.FUTURES)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"instrument": position.get("instrument"),
                            "reason": f"no price here: {str(exc)[:60]}"})
            continue

        their_lev = position.get("leverage") or 1.0
        lev = min(float(their_lev), float(max_lev))
        margin = starting * margin_pct / 100.0
        notional = margin * lev
        drift = _drift_pct(side, position.get("open_price") or 0.0, mark)

        row = {
            "symbol": symbol,
            "instrument": position.get("instrument"),
            "side": side,
            "their_leverage": their_lev,
            "your_leverage": lev,
            "leverage_capped": lev < float(their_lev),
            "their_open_price": position.get("open_price"),
            "your_entry_price": mark,
            "already_moved_pct": round(drift, 2),
            "your_margin": round(margin, 2),
            "your_notional": round(notional, 2),
            "price_source": source,
        }
        if row["leverage_capped"]:
            row["leverage_note"] = (
                f"they run {their_lev}x and this wallet caps at {max_lev}x, so "
                "this is a different trade from theirs, not a smaller one"
            )
        if drift > 0:
            row["late_note"] = (
                f"already {drift:.1f}% in their favour since they opened — you "
                "would be entering what is left of the move"
            )
        plan.append(row)

    if not plan:
        return make_error(
            ErrorCode.NO_DATA,
            "None of this trader's positions could be copied into this wallet.",
            skipped=skipped or None,
        )

    total_margin = round(sum(r["your_margin"] for r in plan), 2)
    worst = max(plan, key=lambda r: r["already_moved_pct"])
    result: Dict[str, Any] = {
        "trader": profile.get("nickname") or unique_code,
        "unique_code": unique_code,
        "wallet": store.active_wallet(),
        "plan": plan,
        "skipped": skipped or None,
        "total_margin": total_margin,
        "cash_available": round(wallet_row["cash"], 2),
        "enough_cash": total_margin <= wallet_row["cash"],
        "furthest_along": {
            "symbol": worst["symbol"],
            "already_moved_pct": worst["already_moved_pct"],
        },
        "sizing": (
            f"each position uses {margin_pct}% of the wallet's starting "
            f"balance as margin, at the trader's leverage capped to "
            f"{max_lev}x. Their own position sizes are not copied: their "
            "account is not yours."
        ),
        "no_stops": _STOPLESS_NOTICE,
        "not_copy_trading": (
            "You enter at today's price, not theirs. A real copy-trading "
            "platform mirrors you in when they open; a leaderboard shows "
            "trades already running. Read already_moved_pct per row before "
            "deciding."
        ),
        "display_hint": display_hints.COPY_PLAN,
        "simulation": _SIMULATION_NOTICE,
    }

    if not confirm:
        result["opened"] = None
        result["preview"] = (
            "Nothing was opened. This is what would happen; call again with "
            "confirm=true to place these in the wallet."
        )
        return result

    if not result["enough_cash"]:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"This plan needs {total_margin} and the wallet has "
            f"{wallet_row['cash']:.2f} free.",
            required=total_margin, available=round(wallet_row["cash"], 2),
        )

    opened, failed = [], []
    for row in plan:
        out = open_trade(
            row["symbol"], row["side"], row["your_notional"], store.FUTURES,
            row["your_leverage"], None, None,
            f"copied from {result['trader']} ({unique_code}), they opened at "
            f"{row['their_open_price']}",
            wallet,
        )
        if "error" in out:
            failed.append({"symbol": row["symbol"],
                           "reason": out["error"]["message"][:90]})
        else:
            opened.append({
                "symbol": row["symbol"],
                "position_id": out["opened"]["position_id"],
                "entry_price": out["opened"]["entry_price"],
                "notional": out["opened"]["notional"],
                "leverage": out["opened"]["leverage"],
            })
    result["opened"] = opened or None
    result["failed"] = failed or None
    result["preview"] = None
    return result
