"""
Paper Trading Service — a simulated wallet priced against real markets.

**This is a simulator. It cannot place a real order.** The package holds no
exchange credentials and calls no authenticated endpoint; a position is a row
in a local SQLite file, marked against live prices. Nothing here can move
money, and no amount of prompting makes it able to.

What it is for: testing whether a strategy actually works before risking
anything on it. The other 64 tools say what the market is doing; these let a
decision be recorded, priced, and scored honestly afterwards.

Honest by construction, because a flattering simulator is worse than none:

  * Stops and targets settle against bar highs and lows since the position
    opened, so a stop-out that price later recovered from still counts.
  * When one bar holds both the stop and the target, the stop fills.
  * Taker fees are charged on entry and exit; futures positions pay real
    funding, read from Binance's history.
  * Liquidation is checked on every settle, against Binance's own mark price.
  * Slippage is not modelled, and every result says so.

Spot and futures differ deliberately: spot is long-only, unlevered, and
cash-settled from the wallet; futures allow shorts and leverage, post margin,
pay funding and can be liquidated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import paper_pricing as px
from tradingview_mcp.core.services import paper_store as store

_DEFAULT_BALANCE = 10_000.0
_DEFAULT_FEE_PCT = 0.05        # taker, roughly Binance's rate
_DEFAULT_MAX_LEVERAGE = 20.0
_MIN_BALANCE = 10.0
_MAX_BALANCE = 100_000_000.0

_SIMULATION_NOTICE = (
    "SIMULATED. No real order was placed and no real money moved — this "
    "wallet is a local database priced against live market data."
)
_SLIPPAGE_NOTICE = (
    "Fills land exactly on the level; slippage is not modelled. Real stops "
    "slip, most in the fast moves where they matter, so treat simulated exit "
    "quality as a best case."
)

# A hint to the assistant reading this result, not to the user.
#
# The server cannot draw anything — the ui:// resource it can emit is ignored
# by a local desktop host (measured 2026-11-11). But the assistant usually
# CAN draw, through whatever artifact or inline-visual tool it has, and it
# reads tool results. So the result asks for the panel rather than hoping the
# person thinks to.
#
# Phrased as a conditional preference, not a command: a caller mid-way
# through reasoning about a strategy should not be forced to stop and render,
# and an assistant with no rendering tool should ignore it without ceremony.
# Deliberately terse. The full policy travels once, in the server's
# connection instructions; repeating it in every result cost 890 bytes a call
# and took over half the payload of an empty wallet. This is the trigger, not
# the policy.
_DISPLAY_HINT = (
    "If the person is looking at this rather than you reading it to reason "
    "with, draw it as a panel (see this server's PRESENTATION instructions); "
    "surface unsettled_positions and price_errors, not just the healthy "
    "numbers."
)


class SettleFailed(RuntimeError):
    """A position could not be settled because its bars were unavailable.

    Kept separate from a fill so that `settled` holds one shape only: an
    unsettled position is a gap in knowledge, not an exit, and must never be
    read as "nothing happened".
    """

    def __init__(self, position_id: int, reason: str):
        super().__init__(reason)
        self.position_id = position_id
        self.reason = reason


def _wallet_or_error(action: str):
    wallet = store.get_wallet()
    if not wallet:
        return None, make_error(
            ErrorCode.NO_DATA,
            f"No paper wallet exists yet — create one with paper_wallet_reset "
            f"before {action}.",
        )
    return wallet, None


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_wallet(balance: float = _DEFAULT_BALANCE, currency: str = "USDT",
                 fee_pct: float = _DEFAULT_FEE_PCT,
                 max_leverage: float = _DEFAULT_MAX_LEVERAGE,
                 note: Optional[str] = None) -> dict:
    """Create the wallet, or wipe it and start again.

    Destructive: every position, open and closed, is deleted. That is what a
    practice wallet is for, but the history is not recoverable.
    """
    try:
        balance = float(balance)
        fee_pct = float(fee_pct)
        max_leverage = float(max_leverage)
    except (TypeError, ValueError):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            "balance, fee_pct and max_leverage must be numbers",
        )
    if not _MIN_BALANCE <= balance <= _MAX_BALANCE:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"balance must be {_MIN_BALANCE}-{_MAX_BALANCE}, got {balance}",
        )
    if not 0 <= fee_pct <= 1:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"fee_pct must be 0-1 (percent per side), got {fee_pct}",
        )
    if not 1 <= max_leverage <= 125:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"max_leverage must be 1-125, got {max_leverage}",
        )

    previous = store.get_wallet()
    discarded = len(store.list_positions()) if previous else 0
    wallet = store.reset_wallet(balance, currency.strip().upper(), fee_pct,
                                max_leverage, note)
    return {
        "wallet": {
            "currency": wallet["currency"],
            "starting_balance": wallet["starting_balance"],
            "cash": wallet["cash"],
            "fee_pct_per_side": wallet["fee_pct"],
            "max_leverage": wallet["max_leverage"],
            "created_at": wallet["created_at"],
            "reset_count": wallet["reset_count"],
            "note": wallet["note"],
        },
        "positions_discarded": discarded,
        "simulation": _SIMULATION_NOTICE,
        "database": str(store.db_path()),
    }


# ── Settlement ────────────────────────────────────────────────────────────────

def _settle_one(position: Dict[str, Any], wallet: Dict[str, Any]) -> Optional[dict]:
    """Close the position if a level was reached since it opened."""
    liq = (
        px.liquidation_price(position["side"], position["entry_price"],
                             position["leverage"])
        if position["market"] == store.FUTURES else None
    )
    if position["stop_loss"] is None and position["take_profit"] is None and liq is None:
        return None

    try:
        bars, interval = px.bars_since(
            position["symbol"], position["opened_ms"], position["market"]
        )
    except Exception as exc:  # noqa: BLE001
        # Raised, not returned: a failure has nothing in common with a fill,
        # and returning it into the same list gave `settled` two different
        # shapes — a consumer iterating it hit KeyError on the first failure.
        raise SettleFailed(position["id"], str(exc)[:100]) from exc

    hit = px.scan_bars_for_exit(
        bars, position["side"], position["stop_loss"], position["take_profit"], liq
    )
    if not hit:
        return None

    exit_price = float(hit["price"])
    closed_ms = int(hit["bar"].get("open_ms") or store.now_ms())
    gross = px.position_pnl(position["side"], position["entry_price"],
                            exit_price, position["quantity"])
    exit_fee = abs(exit_price * position["quantity"]) * wallet["fee_pct"] / 100.0
    funding, payments = (0.0, 0)
    if position["market"] == store.FUTURES:
        funding, payments = px.funding_cost(
            position["symbol"], position["side"], position["notional"],
            position["opened_ms"], closed_ms,
        )
    realised = gross - exit_fee - funding
    if hit["reason"] == "liquidated":
        # A liquidation takes the margin; it cannot take more than that here.
        realised = max(realised, -position["margin"])

    store.close_position(
        position["id"], exit_price, realised, hit["reason"], exit_fee,
        funding, closed_ms,
    )
    store.adjust_cash(position["margin"] + realised)
    return {
        "position_id": position["id"],
        "symbol": position["symbol"],
        "side": position["side"],
        "reason": hit["reason"],
        "exit_price": exit_price,
        "gross_pnl": round(gross, 4),
        "exit_fee": round(exit_fee, 4),
        "funding_paid": round(funding, 4),
        "funding_payments": payments,
        "realised_pnl": round(realised, 4),
        "settled_from_bars": interval,
        "ambiguous_bar": hit["ambiguous"],
        "ambiguity_note": (
            "stop and target were both inside one bar; the stop was taken "
            "because the bar does not record which came first"
            if hit["ambiguous"] else None
        ),
    }


def settle(wallet: Optional[Dict[str, Any]] = None) -> Tuple[List[dict], List[dict]]:
    """Run settlement across every open position. Safe to call repeatedly.

    Returns ``(fills, failures)``. They are separate lists because they mean
    opposite things: a fill is a decided outcome, a failure means the position
    could not be checked at all and is still live with its levels unknown.
    """
    wallet = wallet or store.get_wallet()
    if not wallet:
        return [], []
    fills: List[dict] = []
    failures: List[dict] = []
    for position in store.list_positions(store.OPEN):
        try:
            result = _settle_one(position, wallet)
        except SettleFailed as exc:
            failures.append({
                "position_id": exc.position_id,
                "symbol": position["symbol"],
                "unsettled": exc.reason,
                "consequence": (
                    "levels could not be checked, so this position stays open "
                    "and may already have hit its stop or target"
                ),
            })
            continue
        if result:
            fills.append(result)
    return fills, failures


# ── Open ──────────────────────────────────────────────────────────────────────

def open_trade(symbol: str, side: str, notional: float,
               market: str = "futures", leverage: float = 1.0,
               stop_loss: Optional[float] = None,
               take_profit: Optional[float] = None,
               note: Optional[str] = None) -> dict:
    """Open a simulated position at the current market price.

    ``notional`` is position value in the wallet's currency; the margin
    deducted is ``notional / leverage``.
    """
    wallet, error = _wallet_or_error("opening a trade")
    if error:
        return error

    side = (side or "").strip().lower()
    market = (market or "").strip().lower()
    if side not in (px.LONG, px.SHORT):
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"side must be 'long' or 'short', got {side!r}"
        )
    if market not in (store.SPOT, store.FUTURES):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"market must be 'spot' or 'futures', got {market!r}",
        )
    if market == store.SPOT:
        if side == px.SHORT:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                "Spot cannot be shorted — selling an asset the wallet does not "
                "hold is a margin trade. Use market='futures' to go short.",
            )
        if float(leverage) != 1.0:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                "Spot is unlevered; leverage must be 1. Use market='futures' "
                "for leverage.",
            )
    try:
        notional = float(notional)
        leverage = float(leverage)
    except (TypeError, ValueError):
        return make_error(
            ErrorCode.INVALID_PARAMETER, "notional and leverage must be numbers"
        )
    if notional <= 0:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"notional must be positive, got {notional}"
        )
    if leverage < 1 or leverage > wallet["max_leverage"]:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"leverage must be 1-{wallet['max_leverage']} for this wallet",
            wallet_max_leverage=wallet["max_leverage"],
        )

    settled, settle_failures = settle(wallet)
    wallet = store.get_wallet()  # settlement may have returned margin

    try:
        price, source = px.mark_price(symbol, market)
    except px.PriceUnavailable as exc:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, str(exc), symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Price fetch failed: {str(exc)[:100]}",
            retryable=True,
        )

    margin = notional / leverage
    entry_fee = notional * wallet["fee_pct"] / 100.0
    required = margin + entry_fee
    if required > wallet["cash"]:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Not enough cash: this needs {required:.2f} "
            f"({margin:.2f} margin + {entry_fee:.2f} fee) and the wallet has "
            f"{wallet['cash']:.2f} {wallet['currency']}.",
            required=round(required, 2),
            available=round(wallet["cash"], 2),
        )

    quantity = notional / price
    # Validate levels against direction before charging anything: a stop above
    # entry on a long would fill instantly on the next settle, which is a typo
    # rather than a strategy.
    if stop_loss is not None:
        stop_loss = float(stop_loss)
        if side == px.LONG and stop_loss >= price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A long's stop must be below entry: stop {stop_loss} vs price {price}",
            )
        if side == px.SHORT and stop_loss <= price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A short's stop must be above entry: stop {stop_loss} vs price {price}",
            )
    if take_profit is not None:
        take_profit = float(take_profit)
        if side == px.LONG and take_profit <= price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A long's target must be above entry: target {take_profit} vs "
                f"price {price}",
            )
        if side == px.SHORT and take_profit >= price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A short's target must be below entry: target {take_profit} vs "
                f"price {price}",
            )

    opened_ms = store.now_ms()
    from datetime import datetime, timezone
    position = store.add_position(
        symbol=px.normalise_symbol(symbol), market=market, side=side,
        quantity=quantity, entry_price=price, notional=notional,
        leverage=leverage, margin=margin, stop_loss=stop_loss,
        take_profit=take_profit, status=store.OPEN,
        opened_at=datetime.fromtimestamp(opened_ms / 1000.0, tz=timezone.utc)
        .isoformat(timespec="seconds"),
        opened_ms=opened_ms, entry_fee=entry_fee, note=note,
    )
    store.adjust_cash(-required)

    liq = (
        px.liquidation_price(side, price, leverage)
        if market == store.FUTURES else None
    )
    risk = None
    if stop_loss is not None:
        loss_at_stop = abs(px.position_pnl(side, price, stop_loss, quantity))
        risk = {
            "loss_at_stop": round(loss_at_stop + entry_fee, 2),
            "pct_of_wallet": round(
                (loss_at_stop + entry_fee) / wallet["starting_balance"] * 100, 2
            ),
            "distance_pct": round(abs(stop_loss - price) / price * 100, 2),
        }
        if take_profit is not None:
            reward = abs(px.position_pnl(side, price, take_profit, quantity))
            risk["reward_at_target"] = round(reward, 2)
            risk["risk_reward"] = (
                round(reward / loss_at_stop, 2) if loss_at_stop else None
            )

    return {
        "opened": {
            "position_id": position["id"],
            "symbol": position["symbol"],
            "market": market,
            "side": side,
            "entry_price": price,
            "price_source": source,
            "quantity": round(quantity, 10),
            "notional": round(notional, 2),
            "leverage": leverage,
            "margin": round(margin, 2),
            "entry_fee": round(entry_fee, 4),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "liquidation_price": None if liq is None else round(liq, 6),
            "opened_at": position["opened_at"],
            "note": note,
        },
        "risk": risk,
        "risk_note": (
            None if stop_loss is not None else
            "No stop set, so this position has no defined loss and can only be "
            "closed by hand or, on futures, by liquidation."
        ),
        "cash_remaining": round(store.get_wallet()["cash"], 2),
        "settled_before_opening": settled or None,
        "unsettled_positions": settle_failures or None,
        "simulation": _SIMULATION_NOTICE,
        "slippage": _SLIPPAGE_NOTICE,
    }


# ── Close / modify ────────────────────────────────────────────────────────────

def close_trade(position_id: int, note: Optional[str] = None) -> dict:
    """Close an open position at the current market price."""
    wallet, error = _wallet_or_error("closing a trade")
    if error:
        return error

    position = store.get_position(int(position_id))
    if not position:
        return make_error(
            ErrorCode.SYMBOL_NOT_FOUND, f"No position with id {position_id}",
            position_id=position_id,
        )
    if position["status"] != store.OPEN:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Position {position_id} is already closed "
            f"({position['close_reason']}) at {position['exit_price']}.",
            position_id=position_id,
        )

    # Settle first: a level may already have been reached, and closing at the
    # current price would silently overwrite that exit at a better number.
    settled, settle_failures = settle(wallet)
    refreshed = store.get_position(int(position_id))
    if refreshed and refreshed["status"] == store.CLOSED:
        return {
            "closed_by_settlement": next(
                (e for e in settled if e.get("position_id") == refreshed["id"]), None
            ),
            "note": (
                "This position had already reached its stop, target or "
                "liquidation price before the close request; the recorded exit "
                "is that level, not the current price."
            ),
            "simulation": _SIMULATION_NOTICE,
        }

    wallet = store.get_wallet()
    try:
        price, source = px.mark_price(position["symbol"], position["market"])
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Price fetch failed: {str(exc)[:100]}",
            retryable=True,
        )

    closed_ms = store.now_ms()
    gross = px.position_pnl(position["side"], position["entry_price"], price,
                            position["quantity"])
    exit_fee = abs(price * position["quantity"]) * wallet["fee_pct"] / 100.0
    funding, payments = (0.0, 0)
    if position["market"] == store.FUTURES:
        funding, payments = px.funding_cost(
            position["symbol"], position["side"], position["notional"],
            position["opened_ms"], closed_ms,
        )
    realised = gross - exit_fee - funding
    store.close_position(position["id"], price, realised, "manual", exit_fee,
                         funding, closed_ms)
    store.adjust_cash(position["margin"] + realised)

    held_h = (closed_ms - position["opened_ms"]) / 3_600_000.0
    return {
        "closed": {
            "position_id": position["id"],
            "symbol": position["symbol"],
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": price,
            "price_source": source,
            "held_hours": round(held_h, 2),
            "gross_pnl": round(gross, 4),
            "entry_fee": round(position["entry_fee"], 4),
            "exit_fee": round(exit_fee, 4),
            "funding_paid": round(funding, 4),
            "funding_payments": payments,
            "realised_pnl": round(realised, 4),
            "return_on_margin_pct": (
                round(realised / position["margin"] * 100, 2)
                if position["margin"] else None
            ),
            "reason": "manual",
            "note": note,
        },
        "cash_after": round(store.get_wallet()["cash"], 2),
        "also_settled": [e for e in settled if e.get("position_id") != position["id"]]
        or None,
        "unsettled_positions": settle_failures or None,
        "simulation": _SIMULATION_NOTICE,
    }


def modify_trade(position_id: int, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None) -> dict:
    """Set or move the stop and target on an open position."""
    wallet, error = _wallet_or_error("modifying a trade")
    if error:
        return error
    if stop_loss is None and take_profit is None:
        return make_error(
            ErrorCode.INVALID_PARAMETER, "Pass stop_loss, take_profit, or both"
        )
    position = store.get_position(int(position_id))
    if not position:
        return make_error(
            ErrorCode.SYMBOL_NOT_FOUND, f"No position with id {position_id}"
        )
    if position["status"] != store.OPEN:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"Position {position_id} is closed"
        )

    try:
        price, _ = px.mark_price(position["symbol"], position["market"])
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Price fetch failed: {str(exc)[:100]}",
            retryable=True,
        )

    side = position["side"]
    updates: Dict[str, Any] = {}
    if stop_loss is not None:
        stop_loss = float(stop_loss)
        wrong = (side == px.LONG and stop_loss >= price) or (
            side == px.SHORT and stop_loss <= price
        )
        if wrong:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A {side}'s stop must be "
                f"{'below' if side == px.LONG else 'above'} the current price "
                f"({price}); {stop_loss} would fill immediately.",
            )
        updates["stop_loss"] = stop_loss
    if take_profit is not None:
        take_profit = float(take_profit)
        wrong = (side == px.LONG and take_profit <= price) or (
            side == px.SHORT and take_profit >= price
        )
        if wrong:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A {side}'s target must be "
                f"{'above' if side == px.LONG else 'below'} the current price "
                f"({price}); {take_profit} would fill immediately.",
            )
        updates["take_profit"] = take_profit

    store.update_position(int(position_id), **updates)
    updated = store.get_position(int(position_id))
    return {
        "modified": {
            "position_id": updated["id"],
            "symbol": updated["symbol"],
            "side": updated["side"],
            "current_price": price,
            "stop_loss": updated["stop_loss"],
            "take_profit": updated["take_profit"],
        },
        "simulation": _SIMULATION_NOTICE,
    }


# ── Status and history ────────────────────────────────────────────────────────

def wallet_status(include_closed: int = 0) -> dict:
    """The wallet: cash, open positions marked to live prices, and equity."""
    wallet, error = _wallet_or_error("checking the wallet")
    if error:
        return error

    settled, settle_failures = settle(wallet)
    wallet = store.get_wallet()

    open_rows: List[dict] = []
    unrealised_total = 0.0
    price_errors: List[dict] = []
    for position in store.list_positions(store.OPEN):
        try:
            price, source = px.mark_price(position["symbol"], position["market"])
        except Exception as exc:  # noqa: BLE001
            price_errors.append({
                "position_id": position["id"], "symbol": position["symbol"],
                "error": str(exc)[:80],
            })
            open_rows.append({
                "position_id": position["id"], "symbol": position["symbol"],
                "side": position["side"], "entry_price": position["entry_price"],
                "current_price": None, "unrealised_pnl": None,
                "note": "price unavailable, so this position is not marked",
            })
            continue
        unrealised = px.position_pnl(position["side"], position["entry_price"],
                                     price, position["quantity"])
        unrealised_total += unrealised
        liq = (
            px.liquidation_price(position["side"], position["entry_price"],
                                 position["leverage"])
            if position["market"] == store.FUTURES else None
        )
        open_rows.append({
            "position_id": position["id"],
            "symbol": position["symbol"],
            "market": position["market"],
            "side": position["side"],
            "entry_price": position["entry_price"],
            "current_price": price,
            "price_source": source,
            "quantity": round(position["quantity"], 10),
            "notional": round(position["notional"], 2),
            "leverage": position["leverage"],
            "margin": round(position["margin"], 2),
            "unrealised_pnl": round(unrealised, 4),
            "unrealised_pct_of_margin": (
                round(unrealised / position["margin"] * 100, 2)
                if position["margin"] else None
            ),
            "stop_loss": position["stop_loss"],
            "take_profit": position["take_profit"],
            "liquidation_price": None if liq is None else round(liq, 6),
            "opened_at": position["opened_at"],
            "note": position["note"],
        })

    closed = store.list_positions(store.CLOSED)
    realised_total = sum(p["realised_pnl"] or 0.0 for p in closed)
    margin_posted = sum(p["margin"] for p in store.list_positions(store.OPEN))
    equity = wallet["cash"] + margin_posted + unrealised_total

    result: Dict[str, Any] = {
        "wallet": {
            "currency": wallet["currency"],
            "starting_balance": round(wallet["starting_balance"], 2),
            "cash": round(wallet["cash"], 2),
            "margin_posted": round(margin_posted, 2),
            "unrealised_pnl": round(unrealised_total, 4),
            "equity": round(equity, 2),
            "total_return_pct": round(
                (equity - wallet["starting_balance"]) / wallet["starting_balance"]
                * 100, 2
            ),
            "realised_pnl": round(realised_total, 4),
            "fee_pct_per_side": wallet["fee_pct"],
            "max_leverage": wallet["max_leverage"],
            "created_at": wallet["created_at"],
            "reset_count": wallet["reset_count"],
        },
        "open_positions": open_rows,
        "open_count": len(open_rows),
        "closed_count": len(closed),
        "settled_this_check": settled or None,
        "unsettled_positions": settle_failures or None,
        "equity_note": "equity = cash + margin posted + unrealised pnl",
        "display_hint": _DISPLAY_HINT,
        "simulation": _SIMULATION_NOTICE,
        "slippage": _SLIPPAGE_NOTICE,
    }
    if price_errors:
        result["price_errors"] = price_errors
    if include_closed:
        result["recent_closed"] = _closed_rows(
            store.list_positions(store.CLOSED, limit=int(include_closed))
        )
    return result


def _closed_rows(rows: List[Dict[str, Any]]) -> List[dict]:
    out = []
    for p in rows:
        held = (
            None if not p.get("closed_ms")
            else round((p["closed_ms"] - p["opened_ms"]) / 3_600_000.0, 2)
        )
        out.append({
            "position_id": p["id"],
            "symbol": p["symbol"],
            "market": p["market"],
            "side": p["side"],
            "leverage": p["leverage"],
            "entry_price": p["entry_price"],
            "exit_price": p["exit_price"],
            "reason": p["close_reason"],
            "held_hours": held,
            "entry_fee": round(p["entry_fee"], 4),
            "exit_fee": round(p["exit_fee"], 4),
            "funding_paid": round(p["funding_paid"], 4),
            "realised_pnl": round(p["realised_pnl"] or 0.0, 4),
            "return_on_margin_pct": (
                round((p["realised_pnl"] or 0.0) / p["margin"] * 100, 2)
                if p["margin"] else None
            ),
            "closed_at": p["closed_at"],
            "note": p["note"],
        })
    return out


def trade_history(limit: int = 50) -> dict:
    """Closed positions and the performance they add up to."""
    wallet, error = _wallet_or_error("reading history")
    if error:
        return error
    if not 1 <= int(limit) <= 500:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"limit must be 1-500, got {limit}"
        )

    _, settle_failures = settle(wallet)
    closed = store.list_positions(store.CLOSED)
    if not closed:
        return {
            "trades": [],
            "statistics": None,
            "note": "No closed positions yet — open one with paper_trade_open.",
            "simulation": _SIMULATION_NOTICE,
        }

    pnls = [p["realised_pnl"] or 0.0 for p in closed]
    wins = [v for v in pnls if v > 0]
    losses = [v for v in pnls if v < 0]
    fees = sum((p["entry_fee"] or 0.0) + (p["exit_fee"] or 0.0) for p in closed)
    funding = sum(p["funding_paid"] or 0.0 for p in closed)
    gross_profit, gross_loss = sum(wins), abs(sum(losses))

    # Equity curve in close order, for a drawdown that sees the sequence.
    ordered = sorted(
        (p for p in closed if p.get("closed_ms")), key=lambda p: p["closed_ms"]
    )
    cumulative, peak, worst = 0.0, 0.0, 0.0
    for p in ordered:
        cumulative += p["realised_pnl"] or 0.0
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)

    reasons: Dict[str, int] = {}
    for p in closed:
        reasons[p["close_reason"] or "unknown"] = (
            reasons.get(p["close_reason"] or "unknown", 0) + 1
        )

    return {
        "trades": _closed_rows(store.list_positions(store.CLOSED, limit=int(limit))),
        "statistics": {
            "trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
            "net_pnl": round(sum(pnls), 4),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "profit_factor": (
                None if not wins or not losses
                else round(gross_profit / gross_loss, 2)
            ),
            "avg_win": round(gross_profit / len(wins), 4) if wins else None,
            "avg_loss": round(gross_loss / len(losses), 4) if losses else None,
            "expectancy_per_trade": round(sum(pnls) / len(pnls), 4) if pnls else None,
            "best_trade": round(max(pnls), 4) if pnls else None,
            "worst_trade": round(min(pnls), 4) if pnls else None,
            "realised_drawdown": round(worst, 4),
            "total_fees_paid": round(fees, 4),
            "total_funding_paid": round(funding, 4),
            "costs_vs_gross": (
                None if not pnls else
                round((fees + funding) / max(abs(sum(pnls)) + fees + funding, 1e-9)
                      * 100, 1)
            ),
            "closed_by": reasons,
        },
        "unsettled_positions": settle_failures or None,
        "display_hint": _DISPLAY_HINT,
        "cost_note": (
            "Fees and funding are already inside realised_pnl; "
            "total_fees_paid and total_funding_paid show how much of the "
            "result they consumed."
        ),
        "simulation": _SIMULATION_NOTICE,
        "slippage": _SLIPPAGE_NOTICE,
    }
