"""
Paper Orders — resting limit orders for the simulated wallet.

The wallet could only ever fill at the current market price, which is the
one thing a swing trader does not do. "Sell at resistance, buy at support"
means leaving an order at a level and walking away; without that, the only
way to enter at 72,461 was to sit and watch for it.

Nothing here watches anything. The wallet already settles retroactively —
every check pulls the bars since you last looked and asks whether a stop or
target was reached — and a resting order is the same question asked about an
entry price. So an order placed on Monday and checked on Friday fills at the
bar that reached it, with its stop and target settling from that bar onward.

Three rules shape it, and each one is a way this could have taught something
false:

  * **Margin is held from the moment the order is placed**, the way an
    exchange holds it. Without that, two orders resting on the same cash
    would both fill, and a wallet could carry more risk than it has money.
  * **A fill at a touched level is a BEST CASE.** The stop logic treats a
    touch as a fill, which is pessimistic and right. The same rule on an
    entry is optimistic: in a real book you sit behind everyone already
    resting at that price, and a level merely tagged often fills nobody.
  * **An order that would fill immediately is refused.** A buy resting above
    the market is a market order with extra steps, and letting it through
    would quietly turn "wait for support" into "buy now".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import display_hints
from tradingview_mcp.core.services import paper_pricing as px
from tradingview_mcp.core.services import paper_store as store

_FILL_NOTICE = (
    "A resting order is filled when a bar's range reaches the level, at the "
    "level exactly. That is a BEST CASE: in a real book you queue behind the "
    "orders already there, and a price merely tagged often fills nobody. "
    "Treat simulated entry quality the way you treat simulated exits."
)

_MAX_GOOD_FOR_HOURS = 24 * 90


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _order_row(order: Dict[str, Any], mark: Optional[float] = None) -> dict:
    """One order as the caller sees it, with the distance left to run."""
    row = {
        "order_id": order["id"],
        "symbol": order["symbol"],
        "market": order["market"],
        "side": order["side"],
        "limit_price": order["limit_price"],
        "notional": round(order["notional"], 2),
        "leverage": order["leverage"],
        "margin_held": round(order["margin_held"], 2),
        "stop_loss": order["stop_loss"],
        "take_profit": order["take_profit"],
        "status": order["status"],
        "placed_at": order["placed_at"],
        "expires_at": None if order["expires_ms"] is None
        else _iso(order["expires_ms"]),
        "note": order["note"],
    }
    if mark is not None:
        row["current_price"] = mark
        away = (mark - order["limit_price"]) / mark * 100.0
        row["distance_pct"] = round(abs(away), 2)
    if order["status"] != store.PENDING:
        row["resolution"] = order["resolution"]
        row["resolved_at"] = order["resolved_at"]
        row["position_id"] = order["position_id"]
    return row


# ── Placing ───────────────────────────────────────────────────────────────────

def place_order(symbol: str, side: str, notional: float, limit_price: float,
                market: str = "futures", leverage: float = 1.0,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                good_for_hours: Optional[float] = None,
                note: Optional[str] = None) -> dict:
    """Leave an order resting at a level until a later check reaches it."""
    from tradingview_mcp.core.services.paper_trading_service import (
        _SIMULATION_NOTICE, _wallet_or_error, settle,
    )

    wallet, error = _wallet_or_error("placing an order")
    if error:
        return error

    side = (side or "").strip().lower()
    market = (market or "").strip().lower()
    if side not in (px.LONG, px.SHORT):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"side must be 'long' or 'short', got {side!r}",
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
                "Spot cannot be shorted. Use market='futures' to go short.",
            )
        if float(leverage) != 1.0:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                "Spot is unlevered; leverage must be 1.",
            )
    try:
        notional = float(notional)
        leverage = float(leverage)
        limit_price = float(limit_price)
    except (TypeError, ValueError):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            "notional, leverage and limit_price must be numbers",
        )
    if notional <= 0:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"notional must be positive, got {notional}"
        )
    if limit_price <= 0:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"limit_price must be positive, got {limit_price}",
        )
    if leverage < 1 or leverage > wallet["max_leverage"]:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"leverage must be 1-{wallet['max_leverage']} for this wallet",
            wallet_max_leverage=wallet["max_leverage"],
        )
    if good_for_hours is not None:
        try:
            good_for_hours = float(good_for_hours)
        except (TypeError, ValueError):
            return make_error(
                ErrorCode.INVALID_PARAMETER, "good_for_hours must be a number"
            )
        if not 0 < good_for_hours <= _MAX_GOOD_FOR_HOURS:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"good_for_hours must be 0-{_MAX_GOOD_FOR_HOURS}, "
                f"got {good_for_hours}",
            )

    settle(wallet)
    wallet = store.get_wallet()

    try:
        mark, source = px.mark_price(symbol, market)
    except px.PriceUnavailable as exc:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, str(exc), symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Price fetch failed: {str(exc)[:100]}",
            retryable=True,
        )

    # An order that is already through the market is not a resting order.
    # Letting it stand would turn "wait for support" into "buy now" on the
    # very next check, which is the opposite of what it was placed for.
    if side == px.LONG and limit_price >= mark:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"A resting buy has to sit below the market: limit {limit_price} "
            f"is at or above {mark}. Use paper_trade_open to buy now.",
            limit_price=limit_price, current_price=mark,
        )
    if side == px.SHORT and limit_price <= mark:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"A resting sell has to sit above the market: limit {limit_price} "
            f"is at or below {mark}. Use paper_trade_open to sell now.",
            limit_price=limit_price, current_price=mark,
        )

    # Levels are checked against the LIMIT, not the market: the limit is where
    # this trade would start, so that is what a stop has to be the wrong side
    # of to be a stop at all.
    if stop_loss is not None:
        stop_loss = float(stop_loss)
        if side == px.LONG and stop_loss >= limit_price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A long's stop must be below its entry: stop {stop_loss} vs "
                f"limit {limit_price}",
            )
        if side == px.SHORT and stop_loss <= limit_price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A short's stop must be above its entry: stop {stop_loss} vs "
                f"limit {limit_price}",
            )
    if take_profit is not None:
        take_profit = float(take_profit)
        if side == px.LONG and take_profit <= limit_price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A long's target must be above its entry: target "
                f"{take_profit} vs limit {limit_price}",
            )
        if side == px.SHORT and take_profit >= limit_price:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"A short's target must be below its entry: target "
                f"{take_profit} vs limit {limit_price}",
            )

    margin = notional / leverage
    entry_fee = notional * wallet["fee_pct"] / 100.0
    if margin + entry_fee > wallet["cash"]:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Not enough cash: this holds {margin:.2f} margin and the fee at "
            f"fill is about {entry_fee:.2f}, against {wallet['cash']:.2f} "
            f"{wallet['currency']} free. Cash resting on other orders is "
            f"already excluded.",
            required=round(margin + entry_fee, 2),
            available=round(wallet["cash"], 2),
        )

    placed_ms = store.now_ms()
    expires_ms = (
        None if good_for_hours is None
        else placed_ms + int(good_for_hours * 3600 * 1000)
    )
    order = store.add_order(
        symbol=px.normalise_symbol(symbol), market=market, side=side,
        notional=notional, leverage=leverage, margin_held=margin,
        limit_price=limit_price, stop_loss=stop_loss, take_profit=take_profit,
        status=store.PENDING, placed_at=_iso(placed_ms), placed_ms=placed_ms,
        expires_ms=expires_ms, note=note,
    )
    store.adjust_cash(-margin)

    away_pct = abs((mark - limit_price) / mark * 100.0)
    risk = None
    if stop_loss is not None:
        quantity = notional / limit_price
        loss = abs(px.position_pnl(side, limit_price, stop_loss, quantity))
        risk = {
            "loss_at_stop": round(loss + entry_fee, 2),
            "pct_of_wallet": round(
                (loss + entry_fee) / wallet["starting_balance"] * 100, 2
            ),
        }
        if take_profit is not None:
            reward = abs(px.position_pnl(side, limit_price, take_profit, quantity))
            risk["reward_at_target"] = round(reward, 2)
            risk["risk_reward"] = round(reward / loss, 2) if loss else None

    return {
        "placed": _order_row(order, mark),
        "distance_to_fill_pct": round(away_pct, 2),
        "margin_held_now": round(margin, 2),
        "cash_free": round(store.get_wallet()["cash"], 2),
        "risk_if_filled": risk,
        "price_source": source,
        "how_it_fills": (
            "Nothing watches the market. The next time the wallet is checked, "
            "the bars since now are scanned; if one reached the limit the "
            "order fills at that bar's time, and the stop and target are then "
            "settled from that bar forward."
        ),
        "fill_realism": _FILL_NOTICE,
        "display_hint": display_hints.ORDER_PLACED,
        "simulation": _SIMULATION_NOTICE,
    }


# ── Cancelling ────────────────────────────────────────────────────────────────

def cancel_order(order_id: int, note: Optional[str] = None) -> dict:
    """Take a resting order off the book and give its margin back."""
    from tradingview_mcp.core.services.paper_trading_service import (
        _SIMULATION_NOTICE, _wallet_or_error,
    )

    _, error = _wallet_or_error("cancelling an order")
    if error:
        return error

    order = store.get_order(int(order_id))
    if not order:
        return make_error(
            ErrorCode.RECORD_NOT_FOUND, f"No order with id {order_id}",
            order_id=order_id, retryable=False,
        )
    if order["status"] != store.PENDING:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Order {order_id} is already {order['status']} "
            f"({order['resolution']})",
            order_id=order_id, status=order["status"],
        )

    store.resolve_order(order["id"], store.CANCELLED,
                        note or "cancelled before it filled")
    store.adjust_cash(order["margin_held"])
    return {
        "cancelled": _order_row(store.get_order(order["id"])),
        "margin_released": round(order["margin_held"], 2),
        "cash_free": round(store.get_wallet()["cash"], 2),
        "simulation": _SIMULATION_NOTICE,
    }


# ── Settlement ────────────────────────────────────────────────────────────────

def settle_orders(wallet: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
    """Fill or expire whatever the bars since placement decided.

    Runs before positions settle, so an order that filled and then hit its
    stop inside the same window resolves both in one check: the fill makes a
    position stamped with the fill bar's own time, and the position scan that
    follows starts from that bar.

    Returns ``(resolved, unchecked)``. They mean opposite things: resolved is
    a decided outcome, unchecked means the bars never came back and the order
    is still resting with its fate unknown.
    """
    resolved: List[dict] = []
    unchecked: List[dict] = []

    for order in store.list_orders(store.PENDING):
        try:
            bars, _interval = px.bars_since(
                order["symbol"], order["placed_ms"], order["market"]
            )
        except Exception as exc:  # noqa: BLE001
            unchecked.append({
                "order_id": order["id"],
                "symbol": order["symbol"],
                "unchecked": f"{type(exc).__name__}: {str(exc)[:80]}",
                "consequence": (
                    "the bars could not be read, so this order is still "
                    "resting and may already have been reached"
                ),
            })
            continue

        hit = px.scan_bars_for_fill(bars, order["side"], order["limit_price"])
        if hit:
            filled = _fill(order, hit, wallet)
            if filled:
                resolved.append(filled)
            continue

        # Expiry is checked only after the fill scan: an order that was
        # reached on Tuesday and expired on Wednesday was filled on Tuesday.
        if order["expires_ms"] is not None and store.now_ms() > order["expires_ms"]:
            store.resolve_order(
                order["id"], store.EXPIRED,
                "the time limit passed without the price reaching it",
                resolved_ms=order["expires_ms"],
            )
            store.adjust_cash(order["margin_held"])
            resolved.append({
                "order_id": order["id"],
                "symbol": order["symbol"],
                "outcome": "expired",
                "limit_price": order["limit_price"],
                "margin_released": round(order["margin_held"], 2),
            })

    return resolved, unchecked


def _fill(order: Dict[str, Any], hit: Dict[str, Any],
          wallet: Dict[str, Any]) -> Optional[dict]:
    """Turn a reached order into a position stamped with the bar's own time."""
    price = float(hit["price"])
    filled_ms = int(hit["bar"].get("open_ms") or store.now_ms())
    entry_fee = order["notional"] * wallet["fee_pct"] / 100.0

    # The margin was taken when the order was placed; only the fee is new. If
    # the wallet cannot pay it the order dies rather than opening a position
    # the wallet cannot afford.
    if entry_fee > store.get_wallet()["cash"]:
        store.resolve_order(
            order["id"], store.CANCELLED,
            f"reached {price} but the wallet could not pay the "
            f"{entry_fee:.2f} entry fee",
            resolved_ms=filled_ms,
        )
        store.adjust_cash(order["margin_held"])
        return {
            "order_id": order["id"],
            "symbol": order["symbol"],
            "outcome": "cancelled_at_fill",
            "reason": "not enough cash for the entry fee when it was reached",
            "margin_released": round(order["margin_held"], 2),
        }

    quantity = order["notional"] / price
    position = store.add_position(
        symbol=order["symbol"], market=order["market"], side=order["side"],
        quantity=quantity, entry_price=price, notional=order["notional"],
        leverage=order["leverage"], margin=order["margin_held"],
        stop_loss=order["stop_loss"], take_profit=order["take_profit"],
        status=store.OPEN, opened_at=_iso(filled_ms), opened_ms=filled_ms,
        entry_fee=entry_fee,
        note=order["note"] or f"filled from resting order {order['id']}",
    )
    store.adjust_cash(-entry_fee)
    store.resolve_order(order["id"], store.FILLED,
                        f"reached {price} and opened position {position['id']}",
                        position_id=position["id"], resolved_ms=filled_ms)
    return {
        "order_id": order["id"],
        "symbol": order["symbol"],
        "outcome": "filled",
        "side": order["side"],
        "fill_price": price,
        "filled_at": _iso(filled_ms),
        "position_id": position["id"],
        "entry_fee": round(entry_fee, 4),
        "note": (
            "filled at the level exactly; a real book may not have filled "
            "you there"
        ),
    }


# ── Listing ───────────────────────────────────────────────────────────────────

def list_orders(status: Optional[str] = None, limit: int = 50) -> dict:
    """Resting orders, or the history of what became of them."""
    from tradingview_mcp.core.services.paper_trading_service import (
        _SIMULATION_NOTICE, _wallet_or_error, settle,
    )

    wallet, error = _wallet_or_error("listing orders")
    if error:
        return error
    if status is not None:
        status = str(status).strip().lower()
        valid = (store.PENDING, store.FILLED, store.CANCELLED, store.EXPIRED)
        if status not in valid:
            return make_error(
                ErrorCode.INVALID_PARAMETER,
                f"status must be one of {', '.join(valid)}, got {status!r}",
            )
    if not 1 <= int(limit) <= 500:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"limit must be 1-500, got {limit}"
        )

    settled = settle(wallet)
    rows: List[dict] = []
    marks: Dict[str, Optional[float]] = {}
    for order in store.list_orders(status, int(limit)):
        mark = None
        if order["status"] == store.PENDING:
            key = (order["symbol"], order["market"])
            if key not in marks:
                try:
                    marks[key] = px.mark_price(order["symbol"], order["market"])[0]
                except Exception:  # noqa: BLE001
                    marks[key] = None
            mark = marks[key]
        rows.append(_order_row(order, mark))

    pending = [r for r in rows if r["status"] == store.PENDING]
    return {
        "orders": rows,
        "counts": {
            "pending": len(store.list_orders(store.PENDING)),
            "returned": len(rows),
        },
        "margin_held_by_pending": round(
            sum(o["margin_held"] for o in store.list_orders(store.PENDING)), 2
        ),
        "nearest_fill": min(
            (r for r in pending if r.get("distance_pct") is not None),
            key=lambda r: r["distance_pct"], default=None,
        ),
        "settled_this_check": settled[0] if isinstance(settled, tuple) else None,
        "fill_realism": _FILL_NOTICE if pending else None,
        "display_hint": (
            display_hints.ORDERS_LISTED if rows else None
        ),
        "simulation": _SIMULATION_NOTICE,
    }
