"""
Paper Pricing — real prices and honest fills for the simulated wallet.

The point of a practice wallet is to learn something transferable, so the
simulation has to be hard where the market is hard. Three decisions carry
most of that weight:

1. **Stops and targets settle against bar highs and lows, not against the
   price at the moment you happen to look.** A naive simulator only triggers
   a stop if the viewer checks while price is through it, which quietly
   deletes every stop-out that price recovered from — the most expensive
   lesson a beginner needs. Here, settling a position fetches the bars since
   it opened and asks whether the low (long) or high (short) ever reached the
   level.

2. **When one bar contains both the stop and the target, the stop wins.**
   The bar does not record which came first, and assuming the favourable one
   would inflate every result. Ambiguous fills are counted and reported.

3. **Costs are charged.** Taker fees on entry and exit, and for futures the
   real funding payments over the holding period, read from Binance's funding
   history rather than guessed. A levered position held for days costs money
   to hold, and a simulator that ignores that teaches a strategy that does
   not exist.

Slippage is NOT modelled: fills land exactly on the level. Real stops slip,
especially in the cascades these tools are good at spotting, so a simulated
result is a best case on exit quality. Every result says so.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.services import binance_service as bn

LONG = "long"
SHORT = "short"

# Finest kline interval whose 1500-bar ceiling still spans the holding period.
# A coarser bar hides the order of events inside it, so the settler reports
# which resolution it used.
_INTERVAL_SPAN_MS = (
    ("1m", 60_000),
    ("5m", 300_000),
    ("15m", 900_000),
    ("1h", 3_600_000),
    ("4h", 14_400_000),
    ("1d", 86_400_000),
)
_MAX_BARS = 1500


class PriceUnavailable(RuntimeError):
    """No live price could be read for this instrument."""


def _f(value: Any) -> Optional[float]:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def normalise_symbol(symbol: str) -> str:
    """The Binance pair a wallet symbol refers to. BTC means BTCUSDT."""
    from tradingview_mcp.core.utils.validators import crypto_pair

    return crypto_pair(symbol, "BINANCE")


def mark_price(symbol: str, market: str) -> Tuple[float, str]:
    """Live price for ``symbol``, plus the source that supplied it.

    Futures use Binance's own mark price — the number the exchange liquidates
    against — rather than the last trade, so a simulated liquidation triggers
    on the same input a real one would.
    """
    sym = normalise_symbol(symbol)
    if market == "futures":
        try:
            price = _f((bn.get_premium_index(sym) or {}).get("markPrice"))
            if price:
                return price, "Binance futures mark price"
        except Exception:  # noqa: BLE001
            pass
    for mkt, label in (("futures", "Binance futures last close"),
                       ("spot", "Binance spot last close")):
        if market == "spot" and mkt == "futures":
            continue  # a spot order should not be priced off the perp
        try:
            candles = bn.get_klines(sym, "1m", 2, mkt)
            if candles:
                price = candles[-1].get("close")
                if price:
                    return float(price), label
        except Exception:  # noqa: BLE001
            continue
    raise PriceUnavailable(
        f"No live price for {sym} on {market}. Check the symbol lists on "
        "Binance (spot or perpetual)."
    )


def pick_interval(span_ms: int) -> Tuple[str, int]:
    """The finest interval that covers ``span_ms`` within the bar ceiling."""
    for name, step in _INTERVAL_SPAN_MS:
        if span_ms <= step * _MAX_BARS:
            return name, step
    return _INTERVAL_SPAN_MS[-1]


def bars_since(symbol: str, since_ms: int, market: str,
               now_ms: Optional[int] = None) -> Tuple[List[dict], str]:
    """Bars covering ``since_ms`` to now, with the interval used.

    The bar that was forming when the position opened is included: a stop can
    be hit within the same bar as entry, and dropping it would hide that.
    """
    import time as _time

    sym = normalise_symbol(symbol)
    end = now_ms if now_ms is not None else int(_time.time() * 1000)
    span = max(0, end - since_ms)
    interval, step = pick_interval(span)
    needed = min(_MAX_BARS, max(2, int(span // step) + 2))
    bn_market = "spot" if market == "spot" else "futures"
    candles = bn.get_klines(sym, interval, needed, bn_market)
    kept = [
        c for c in candles
        if c.get("open_ms") is not None and c["open_ms"] + step > since_ms
    ]
    return kept, interval


def funding_cost(symbol: str, side: str, notional: float,
                 opened_ms: int, closed_ms: int) -> Tuple[float, int]:
    """Funding paid (positive) or received (negative) over the holding period.

    Read from the real funding history rather than extrapolated from the
    current rate: funding moves, and a position held through a flip would
    otherwise be charged a cost it never paid. A long pays when the rate is
    positive; a short receives it.

    Returns ``(cost, payments_counted)``.
    """
    try:
        history = bn.get_funding_history(normalise_symbol(symbol), 100) or []
    except Exception:  # noqa: BLE001
        return 0.0, 0
    direction = 1.0 if side == LONG else -1.0
    total, counted = 0.0, 0
    for row in history:
        stamp = _f(row.get("fundingTime"))
        rate = _f(row.get("fundingRate"))
        if stamp is None or rate is None:
            continue
        if opened_ms <= stamp <= closed_ms:
            total += rate * notional * direction
            counted += 1
    return total, counted


def liquidation_price(side: str, entry: float, leverage: float) -> Optional[float]:
    """Price at which the position's loss equals its margin.

    A simplification, and a stated one: real exchanges liquidate earlier,
    at a maintenance margin that varies with position size and tier. This
    returns the zero-margin point, so a simulated position survives slightly
    longer than a real one would.
    """
    if not entry or not leverage or leverage <= 0:
        return None
    move = entry / leverage
    return entry - move if side == LONG else entry + move


def scan_bars_for_exit(
    bars: List[dict],
    side: str,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    liquidation: Optional[float],
) -> Optional[dict]:
    """First bar in which an exit level was reached, and which level it was.

    Within a single bar the order of events is unknown, so the unfavourable
    level is taken first: liquidation, then stop, then target. That biases
    every ambiguous case against the trade, which is the right direction for
    a tool meant to teach.
    """
    for bar in bars:
        high, low = bar.get("high"), bar.get("low")
        if high is None or low is None:
            continue
        hit_liq = (
            liquidation is not None
            and ((side == LONG and low <= liquidation)
                 or (side == SHORT and high >= liquidation))
        )
        hit_stop = (
            stop_loss is not None
            and ((side == LONG and low <= stop_loss)
                 or (side == SHORT and high >= stop_loss))
        )
        hit_tp = (
            take_profit is not None
            and ((side == LONG and high >= take_profit)
                 or (side == SHORT and low <= take_profit))
        )
        if hit_liq:
            return {"reason": "liquidated", "price": liquidation, "bar": bar,
                    "ambiguous": bool(hit_tp)}
        if hit_stop:
            return {"reason": "stop_loss", "price": stop_loss, "bar": bar,
                    "ambiguous": bool(hit_tp)}
        if hit_tp:
            return {"reason": "take_profit", "price": take_profit, "bar": bar,
                    "ambiguous": False}
    return None


def scan_bars_for_fill(bars: List[dict], side: str,
                       limit_price: float) -> Optional[dict]:
    """First bar in which a resting limit order would have been reached.

    A buy rests below the market and is reached when a bar's low trades down
    to it; a sell rests above and is reached when a bar's high trades up.

    Note the asymmetry with ``scan_bars_for_exit``. Treating a touch as a
    fill makes a STOP pessimistic — it counts you out of trades that
    recovered. The same rule makes a LIMIT optimistic: in a real book you sit
    behind the orders already resting at that price, and a level that is
    merely tagged often fills nobody. Every result that uses this says so.
    """
    for bar in bars:
        high, low = bar.get("high"), bar.get("low")
        if high is None or low is None:
            continue
        reached = (low <= limit_price) if side == LONG else (high >= limit_price)
        if reached:
            return {"price": limit_price, "bar": bar}
    return None


def position_pnl(side: str, entry: float, exit_price: float,
                 quantity: float) -> float:
    """PnL in quote currency. Quantity is in base units, so leverage is
    already expressed in the size rather than applied again here."""
    move = (exit_price - entry) if side == LONG else (entry - exit_price)
    return move * quantity
