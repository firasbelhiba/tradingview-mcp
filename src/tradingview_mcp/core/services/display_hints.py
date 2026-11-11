"""
One place for the "draw this" pointer, so the five panel tools agree.

The server's connection-level PRESENTATION block tells the assistant which
results are worth rendering as a panel. That block named five tool families
and only three service modules actually emitted a ``display_hint``, so most
of the tools the instructions pointed at arrived with no pointer of their own.

The hint is a trigger, not a restatement: the policy travels once, in the
connection instructions, and this rides on every call. Everything after the
prefix names the trap specific to that surface — the field a tidy panel would
leave out, which is the only part the shared prefix cannot say.
"""
from __future__ import annotations

_PREFIX = (
    "If the person is looking at this rather than you reading it to reason "
    "with, draw it as a panel (see this server's PRESENTATION instructions); "
)


def hint(surface_specific: str) -> str:
    """The shared pointer plus what this particular result must not bury."""
    return _PREFIX + surface_specific


BREADTH = hint(
    "show the assessment next to the participation percentages it rests on, "
    "and coverage.note - each percentage has its own denominator."
)

RELATIVE_STRENGTH = hint(
    "show leaders and laggards against the benchmark, and say how many "
    "symbols were measured, not just the ranked ones."
)

SQUEEZE = hint(
    "rank by squeeze_risk and surface unavailable - a symbol whose risk could "
    "not be computed is not a low-risk symbol."
)

FUNDING = hint(
    "show both ends of the carry, and interval_note - funding intervals "
    "differ per symbol, so one annualisation does not fit the board."
)

LEAD_TRADERS = hint(
    "put advertised next to measured, and surface unmeasured - a blank "
    "leverage is missing, not low."
)


# ── The paper wallet, at the moments a person is plainly looking ─────────────
#
# These were missing, and the shape of the gap is instructive: the hint sat on
# the tools that READ state and on none of the tools that CHANGE it. So asking
# "how is my wallet?" drew a panel, while creating the wallet, opening a trade
# or closing one came back as prose — even though someone who just acted is
# the person most certainly looking at the result.

WALLET_CREATED = hint(
    "show the balance with the settings that came with it - fee per side and "
    "max leverage decide what every later number means."
)

TRADE_OPENED = hint(
    "show entry, stop, target and what the stop costs; surface risk_note "
    "when there is no stop, because that position has no defined loss."
)

TRADE_CLOSED = hint(
    "show the realised result next to what fees and funding took out of it - "
    "a strategy that only works before costs should look like one."
)

TRADE_MODIFIED = hint(
    "show the levels that changed against the ones they replaced, not just "
    "the new numbers."
)

ORDER_PLACED = hint(
    "show it resting with the distance left to run, and fill_realism - a "
    "touched level filling is the one place this simulator flatters you."
)

ORDERS_LISTED = hint(
    "show what is resting and the margin it holds - that is cash gone from "
    "the balance and not yet in any position."
)

TRADE_HISTORY = hint(
    "draw the equity curve if it was asked for, and show what fees and "
    "funding consumed, not only the win rate."
)


WALLETS_LISTED = hint(
    "draw one row per wallet with its realised return and what is open; "
    "say that unrealised PnL is not in these figures."
)
