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
