"""
Limits — how much one deployment lets its callers create, and how fast.

The server was built to run over stdio for one person. Hosted behind a URL,
every caller shares one process and one disk, and until now nothing stopped
a single caller from creating wallets, alerts, orders and positions without
end, or from calling tools as fast as the network allowed. The upstreams
throttle by the SERVER's address, so one loop degrades it for everybody.

Counts are capped always, because no honest use comes near them. The call
budget is off unless ``TRADINGVIEW_MCP_RATE_LIMIT_PER_MIN`` is set: a local
stdio install has one user and nothing to protect them from.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from threading import Lock
from typing import Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Each wallet is a file on disk, so this one is also a disk limit.
MAX_WALLETS = _env_int("TRADINGVIEW_MCP_MAX_WALLETS", 25)
MAX_OPEN_POSITIONS = _env_int("TRADINGVIEW_MCP_MAX_OPEN_POSITIONS", 50)
MAX_PENDING_ORDERS = _env_int("TRADINGVIEW_MCP_MAX_PENDING_ORDERS", 50)
MAX_ALERTS = _env_int("TRADINGVIEW_MCP_MAX_ALERTS", 200)


class CallBudget:
    """A sliding one-minute window over every tool call in this process.

    Process-wide rather than per caller: behind a reverse proxy every request
    arrives from the proxy's address, so there is no caller identity to key
    on. It protects the upstreams and the box, not one user from another.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._calls: deque = deque()
        self._lock = Lock()

    def take(self, now: Optional[float] = None) -> Optional[float]:
        """None when the call may proceed, else seconds until one can."""
        if self.per_minute <= 0:
            return None
        now = time.time() if now is None else now
        with self._lock:
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) >= self.per_minute:
                return round(60.0 - (now - self._calls[0]), 1)
            self._calls.append(now)
            return None


def budget_from_env() -> CallBudget:
    return CallBudget(_env_int("TRADINGVIEW_MCP_RATE_LIMIT_PER_MIN", 0))


def audit_enabled() -> bool:
    return os.environ.get("TRADINGVIEW_MCP_AUDIT_LOG", "").strip() in ("1", "true", "yes")


def audit(tool: str, seconds: float, outcome: str) -> None:
    """One line per tool call, to stderr, where the container log keeps it.

    The tool's name, how long it took and how it ended - never its
    arguments. The proxy route in front of this has no access log of its
    own, because that log would have to contain the secret path.
    """
    print(f"[audit] tool={tool} sec={seconds:.2f} outcome={outcome}",
          file=sys.stderr, flush=True)
