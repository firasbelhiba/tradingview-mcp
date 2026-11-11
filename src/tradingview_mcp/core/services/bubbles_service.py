"""
Bubbles — the top of the market by size, with how far each has moved.

One picture answers a question no table does quickly: is this a broad move
or a handful of names, and is the money leaving the big caps or the small
ones. Every coin arrives with its change over five windows at once, so the
view can switch between an hour and a year without another request.

Source is CoinGecko's public markets endpoint: no key, one call, and its own
market-cap ranking. That ranking is theirs, not ours — a coin's rank moves
with whatever supply figure they use, and this does not second-guess it.

Two things the numbers are not:

  * **Price change, not market-cap change.** A coin that unlocked supply can
    fall in price while its market cap holds. The field is named for what it
    measures.
  * **Stablecoins are excluded by default.** They sit high in the ranking by
    size and never move, so they take two of the biggest bubbles to say
    nothing. The count that was removed is reported rather than silently
    dropped.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from threading import Lock
from typing import Any, Dict, List, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.display_hints import hint

_DISPLAY_HINT = hint(
    "draw it as bubbles sized by market cap and coloured by change; "
    "compress the radius or bitcoin is the only one visible."
)

_URL = "https://api.coingecko.com/api/v3/coins/markets"
_UA = "tradingview-mcp/0.9"
_TIMEOUT = 15
_TTL = 60.0
_STABLE_TTL = 3600.0
_MAX_LIMIT = 100

_cache: Dict[str, tuple] = {}
_cache_lock = Lock()

# Window -> the field CoinGecko returns it in.
WINDOWS = {
    "1h": "price_change_percentage_1h_in_currency",
    "24h": "price_change_percentage_24h_in_currency",
    "7d": "price_change_percentage_7d_in_currency",
    "30d": "price_change_percentage_30d_in_currency",
    "1y": "price_change_percentage_1y_in_currency",
}

# The fallback list, used when CoinGecko's own stablecoin category cannot be
# reached. Judged by what the asset is for, never by a price check: a
# depegged stablecoin is still a stablecoin, and a price test would surface
# it only on the day its bubble is most misleading.
_STABLES = {
    "usdt", "usdc", "dai", "fdusd", "usde", "pyusd", "tusd", "usdd", "usds",
    "busd", "gusd", "lusd", "frax", "usdp", "eurc", "eurs", "usd1", "rlusd",
    "susds", "susde", "buidl", "usdy", "usdf", "usdx", "usdg",
    "usyc", "usdtb", "syrupusdc", "xaut",
}


def _stablecoin_symbols() -> set:
    """The stablecoin list, from CoinGecko's own category rather than by hand.

    The hand-written set below was already leaking: a live top-60 let USDG
    and USYC through, because new ones arrive faster than anyone edits a
    constant. This asks the source that ranks them, cached for an hour, and
    falls back to the literal set when the call fails — so a failed lookup
    degrades to the old behaviour instead of filling the board with coins
    built not to move.
    """
    key = "stablecoins"
    with _cache_lock:
        hit = _cache.get(key)
    if hit and (time.time() - hit[0]) <= _STABLE_TTL:
        return hit[1]
    url = (
        f"{_URL}?vs_currency=usd&category=stablecoins&order=market_cap_desc"
        "&per_page=250&page=1&sparkline=false"
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": _UA, "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        symbols = {
            str(r.get("symbol", "")).lower() for r in rows if r.get("symbol")
        }
        if not symbols:
            raise RuntimeError("empty stablecoin category")
    except Exception:  # noqa: BLE001
        return set(_STABLES)
    symbols |= _STABLES
    with _cache_lock:
        _cache[key] = (time.time(), symbols)
    return symbols


def _fetch(limit: int) -> List[dict]:
    key = f"markets:{limit}"
    with _cache_lock:
        hit = _cache.get(key)
    if hit and (time.time() - hit[0]) <= _TTL:
        return hit[1]

    url = (
        f"{_URL}?vs_currency=usd&order=market_cap_desc&per_page={limit}"
        "&page=1&sparkline=false"
        "&price_change_percentage=1h%2C24h%2C7d%2C30d%2C1y"
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("CoinGecko returned an unexpected shape")
    with _cache_lock:
        _cache[key] = (time.time(), rows)
    return rows


def _row(raw: dict) -> dict:
    changes = {
        window: (
            None if raw.get(field) is None else round(float(raw[field]), 2)
        )
        for window, field in WINDOWS.items()
    }
    return {
        "rank": raw.get("market_cap_rank"),
        "symbol": (raw.get("symbol") or "").upper(),
        "name": raw.get("name"),
        # CoinGecko's own logo. Served from coin-images.coingecko.com, which
        # some sandboxed renderers refuse on content-security grounds — a
        # client that cannot reach it should fall back to the ticker rather
        # than show a hole.
        "logo": raw.get("image"),
        "price": raw.get("current_price"),
        "market_cap": raw.get("market_cap"),
        "volume_24h": raw.get("total_volume"),
        "change_pct": changes,
    }


def crypto_bubbles(limit: int = 30, window: str = "24h",
                   include_stablecoins: bool = False) -> dict:
    """The largest coins by market cap, with how far each has moved.

    Args:
        limit:               How many coins, 1-100.
        window:              Which change to lead with: 1h, 24h, 7d, 30d, 1y.
                             Every window is returned regardless.
        include_stablecoins: Keep the coins that are built not to move.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return make_error(ErrorCode.INVALID_PARAMETER, "limit must be a number")
    if not 1 <= limit <= _MAX_LIMIT:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"limit must be 1-{_MAX_LIMIT}, got {limit}"
        )
    window = str(window or "24h").strip().lower()
    if window not in WINDOWS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"window must be one of {', '.join(WINDOWS)}, got {window!r}",
            valid_windows=list(WINDOWS),
        )

    # Over-fetch so dropping stablecoins still leaves `limit` coins, rather
    # than quietly returning 48 when 50 were asked for. A flat +10 was not
    # enough: a live top-60 held twelve stablecoins, because they cluster at
    # the large-cap end where the buffer is thinnest. Half again plus ten
    # covers that, and CoinGecko charges the same for one page either way.
    want = (
        min(_MAX_LIMIT, int(limit * 1.5) + 10)
        if not include_stablecoins else limit
    )
    try:
        raw = _fetch(want)
    except urllib.error.HTTPError as exc:
        retryable = exc.code in (429, 500, 502, 503, 504)
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"CoinGecko returned HTTP {exc.code}"
            + (" - the free tier is rate limited, try again in a minute"
               if exc.code == 429 else ""),
            retryable=retryable,
        )
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"CoinGecko unreachable: {str(exc)[:100]}", retryable=True,
        )

    rows = [_row(r) for r in raw]
    removed = []
    if not include_stablecoins:
        stables = _stablecoin_symbols()
        kept = []
        for row in rows:
            if (row["symbol"] or "").lower() in stables:
                removed.append(row["symbol"])
            else:
                kept.append(row)
        rows = kept
    rows = rows[:limit]

    if not rows:
        return make_error(
            ErrorCode.NO_DATA, "CoinGecko returned no coins", retryable=True
        )

    moves = [r["change_pct"][window] for r in rows
             if r["change_pct"][window] is not None]
    up = sum(1 for m in moves if m > 0)
    down = sum(1 for m in moves if m < 0)
    ranked = sorted(
        (r for r in rows if r["change_pct"][window] is not None),
        key=lambda r: r["change_pct"][window],
    )

    def _named(row: Optional[dict]) -> Optional[dict]:
        if not row:
            return None
        return {"symbol": row["symbol"], "change_pct": row["change_pct"][window]}

    return {
        "window": window,
        "coins": rows,
        "count": len(rows),
        "windows_available": list(WINDOWS),
        "breadth": {
            "up": up,
            "down": down,
            "unchanged": len(moves) - up - down,
            "measured": len(moves),
            "median_change_pct": (
                round(sorted(moves)[len(moves) // 2], 2) if moves else None
            ),
            "best": _named(ranked[-1] if ranked else None),
            "worst": _named(ranked[0] if ranked else None),
        },
        "total_market_cap": sum(
            r["market_cap"] for r in rows if r["market_cap"]
        ),
        "stablecoins_removed": removed or None,
        "source": "CoinGecko public markets endpoint (no key), cached 60s",
        "measures": (
            "change_pct is PRICE change, not market-cap change. A coin that "
            "unlocked supply can fall in price while its cap holds. Ranking "
            "and market caps are CoinGecko's own."
        ),
        "display_hint": _DISPLAY_HINT,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }

