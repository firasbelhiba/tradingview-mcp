"""
Sentiment Index Service — crowd mood and sector rotation, keyless.

Two reads that need no API key, which matters because ``market_sentiment``
and ``financial_news`` return ``"Unavailable"`` until MARKETAUX_API_TOKEN is
set. These work regardless.

  * **Fear & Greed** (alternative.me): the composite crypto sentiment index,
    0-100, with history so the direction of travel is visible. A reading of
    32 tells you less than 32-after-65 does.
  * **Narrative rotation** (CoinGecko categories): which sectors are moving.
    The crypto counterpart of ``egx_sector_scan``, which exists for Egyptian
    sectors but had no equivalent here.

A market-cap floor is ON by default for the rotation tool, and that is not a
nicety. Sorted by 24h change the raw category list opens with "Decentralized
Lottery" up 1331% on $0.0B — 778 categories, most of them dust. Without the
floor the tool reports noise with total confidence.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error

_FNG_URL = "https://api.alternative.me/fng/"
_CG_CATEGORIES = "https://api.coingecko.com/api/v3/coins/categories"
_UA = "tradingview-mcp/0.9.0 sentiment"
_TIMEOUT = 20

# These indices move slowly — Fear & Greed updates daily — so a long cache
# keeps a chatty agent from hammering free endpoints that rate-limit hard.
_CACHE_TTL_S = 600.0
_cache: Dict[str, Tuple[float, Any]] = {}
_cache_lock = Lock()

_DEFAULT_MIN_MCAP = 100_000_000.0


def _get(url: str) -> Any:
    with _cache_lock:
        hit = _cache.get(url)
    if hit and (time.time() - hit[0]) <= _ttl():
        return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    with _cache_lock:
        _cache[url] = (time.time(), payload)
    return payload


def _ttl() -> float:
    try:
        return max(0.0, float(os.environ.get("TRADINGVIEW_MCP_SENTIMENT_TTL", _CACHE_TTL_S)))
    except ValueError:
        return _CACHE_TTL_S


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ── Fear & Greed ──────────────────────────────────────────────────────────────

def _classify(value: float) -> str:
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


def get_fear_greed(history_days: int = 30) -> dict:
    """The crypto Fear & Greed index, with history and direction of travel.

    No API key — unlike the Marketaux-backed sentiment tools, which return
    "Unavailable" until a token is configured.

    Args:
        history_days: Days of history to include, 1-365.
    """
    if not 1 <= history_days <= 365:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"history_days must be 1-365, got {history_days}",
        )
    try:
        payload = _get(f"{_FNG_URL}?limit={history_days}")
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"alternative.me unreachable: {str(exc)[:100]}", retryable=True,
        )

    rows = payload.get("data") or []
    series = []
    for row in rows:
        value = _f(row.get("value"))
        if value is None:
            continue
        series.append({
            "value": int(value),
            # The provider's own label is kept rather than recomputed, so the
            # tool never disagrees with the source it is quoting.
            "classification": row.get("value_classification") or _classify(value),
            "timestamp": row.get("timestamp"),
        })
    if not series:
        return make_error(
            ErrorCode.NO_DATA, "alternative.me returned no readings", retryable=True
        )

    current = series[0]
    values = [s["value"] for s in series]
    oldest = series[-1]
    mean = sum(values) / len(values)
    change = current["value"] - oldest["value"] if len(series) > 1 else None

    return {
        "source": "alternative.me Fear & Greed index (public, no key)",
        "current": current,
        "scale": "0 = extreme fear, 100 = extreme greed",
        "history_days": len(series),
        "range": {"min": min(values), "max": max(values)},
        "mean": round(mean, 1),
        "change_over_window": change,
        "direction": (
            None if change is None
            else "toward greed" if change > 5
            else "toward fear" if change < -5 else "flat"
        ),
        "reading_note": (
            "A level alone says little — 32 after 65 is a different market "
            "from 32 after 20. change_over_window and direction give that "
            "context."
        ),
        "history": series,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── Narrative rotation ────────────────────────────────────────────────────────

def get_narrative_rotation(
    min_market_cap_usd: Optional[float] = _DEFAULT_MIN_MCAP,
    limit: int = 15,
    order: str = "gainers",
) -> dict:
    """Which crypto sectors are moving, by market-cap-weighted category.

    The counterpart of ``egx_sector_scan`` for crypto.

    ``min_market_cap_usd`` defaults ON. CoinGecko lists 778 categories and
    sorting the raw list by 24h change opens with dust: "Decentralized
    Lottery" up 1331% on a market cap that rounds to $0.0B. The floor is what
    makes the output mean anything; pass ``None`` only if you want the dust.

    Args:
        min_market_cap_usd: Category market-cap floor. None disables it.
        limit:              Rows to return, 1-50.
        order:              ``gainers`` or ``losers`` first.
    """
    if not 1 <= limit <= 50:
        return make_error(ErrorCode.INVALID_PARAMETER, f"limit must be 1-50, got {limit}")
    if order not in ("gainers", "losers"):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"order must be 'gainers' or 'losers', got {order!r}",
        )
    try:
        payload = _get(f"{_CG_CATEGORIES}?order=market_cap_desc")
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"CoinGecko unreachable: {str(exc)[:100]}", retryable=True,
        )
    if not isinstance(payload, list) or not payload:
        return make_error(
            ErrorCode.NO_DATA, "CoinGecko returned no categories", retryable=True
        )

    rows: List[dict] = []
    below_floor = 0
    no_change_data = 0
    for cat in payload:
        mcap = _f(cat.get("market_cap"))
        change = _f(cat.get("market_cap_change_24h"))
        if change is None:
            no_change_data += 1
            continue
        if min_market_cap_usd is not None and (mcap or 0) < min_market_cap_usd:
            below_floor += 1
            continue
        rows.append({
            "name": cat.get("name"),
            "category_id": cat.get("id"),
            "market_cap_usd": mcap,
            "market_cap_change_24h_pct": round(change, 2),
            "volume_24h_usd": _f(cat.get("volume_24h")),
            "top_coins": [
                c for c in (cat.get("top_3_coins_id") or [])[:3] if c
            ] or None,
        })

    if not rows:
        return make_error(
            ErrorCode.NO_DATA,
            "No category passed the market-cap floor",
            min_market_cap_usd=min_market_cap_usd,
            categories_seen=len(payload),
        )

    rows.sort(key=lambda r: r["market_cap_change_24h_pct"],
              reverse=(order == "gainers"))
    changes = sorted(r["market_cap_change_24h_pct"] for r in rows)
    return {
        "source": "CoinGecko categories (public, no key)",
        "categories_seen": len(payload),
        "below_market_cap_floor": below_floor,
        "without_change_data": no_change_data,
        "ranked_categories": len(rows),
        "accounting_note": (
            "ranked_categories + below_market_cap_floor + without_change_data "
            "= categories_seen, so nothing disappears unexplained."
        ),
        "median_change_24h_pct": changes[len(changes) // 2],
        "advancing": sum(1 for c in changes if c > 0),
        "declining": sum(1 for c in changes if c < 0),
        "order": order,
        "rows": rows[:limit],
        "filters": {"min_market_cap_usd": min_market_cap_usd},
        "floor_note": (
            "The market-cap floor is on by default: unfiltered, this list is "
            "led by categories worth almost nothing posting four-digit "
            "percentage moves."
        ),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
