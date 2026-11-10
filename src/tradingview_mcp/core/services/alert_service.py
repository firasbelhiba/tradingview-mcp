"""
Alert Service — standing conditions, evaluated on demand.

Every other tool answers a question you remember to ask. These let a
condition wait: "funding on BTC turns negative", "breadth drops under 10%",
"ETH RSI falls below 25". Nothing here pushes a notification — MCP is
request/response — so the loop is a scheduler calling ``alert_check`` and
relaying what fired.

Two deliberate limits:

  * **Conditions are a fixed vocabulary, not expressions.** A metric, an
    operator and a threshold. No eval, no parser, no arbitrary code from a
    stored string — and every condition is therefore checkable by a test.
  * **``above``/``below`` test the value at check time**, not a crossing.
    Crossing detection needs a reliable previous value for every alert, and
    between two checks hours apart a "cross" is a guess about what happened
    in the gap. ``last_value`` is recorded so a caller can see the move.

Checks are batched: every scanner-backed metric for every alert on one
exchange costs ONE upstream request, however many alerts there are.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services import alert_store as store
from tradingview_mcp.core.services.indicators import RELATIVE_VOLUME
from tradingview_mcp.core.services.screener_provider import resilient_scan_columns
from tradingview_mcp.core.utils.validators import (
    EXCHANGE_SCREENER,
    normalize_tradingview_symbol,
)

OPERATORS = ("above", "below")

# metric -> (source, scanner columns needed, human description)
_SCANNER = "scanner"
_POSITIONING = "positioning"
_BREADTH = "breadth"

METRICS: Dict[str, Dict[str, Any]] = {
    "price": {
        "source": _SCANNER, "columns": ["close"],
        "unit": "quote currency",
        "description": "last price",
    },
    "rsi": {
        "source": _SCANNER, "columns": ["RSI"],
        "unit": "0-100",
        "description": "RSI at the alert's timeframe",
    },
    "change_pct": {
        "source": _SCANNER, "columns": ["change"],
        "unit": "%",
        "description": "change on the current bar",
    },
    "volume_ratio": {
        "source": _SCANNER, "columns": [RELATIVE_VOLUME],
        "unit": "x",
        "description": "volume vs the 10-bar mean",
    },
    "ema50_distance_pct": {
        "source": _SCANNER, "columns": ["close", "EMA50"],
        "unit": "%",
        "description": "distance from EMA50, negative when below",
    },
    "funding_annualised_pct": {
        "source": _POSITIONING, "columns": [],
        "unit": "%/yr",
        "description": "perpetual funding, annualised; negative means shorts pay",
    },
    "funding_pct": {
        "source": _POSITIONING, "columns": [],
        "unit": "% per interval",
        "description": "perpetual funding for the current interval",
    },
    "oi_change_pct": {
        "source": _POSITIONING, "columns": [],
        "unit": "%",
        "description": "open-interest change over the sampling window",
    },
    "crowd_long_pct": {
        "source": _POSITIONING, "columns": [],
        "unit": "%",
        "description": "share of futures accounts positioned long",
    },
    "breadth_above_ema50_pct": {
        "source": _BREADTH, "columns": [],
        "unit": "%",
        "description": "share of an exchange above its EMA50 (market-wide, "
                       "symbol is ignored)",
    },
}


def metric_catalogue() -> List[dict]:
    return [
        {"metric": name, "unit": spec["unit"], "source": spec["source"],
         "description": spec["description"]}
        for name, spec in sorted(METRICS.items())
    ]


def _screener_for(exchange: str) -> str:
    return EXCHANGE_SCREENER.get((exchange or "").strip().lower(), "crypto")


# ── Create / list / delete ────────────────────────────────────────────────────

def create_alert(symbol: str, metric: str, operator: str, threshold: float,
                 exchange: str = "KUCOIN", timeframe: str = "4h",
                 note: Optional[str] = None,
                 repeat_after_fire: bool = False) -> dict:
    """Store a standing condition. Returns the stored alert, including its id."""
    metric_key = (metric or "").strip().lower()
    if metric_key not in METRICS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Unknown metric {metric!r}",
            valid_metrics=sorted(METRICS),
        )
    op = (operator or "").strip().lower()
    if op not in OPERATORS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"Unknown operator {operator!r}",
            valid_operators=list(OPERATORS),
        )
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"threshold must be a number, got {threshold!r}"
        )
    if not symbol or not str(symbol).strip():
        if METRICS[metric_key]["source"] != _BREADTH:
            return make_error(ErrorCode.INVALID_PARAMETER, "symbol is required")
        symbol = "*"

    stored = store.add_alert(
        symbol=str(symbol).strip().upper(),
        exchange=(exchange or "KUCOIN").strip().upper(),
        metric=metric_key,
        operator=op,
        threshold=threshold,
        timeframe=timeframe,
        note=note,
        repeat_after_fire=repeat_after_fire,
    )
    stored["condition"] = _describe(stored)
    stored["unit"] = METRICS[metric_key]["unit"]
    return stored


def _describe(alert: Dict[str, Any]) -> str:
    spec = METRICS.get(alert["metric"], {})
    unit = spec.get("unit", "")
    target = alert["symbol"] if alert["symbol"] != "*" else alert["exchange"]
    return (
        f"{target} {alert['metric']} {alert['operator']} {alert['threshold']}"
        + (f" {unit}" if unit and unit != "quote currency" else "")
    )


def list_alerts(state: Optional[str] = None) -> dict:
    if state not in (None, store.ARMED, store.TRIGGERED):
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"state must be {store.ARMED!r}, {store.TRIGGERED!r} or omitted",
        )
    rows = store.list_alerts(state)
    for r in rows:
        r["condition"] = _describe(r)
        r["unit"] = METRICS.get(r["metric"], {}).get("unit")
    return {
        "alerts": rows,
        "count": len(rows),
        "armed": sum(1 for r in rows if r["state"] == store.ARMED),
        "triggered": sum(1 for r in rows if r["state"] == store.TRIGGERED),
        "database": str(store.db_path()),
        "filter": state,
    }


def delete_alert(alert_id: int) -> dict:
    existing = store.get_alert(alert_id)
    if not existing:
        return make_error(
            ErrorCode.SYMBOL_NOT_FOUND, f"No alert with id {alert_id}", alert_id=alert_id
        )
    store.delete_alert(alert_id)
    return {"deleted": True, "alert_id": alert_id, "was": _describe(existing)}


def rearm_alert(alert_id: int) -> dict:
    existing = store.get_alert(alert_id)
    if not existing:
        return make_error(
            ErrorCode.SYMBOL_NOT_FOUND, f"No alert with id {alert_id}", alert_id=alert_id
        )
    store.rearm(alert_id)
    return {"rearmed": True, "alert_id": alert_id, "condition": _describe(existing)}


# ── Evaluation ────────────────────────────────────────────────────────────────

def _fetch_scanner_values(alerts: List[dict]) -> Tuple[Dict[int, Optional[float]], Dict[int, str]]:
    """Resolve every scanner-backed alert, one request per (exchange, timeframe).

    Batching matters: fifty price alerts on one exchange are one upstream call,
    not fifty. Alerts are grouped by the pair that defines a scanner request.
    """
    values: Dict[int, Optional[float]] = {}
    errors: Dict[int, str] = {}

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for a in alerts:
        groups.setdefault((a["exchange"], a["timeframe"]), []).append(a)

    for (exchange, timeframe), batch in groups.items():
        columns = sorted({
            col for a in batch for col in METRICS[a["metric"]]["columns"]
        })
        tickers = sorted({normalize_tradingview_symbol(a["symbol"], exchange) for a in batch})
        try:
            rows = resilient_scan_columns(
                _screener_for(exchange), timeframe, tickers, columns
            )
        except Exception as exc:  # noqa: BLE001
            for a in batch:
                errors[a["id"]] = f"scanner request failed: {str(exc)[:80]}"
            continue
        if not rows:
            for a in batch:
                errors[a["id"]] = (
                    "scanner returned no rows — likely throttling; the condition "
                    "was NOT evaluated"
                )
            continue
        for a in batch:
            ticker = normalize_tradingview_symbol(a["symbol"], exchange)
            row = rows.get(ticker)
            if not row:
                errors[a["id"]] = f"{ticker} unknown to this exchange's scanner"
                continue
            values[a["id"]] = _scanner_metric(a["metric"], row)
            if values[a["id"]] is None:
                errors[a["id"]] = f"no {a['metric']} data for {ticker} at {timeframe}"
    return values, errors


def _scanner_metric(metric: str, row: dict) -> Optional[float]:
    if metric == "price":
        return row.get("close")
    if metric == "rsi":
        return row.get("RSI")
    if metric == "change_pct":
        return row.get("change")
    if metric == "volume_ratio":
        return row.get(RELATIVE_VOLUME)
    if metric == "ema50_distance_pct":
        close, ema = row.get("close"), row.get("EMA50")
        if not close or not ema:
            return None
        return (close - ema) / ema * 100.0
    return None


def _positioning_value(alert: dict) -> Tuple[Optional[float], Optional[str]]:
    from tradingview_mcp.core.services.positioning_service import analyze_positioning

    out = analyze_positioning(alert["symbol"], alert["timeframe"])
    if "error" in out:
        return None, f"positioning unavailable: {out['error'].get('code')}"
    metric = alert["metric"]
    if metric == "funding_annualised_pct":
        return out["funding"].get("annualised_pct"), None
    if metric == "funding_pct":
        return out["funding"].get("pct_per_interval"), None
    if metric == "oi_change_pct":
        return out["open_interest"].get("change_pct"), None
    if metric == "crowd_long_pct":
        return out["crowd"].get("long_pct"), None
    return None, f"metric {metric} has no positioning mapping"


def _breadth_value(alert: dict) -> Tuple[Optional[float], Optional[str]]:
    from tradingview_mcp.core.services.market_structure_service import (
        analyze_market_breadth,
    )

    out = analyze_market_breadth(alert["exchange"], alert["timeframe"])
    if "error" in out:
        return None, f"breadth unavailable: {out['error'].get('code')}"
    return out["trend_participation"].get("above_ema50_pct"), None


def _condition_met(operator: str, value: float, threshold: float) -> bool:
    return value > threshold if operator == "above" else value < threshold


def check_alerts(include_triggered: bool = False) -> dict:
    """Evaluate armed alerts and report which fired.

    Nothing is pushed anywhere — the caller relays what comes back. An alert
    that fires is marked triggered and stops being checked, unless it was
    created with ``repeat_after_fire``.

    Alerts whose data could not be fetched are reported under ``not_evaluated``
    rather than counted as "condition not met": a dead feed must not read as a
    quiet market.
    """
    alerts = store.list_alerts(None if include_triggered else store.ARMED)
    if not alerts:
        return {
            "checked": 0, "fired": [], "still_waiting": [], "not_evaluated": [],
            "note": "no alerts are armed — create one with alert_create",
            "database": str(store.db_path()),
        }

    by_source: Dict[str, List[dict]] = {}
    for a in alerts:
        source = METRICS.get(a["metric"], {}).get("source")
        by_source.setdefault(source, []).append(a)

    values: Dict[int, Optional[float]] = {}
    errors: Dict[int, str] = {}

    if by_source.get(_SCANNER):
        v, e = _fetch_scanner_values(by_source[_SCANNER])
        values.update(v)
        errors.update(e)
    for a in by_source.get(_POSITIONING, []):
        val, err = _positioning_value(a)
        values[a["id"]] = val
        if err:
            errors[a["id"]] = err
    for a in by_source.get(_BREADTH, []):
        val, err = _breadth_value(a)
        values[a["id"]] = val
        if err:
            errors[a["id"]] = err

    fired: List[dict] = []
    waiting: List[dict] = []
    not_evaluated: List[dict] = []

    for a in alerts:
        entry = {
            "alert_id": a["id"],
            "condition": _describe(a),
            "symbol": a["symbol"],
            "exchange": a["exchange"],
            "metric": a["metric"],
            "timeframe": a["timeframe"],
            "threshold": a["threshold"],
            "unit": METRICS.get(a["metric"], {}).get("unit"),
            "note": a.get("note"),
        }
        err = errors.get(a["id"])
        value = values.get(a["id"])

        if value is None:
            entry["reason"] = err or "no value returned"
            not_evaluated.append(entry)
            store.record_check(a["id"], None, entry["reason"])
            continue

        entry["value"] = round(value, 8)
        if _condition_met(a["operator"], value, a["threshold"]):
            entry["previous_value"] = a.get("last_value")
            fired.append(entry)
            if a["repeat_after_fire"]:
                store.record_check(a["id"], value)
            else:
                store.mark_triggered(a["id"], value)
                entry["disarmed"] = True
        else:
            distance = (
                a["threshold"] - value if a["operator"] == "above"
                else value - a["threshold"]
            )
            entry["distance_to_trigger"] = round(distance, 8)
            waiting.append(entry)
            store.record_check(a["id"], value)

    return {
        "checked": len(alerts),
        "fired": fired,
        "fired_count": len(fired),
        "still_waiting": waiting,
        "not_evaluated": not_evaluated,
        "upstream_problem": bool(not_evaluated) or None,
        "database": str(store.db_path()),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }
