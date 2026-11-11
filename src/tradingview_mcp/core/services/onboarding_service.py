"""
Onboarding Service — where to start, and how to make alerts actually alert.

Two tools that answer questions the other 72 cannot:

  * ``get_started`` looks at the real state of this installation — is there a
    wallet, are there alerts, is the Marketaux key set, does the data layer
    answer — and returns an ordered list of next steps. Not a feature tour: a
    feature list is what a README is for, and whoever needs this is precisely
    the person who did not read one.
  * ``alert_schedule_recipe`` closes the gap that makes alerts disappointing.
    They are stored and evaluated on demand, so a condition met at 3am is
    noticed at 9am. MCP has no way for a server to wake anyone up, so this
    returns the exact instruction to hand a scheduler — the honest answer,
    rather than pretending the server can push.

Neither invents anything. Both report measured state and say plainly what is
missing.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

_MARKETAUX_TOOLS = ("market_sentiment", "financial_news", "combined_analysis")


def _wallet_state() -> Dict[str, Any]:
    from tradingview_mcp.core.services import paper_store as store
    wallet = store.get_wallet()
    if not wallet:
        return {"exists": False}
    positions = store.list_positions()
    closed = [p for p in positions if p["status"] == store.CLOSED]
    return {
        "exists": True,
        "currency": wallet["currency"],
        "starting_balance": wallet["starting_balance"],
        "open": len(positions) - len(closed),
        "closed": len(closed),
    }


def _alert_state() -> Dict[str, Any]:
    from tradingview_mcp.core.services import alert_store as store
    alerts = store.list_alerts()
    return {
        "total": len(alerts),
        "armed": sum(1 for a in alerts if a["state"] == store.ARMED),
        "triggered": sum(1 for a in alerts if a["state"] == store.TRIGGERED),
    }


def _data_layer_state() -> Dict[str, Any]:
    """One cheap live call, so "is it working?" is answered, not assumed."""
    from tradingview_mcp.core.services.screener_provider import resilient_scan_columns
    try:
        rows = resilient_scan_columns("crypto", "1D", ["KUCOIN:BTCUSDT"], ["close"])
        price = (rows.get("KUCOIN:BTCUSDT") or {}).get("close")
        if price:
            return {"working": True, "btc_price": price}
        return {
            "working": False,
            "reason": "the scanner returned no rows — usually throttling after "
                      "a burst of requests; it clears within a minute",
        }
    except Exception as exc:  # noqa: BLE001
        return {"working": False, "reason": str(exc)[:120]}


def get_started() -> dict:
    """What to do next, based on what this installation actually has."""
    wallet = _wallet_state()
    alerts = _alert_state()
    data = _data_layer_state()
    marketaux = bool(os.environ.get("MARKETAUX_API_TOKEN"))

    steps: List[Dict[str, str]] = []

    if not data["working"]:
        steps.append({
            "step": "The market data layer is not answering right now",
            "why": data.get("reason", ""),
            "do": "Wait a minute and ask again — most tools depend on it.",
        })

    if not wallet["exists"]:
        steps.append({
            "step": "Create a practice wallet",
            "why": "It is the only way to find out whether a setup works "
                   "before it costs anything. Nothing here can place a real "
                   "order.",
            "do": "Decide a starting balance — it is deliberately required, "
                  "because it silently defines what every later number means "
                  "— then ask for a paper wallet with it.",
        })
    elif wallet["closed"] == 0:
        steps.append({
            "step": "Take a practice trade",
            "why": f"The wallet has {wallet['open']} open and no closed "
                   "trades, so there is no record to learn from yet.",
            "do": "Ask for a trade plan on a pair you follow, then open it on "
                  "the wallet with a stop and a target.",
        })
    else:
        steps.append({
            "step": "Review what the practice trades actually did",
            "why": f"{wallet['closed']} closed trade(s) are enough for a "
                   "first read, including what fees and funding consumed.",
            "do": "Ask for the paper trade history with its equity curve.",
        })

    if alerts["total"] == 0:
        steps.append({
            "step": "Set one standing alert",
            "why": "Everything else here only answers when asked. An alert "
                   "is the one thing that waits.",
            "do": "Pick a condition that would change your mind about a "
                  "position — funding flipping negative, breadth dropping "
                  "under 10% — and ask for an alert on it.",
        })
    else:
        steps.append({
            "step": "Have the alerts checked on a schedule",
            "why": f"{alerts['armed']} alert(s) are armed, but nothing fires "
                   "them: they are evaluated only when asked, so a condition "
                   "met overnight is noticed in the morning.",
            "do": "Ask for the alert schedule recipe and give it to a "
                  "scheduled task.",
        })

    if not marketaux:
        steps.append({
            "step": "Optional: news sentiment needs a free key",
            "why": f"Without MARKETAUX_API_TOKEN, {', '.join(_MARKETAUX_TOOLS)} "
                   "return 'Unavailable'. fear_greed needs no key and covers "
                   "the same ground.",
            "do": "Get a free token at marketaux.com and set "
                  "MARKETAUX_API_TOKEN, or ignore this and use fear_greed.",
        })

    return {
        "state": {
            "paper_wallet": wallet,
            "alerts": alerts,
            "market_data": data,
            "marketaux_key_set": marketaux,
        },
        "next_steps": steps,
        "try_asking": [
            "what's the market doing?",
            "where is the crowd most exposed right now?",
            "give me a trade plan for BTC on the 4h",
            "how is my paper wallet?",
        ],
        "note": (
            "Every tool here is read-only against the market: nothing places "
            "an order, and the wallet is a simulation."
        ),
        "display_hint": (
            "If the person is looking at this, draw next_steps as a short "
            "ordered list (see this server's PRESENTATION instructions); keep "
            "it to the steps, not a feature tour."
        ),
    }


def alert_schedule_recipe(every_minutes: int = 15) -> dict:
    """The instruction to hand a scheduler so alerts actually fire.

    MCP is request/response: a server cannot wake anyone. What it can do is
    say exactly what a scheduler should run, and what the result means.

    Args:
        every_minutes: How often to check, 5-1440.
    """
    from tradingview_mcp.core.errors import ErrorCode, make_error

    if not 5 <= every_minutes <= 1440:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"every_minutes must be 5-1440, got {every_minutes}",
        )

    alerts = _alert_state()
    return {
        "why": (
            "Alerts are stored and evaluated on demand, so nothing fires them "
            "on its own. A scheduled task that calls alert_check turns them "
            "into real alerts."
        ),
        "armed_now": alerts["armed"],
        "schedule": {
            "every_minutes": every_minutes,
            "cron": f"*/{every_minutes} * * * *" if every_minutes < 60
                    else f"0 */{every_minutes // 60} * * *",
        },
        "task_prompt": (
            "Run alert_check on the tradingview server. If fired is empty, "
            "reply with nothing but 'no alerts'. If anything fired, report "
            "each one's condition and the value that triggered it. If "
            "not_evaluated is non-empty, say so explicitly — those conditions "
            "were NOT checked, which is different from not being met."
        ),
        "how": [
            "Create a scheduled task with the prompt above at the cadence in "
            "schedule.",
            "Point it at this server — the task needs the same connector you "
            "are using now.",
            "Keep the cadence honest: a 15-minute check cannot catch a wick "
            "that lasted seconds, and checking every minute mostly produces "
            "upstream throttling.",
        ],
        "caveats": [
            "The check reads the price AT CHECK TIME, so a condition met and "
            "unmet between two checks is missed entirely.",
            "A one-shot alert disarms when it fires; create it with "
            "repeat_after_fire if it should keep watching.",
            "The alerts live in a local database on this machine, so the "
            "scheduled task has to run on this machine too.",
        ],
        "display_hint": (
            "If the person is looking at this, show the schedule and the "
            "task_prompt plainly (see this server's PRESENTATION "
            "instructions); the caveats matter more than the steps."
        ),
    }
