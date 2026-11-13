"""Does an agent reach for the right tool, and then do the right thing?

A description is only as good as the behaviour it produces, and that cannot
be judged by reading it. This turns the question into numbers:

    python evals/tool_selection/score.py export OUT_DIR
        Writes into OUT_DIR, WITHOUT any answers:
          catalogue.json   the tool list exactly as a client receives it
          questions.json   single asks  (cases.json)
          flows.json       multi-step asks  (flows.json)

        Hand them to a model. For questions it writes the first tool it
        would call:

            [{"id": "brief-en", "tool": "market_brief", "args": {...}}, ...]

        For flows it writes every step until it would stop and wait for
        the person. A step is a call or a question to the person:

            [{"id": "close-short", "steps": [
                {"call": "paper_wallet_status", "args": {}},
                {"call": "paper_trade_close", "args": {"position_id": "<from status>"}},
                {"ask": "..."}]}, ...]

    python evals/tool_selection/score.py score ANSWERS.json [LABEL]
        1 for a best tool, 0.5 for an acceptable one, 0 otherwise.
        Argument checks are reported separately.

    python evals/tool_selection/score.py flows ANSWERS.json [LABEL]
        One point per rule in flows.json.

The catalogue comes from importing the server, so no deployment is needed
and what is measured is what the working tree would ship.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
FLOWS = json.loads((HERE / "flows.json").read_text(encoding="utf-8"))["flows"]


def catalogue() -> list:
    from tradingview_mcp import sdk
    from tradingview_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    return [
        {"name": t.name, "description": t.description,
         "inputSchema": sdk.field(t, "inputSchema")}
        for t in tools
    ]


def export(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tools = catalogue()

    def dump(name, value):
        (out / name).write_text(
            json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    dump("catalogue.json", tools)
    dump("questions.json", [{"id": c["id"], "ask": c["ask"]} for c in CASES])
    dump("flows.json", [{"id": f["id"], "ask": f["ask"]} for f in FLOWS])
    size = len(json.dumps(tools))
    print(f"{len(tools)} tools, {size / 1024:.0f} KB, {len(CASES)} questions, "
          f"{len(FLOWS)} flows -> {out}")


# ── single asks ───────────────────────────────────────────────────────────────

def _same(expected, got) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(got) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(got).strip().lower() == str(expected).strip().lower()


def score(answers_path: str, label: str = "") -> dict:
    raw = json.loads(pathlib.Path(answers_path).read_text(encoding="utf-8"))
    answers = {a["id"]: a for a in raw}
    total, rows, arg_checks, arg_ok = 0.0, [], 0, 0
    for case in CASES:
        got = answers.get(case["id"]) or {}
        tool = got.get("tool")
        points = 1.0 if tool in case["best"] else 0.5 if tool in case["ok"] else 0.0
        total += points
        if points < 1.0:
            rows.append((points, case["id"], tool, case["best"]))
        if points and tool in case["best"]:
            for key, want in (case.get("args") or {}).items():
                arg_checks += 1
                arg_ok += _same(want, (got.get("args") or {}).get(key))
    return {
        "label": label,
        "score": total,
        "out_of": len(CASES),
        "percent": round(total * 100 / len(CASES), 1),
        "arguments_right": f"{arg_ok}/{arg_checks}",
        "missed": [
            {"id": i, "points": p, "picked": t, "wanted": b}
            for p, i, t, b in sorted(rows)
        ],
    }


# ── flows ─────────────────────────────────────────────────────────────────────

def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _filled(value) -> bool:
    return value is not None and str(value).strip() not in ("", "null", "None")


def check(rule: dict, steps: list) -> bool:
    """Whether one run of steps satisfies one rule."""
    calls = [(i, s["call"], s.get("args") or {})
             for i, s in enumerate(steps) if s.get("call")]
    asks = [i for i, s in enumerate(steps) if s.get("ask")]

    def first(tool):
        return next(((i, a) for i, name, a in calls if name == tool), None)

    (kind, want), = rule.items()
    if kind == "first_call":
        return bool(calls) and calls[0][1] == want
    if kind == "calls":
        return first(want) is not None
    if kind == "order":
        a, b = first(want[0]), first(want[1])
        return a is not None and b is not None and a[0] < b[0]
    if kind == "arg_set":
        hit = first(want[0])
        return hit is not None and _filled(hit[1].get(want[1]))
    if kind == "arg_not_true":
        hit = first(want[0])
        return hit is None or not _truthy(hit[1].get(want[1]))
    if kind == "never_arg_true":
        return not any(name == want[0] and _truthy(a.get(want[1]))
                       for _, name, a in calls)
    if kind == "ask_before":
        hit = first(want)
        return hit is None or any(i < hit[0] for i in asks)
    raise ValueError(f"unknown rule {kind!r}")


def score_flows(answers_path: str, label: str = "") -> dict:
    raw = json.loads(pathlib.Path(answers_path).read_text(encoding="utf-8"))
    answers = {a["id"]: a.get("steps") or [] for a in raw}
    passed, total, broken = 0, 0, []
    for flow in FLOWS:
        steps = answers.get(flow["id"], [])
        for rule in flow["rules"]:
            total += 1
            if check(rule, steps):
                passed += 1
            else:
                broken.append({"flow": flow["id"], "rule": rule})
    return {
        "label": label,
        "rules_passed": passed,
        "out_of": total,
        "percent": round(passed * 100 / total, 1),
        "broken": broken,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    command = sys.argv[1] if len(sys.argv) >= 3 else ""
    if command == "export":
        export(sys.argv[2])
    elif command == "score":
        print(json.dumps(score(sys.argv[2], *sys.argv[3:4]), indent=1,
                         ensure_ascii=False))
    elif command == "flows":
        print(json.dumps(score_flows(sys.argv[2], *sys.argv[3:4]), indent=1,
                         ensure_ascii=False))
    else:
        print(__doc__)
