"""The yardstick is code too, and a wrong rule reports a wrong score. Offline."""
import importlib.util
import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "evals" / "tool_selection"
_spec = importlib.util.spec_from_file_location("tool_selection_score", _ROOT / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def _call(tool, **args):
    return {"call": tool, "args": args}


ASK = {"ask": "how much?"}


@pytest.mark.parametrize("rule, steps, expected", [
    ({"first_call": "a"}, [_call("a"), _call("b")], True),
    ({"first_call": "a"}, [ASK, _call("a")], True),
    ({"first_call": "a"}, [_call("b"), _call("a")], False),
    ({"first_call": "a"}, [ASK], False),
    ({"calls": "a"}, [_call("b"), _call("a")], True),
    ({"calls": "a"}, [_call("b")], False),
    ({"order": ["a", "b"]}, [_call("a"), _call("x"), _call("b")], True),
    ({"order": ["a", "b"]}, [_call("b"), _call("a")], False),
    ({"order": ["a", "b"]}, [_call("a")], False),
    ({"arg_set": ["a", "wallet"]}, [_call("a", wallet="scalps")], True),
    ({"arg_set": ["a", "wallet"]}, [_call("a", wallet=None)], False),
    ({"arg_set": ["a", "wallet"]}, [_call("a")], False),
    ({"arg_not_true": ["a", "confirm"]}, [_call("a", confirm=False), _call("a", confirm=True)], True),
    ({"arg_not_true": ["a", "confirm"]}, [_call("a", confirm=True)], False),
    ({"arg_not_true": ["a", "confirm"]}, [_call("a", confirm="true")], False),
    ({"arg_not_true": ["a", "confirm"]}, [ASK], True),
    ({"never_arg_true": ["a", "wipe"]}, [_call("a"), _call("a", wipe=True)], False),
    ({"never_arg_true": ["a", "wipe"]}, [_call("a", wipe=False)], True),
    ({"ask_before": "a"}, [ASK, _call("a")], True),
    ({"ask_before": "a"}, [_call("a"), ASK], False),
    ({"ask_before": "a"}, [_call("b")], True),
])
def test_each_rule_means_what_it_says(rule, steps, expected):
    assert score.check(rule, steps) is expected


def test_an_unknown_rule_fails_loudly_rather_than_passing():
    with pytest.raises(ValueError):
        score.check({"sometimes": "a"}, [])


def test_every_tool_an_answer_key_names_exists():
    """A key that names a tool the server no longer has can never be scored
    right, and would read as the descriptions getting worse."""
    names = {t["name"] for t in score.catalogue()}
    wanted = set()
    for case in score.CASES:
        wanted |= set(case["best"]) | set(case["ok"])
    for flow in score.FLOWS:
        for rule in flow["rules"]:
            (_, value), = rule.items()
            wanted |= {value} if isinstance(value, str) else {value[0]}
            if rule.get("order"):
                wanted |= set(rule["order"])
    assert wanted <= names, sorted(wanted - names)


def test_every_argument_an_answer_key_checks_is_a_real_parameter():
    """This one was wrong on the first run: the key said starting_balance,
    the tool says balance, and a correct answer was marked down."""
    props = {t["name"]: set(t["inputSchema"].get("properties") or {})
             for t in score.catalogue()}
    for case in score.CASES:
        for key in case.get("args") or {}:
            for tool in case["best"]:
                assert key in props[tool], f"{case['id']}: {tool} has no {key}"
    for flow in score.FLOWS:
        for rule in flow["rules"]:
            (kind, value), = rule.items()
            if kind in ("arg_set", "arg_not_true", "never_arg_true"):
                assert value[1] in props[value[0]], f"{flow['id']}: {value}"


def test_scoring_counts_best_acceptable_and_wrong(tmp_path):
    answers = [{"id": c["id"], "tool": c["best"][0], "args": c.get("args") or {}}
               for c in score.CASES]
    answers[0]["tool"] = "no_such_tool"
    path = tmp_path / "a.json"
    path.write_text(json.dumps(answers), encoding="utf-8")
    out = score.score(str(path))
    assert out["score"] == len(score.CASES) - 1
    assert out["missed"][0]["picked"] == "no_such_tool"
