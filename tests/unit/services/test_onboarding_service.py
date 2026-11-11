"""Unit tests for the first-run path and the alert schedule recipe. Offline.

Two things matter here. The steps must follow the state that was actually
measured — a wallet that exists must not be offered again, a key that is set
must not be asked for. And the recipe must keep its caveats, because a
schedule that quietly misses a wick is worse than no schedule at all.
"""
import pytest

from tradingview_mcp.core.services import onboarding_service as svc

_WORKING = {"working": True, "btc_price": 73000.0}


def _patch(monkeypatch, wallet=None, alerts=None, data=None, key=False):
    monkeypatch.setattr(svc, "_wallet_state",
                        lambda: wallet if wallet is not None else {"exists": False})
    monkeypatch.setattr(svc, "_alert_state",
                        lambda: alerts if alerts is not None
                        else {"total": 0, "armed": 0, "triggered": 0})
    monkeypatch.setattr(svc, "_data_layer_state",
                        lambda: data if data is not None else dict(_WORKING))
    if key:
        monkeypatch.setenv("MARKETAUX_API_TOKEN", "x")
    else:
        monkeypatch.delenv("MARKETAUX_API_TOKEN", raising=False)


def _steps(out):
    return [s["step"] for s in out["next_steps"]]


# ── get_started: the steps follow the state ───────────────────────────────────

def test_a_fresh_install_is_told_to_make_a_wallet_and_an_alert(monkeypatch):
    _patch(monkeypatch)
    steps = " | ".join(_steps(svc.get_started()))
    assert "practice wallet" in steps
    assert "standing alert" in steps


def test_an_existing_wallet_is_not_offered_again(monkeypatch):
    _patch(monkeypatch, wallet={"exists": True, "currency": "USDT",
                                "starting_balance": 5000.0, "open": 1,
                                "closed": 0})
    steps = _steps(svc.get_started())
    assert not any("Create a practice wallet" == s for s in steps)
    assert any("Take a practice trade" in s for s in steps)


def test_closed_trades_turn_the_step_into_a_review(monkeypatch):
    _patch(monkeypatch, wallet={"exists": True, "currency": "USDT",
                                "starting_balance": 5000.0, "open": 0,
                                "closed": 3})
    step = next(s for s in svc.get_started()["next_steps"]
                if "Review" in s["step"])
    assert "3 closed trade" in step["why"]


def test_armed_alerts_turn_the_step_into_scheduling_them(monkeypatch):
    _patch(monkeypatch, alerts={"total": 2, "armed": 2, "triggered": 0})
    step = next(s for s in svc.get_started()["next_steps"]
                if "schedule" in s["step"])
    assert "nothing fires" in step["why"]
    assert "2 alert(s) are armed" in step["why"]


def test_a_set_marketaux_key_is_not_asked_for(monkeypatch):
    _patch(monkeypatch, key=True)
    assert not any("Marketaux" in s or "MARKETAUX" in s
                   for s in _steps(svc.get_started()))
    assert svc.get_started()["state"]["marketaux_key_set"] is True


def test_a_missing_key_is_optional_and_names_the_free_alternative(monkeypatch):
    _patch(monkeypatch)
    step = next(s for s in svc.get_started()["next_steps"]
                if "Optional" in s["step"])
    assert "fear_greed" in step["why"]


# ── get_started: a broken data layer is said first ────────────────────────────

def test_a_dead_data_layer_is_the_first_thing_said(monkeypatch):
    _patch(monkeypatch, data={"working": False, "reason": "throttled"})
    first = svc.get_started()["next_steps"][0]
    assert "not answering" in first["step"]
    assert "throttled" in first["why"]


def test_a_working_data_layer_is_not_a_step(monkeypatch):
    _patch(monkeypatch)
    assert not any("not answering" in s for s in _steps(svc.get_started()))


def test_the_measured_state_is_returned_not_just_the_advice(monkeypatch):
    _patch(monkeypatch)
    state = svc.get_started()["state"]
    assert set(state) == {"paper_wallet", "alerts", "market_data",
                          "marketaux_key_set"}
    assert state["market_data"]["btc_price"] == 73000.0


def test_every_step_says_what_to_do_about_it(monkeypatch):
    """A step without a "do" is an observation, not a next step."""
    _patch(monkeypatch, data={"working": False, "reason": "throttled"})
    for step in svc.get_started()["next_steps"]:
        assert step["why"].strip(), step
        assert step["do"].strip(), step


def test_it_does_not_promise_real_trading(monkeypatch):
    _patch(monkeypatch)
    out = svc.get_started()
    blob = (out["note"] + " ".join(s["why"] for s in out["next_steps"])).lower()
    assert "nothing" in blob and "order" in blob


# ── alert_schedule_recipe ─────────────────────────────────────────────────────

def test_the_recipe_gives_a_cron_and_a_prompt(monkeypatch):
    _patch(monkeypatch, alerts={"total": 1, "armed": 1, "triggered": 0})
    out = svc.alert_schedule_recipe(15)
    assert out["schedule"]["cron"] == "*/15 * * * *"
    assert "alert_check" in out["task_prompt"]
    assert out["armed_now"] == 1


def test_hourly_and_longer_cadences_use_the_hour_field(monkeypatch):
    _patch(monkeypatch)
    assert svc.alert_schedule_recipe(60)["schedule"]["cron"] == "0 */1 * * *"
    assert svc.alert_schedule_recipe(240)["schedule"]["cron"] == "0 */4 * * *"


@pytest.mark.parametrize("bad", [0, 4, 1441, -15])
def test_an_impossible_cadence_is_refused(monkeypatch, bad):
    _patch(monkeypatch)
    out = svc.alert_schedule_recipe(bad)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert str(bad) in out["error"]["message"]


def test_the_prompt_preserves_the_distinction_alerts_exist_to_keep(monkeypatch):
    """"Not evaluated" is not "not met" — a schedule must not blur that."""
    _patch(monkeypatch)
    prompt = svc.alert_schedule_recipe()["task_prompt"]
    assert "not_evaluated" in prompt
    assert "NOT checked" in prompt


def test_the_caveats_name_what_a_schedule_cannot_do(monkeypatch):
    _patch(monkeypatch)
    caveats = " ".join(svc.alert_schedule_recipe()["caveats"]).lower()
    assert "at check time" in caveats, "must admit it samples, not watches"
    assert "one-shot" in caveats
    assert "local" in caveats, "must say the DB is on this machine"


def test_the_recipe_does_not_claim_the_server_can_push(monkeypatch):
    _patch(monkeypatch)
    out = svc.alert_schedule_recipe()
    blob = (out["why"] + " ".join(out["how"])).lower()
    assert "nothing fires them" in blob
    for overclaim in ("notify you", "we will alert", "push notification"):
        assert overclaim not in blob, overclaim
