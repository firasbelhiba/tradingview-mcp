"""The parts of this server a person meets directly. Offline.

Four defects found by running every tool against live data and measuring the
result rather than reading the code. Each test below pins one of them shut:

  1. Not one of 255 parameters carried a description in inputSchema, so a
     client rendering a form or a confirmation dialog showed bare names —
     `notional`, `fee_pct` — and `fee_pct = 0.05` reads as 5% to anyone who
     has not seen the docstring.
  2. The connection instructions name five tool families worth drawing as a
     panel; only three service modules emitted a `display_hint`, so most of
     the tools the policy pointed at arrived with no pointer.
  3. `lead_traders` returned rows whose measurement had failed — null
     leverage, null worst trade — without saying which. A blank leverage
     beside measured ones reads as a low one, inverting the tool's purpose.
  4. `squeeze_scan` sent every elevated row twice: 5,753 of 21,710 live
     characters were rows already present in `ranked`, byte for byte.
"""
import asyncio
import json

import pytest

from tradingview_mcp.core.services import display_hints
from tradingview_mcp.server import mcp

# Every tool a person drives by hand — picks a balance for, sets a stop on,
# names a threshold for. A screener nobody types arguments to is not here.
_HAND_DRIVEN = (
    "crypto_trade_plan", "paper_wallet_reset", "paper_trade_open",
    "paper_trade_close", "paper_trade_modify", "paper_wallet_status",
    "paper_trade_history", "alert_create", "alert_check", "alert_list",
    "alert_manage",
)

# Named in the server's PRESENTATION block as worth rendering.
_PANEL_SURFACES = (
    display_hints.BREADTH, display_hints.RELATIVE_STRENGTH,
    display_hints.SQUEEZE, display_hints.FUNDING, display_hints.LEAD_TRADERS,
)


@pytest.fixture(scope="module")
def tools():
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


# ── 1. Parameters that explain themselves where a form can read them ──────────

def test_every_hand_driven_parameter_has_a_schema_description(tools):
    missing = []
    for name in _HAND_DRIVEN:
        for arg, spec in (tools[name].inputSchema.get("properties") or {}).items():
            if not (spec.get("description") or "").strip():
                missing.append(f"{name}.{arg}")
    assert not missing, f"no description in the schema for: {missing}"


def test_the_docstring_is_not_the_only_place_the_units_live(tools):
    """A docstring reaches the model. A form reaches the person."""
    fee = tools["paper_wallet_reset"].inputSchema["properties"]["fee_pct"]
    assert "0.05%" in fee["description"], "must disambiguate percent from fraction"
    assert "not 5%" in fee["description"]


def test_money_arguments_say_what_currency_and_what_the_number_means(tools):
    notional = tools["paper_trade_open"].inputSchema["properties"]["notional"]
    assert "wallet's currency" in notional["description"]
    assert "leverage" in notional["description"], "margin vs notional must be said"


def test_required_arguments_survived_the_rewrite(tools):
    """Annotated[...] with no default must stay required, or a missing
    balance would silently become one the server chose."""
    assert tools["paper_wallet_reset"].inputSchema["required"] == ["balance"]
    assert set(tools["paper_trade_open"].inputSchema["required"]) == {
        "symbol", "side", "notional"
    }
    assert set(tools["alert_create"].inputSchema["required"]) == {
        "symbol", "metric", "operator", "threshold"
    }


def test_defaults_survived_the_rewrite(tools):
    props = tools["paper_wallet_reset"].inputSchema["properties"]
    assert props["fee_pct"]["default"] == 0.05
    assert props["max_leverage"]["default"] == 20.0
    assert props["confirm_wipe"]["default"] is False
    assert tools["paper_trade_open"].inputSchema["properties"]["market"]["default"] \
        == "futures"


# ── 2. The rendering pointer, on the tools the instructions point at ──────────

@pytest.mark.parametrize("hint", _PANEL_SURFACES)
def test_every_panel_hint_points_at_the_full_policy(hint):
    assert "draw it as a panel" in hint
    assert "PRESENTATION" in hint, "the policy travels once; this is a pointer"


@pytest.mark.parametrize("hint", _PANEL_SURFACES)
def test_every_panel_hint_stays_a_pointer(hint):
    """It rides on every call. The first wallet hint ran 890 bytes — 52% of
    an empty wallet's payload — which is how that limit got written down."""
    assert len(json.dumps(hint)) < 320, f"{len(hint)} chars; keep it a pointer"


@pytest.mark.parametrize("hint", _PANEL_SURFACES)
def test_every_panel_hint_names_what_a_tidy_panel_would_drop(hint):
    """The shared prefix cannot say this; it is the whole point of the tail."""
    tail = hint.split("instructions); ", 1)[1]
    assert len(tail) > 30, "a hint with no surface-specific warning is decoration"


def test_the_hints_differ_from_each_other():
    assert len(set(_PANEL_SURFACES)) == len(_PANEL_SURFACES)


def test_the_hints_stay_about_presentation_only():
    """The server may say how to SHOW a reading, never what to conclude."""
    blob = " ".join(_PANEL_SURFACES).lower()
    for forbidden in ("recommend", "advise", "you should", "good entry",
                      "profitable", "best trade"):
        assert forbidden not in blob, f"drifted into advice: {forbidden}"


def test_each_hint_warns_about_its_own_trap():
    assert "denominator" in display_hints.BREADTH
    assert "not a low-risk symbol" in display_hints.SQUEEZE
    assert "per symbol" in display_hints.FUNDING
    assert "missing, not low" in display_hints.LEAD_TRADERS
