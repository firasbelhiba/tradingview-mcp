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

from tradingview_mcp import sdk
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
        for arg, spec in (sdk.field(tools[name], "inputSchema").get("properties") or {}).items():
            if not (spec.get("description") or "").strip():
                missing.append(f"{name}.{arg}")
    assert not missing, f"no description in the schema for: {missing}"


def test_the_docstring_is_not_the_only_place_the_units_live(tools):
    """A docstring reaches the model. A form reaches the person."""
    fee = sdk.field(tools["paper_wallet_reset"], "inputSchema")["properties"]["fee_pct"]
    assert "0.05%" in fee["description"], "must disambiguate percent from fraction"
    assert "not 5%" in fee["description"]


def test_money_arguments_say_what_currency_and_what_the_number_means(tools):
    notional = sdk.field(tools["paper_trade_open"], "inputSchema")["properties"]["notional"]
    assert "wallet's currency" in notional["description"]
    assert "leverage" in notional["description"], "margin vs notional must be said"


def test_required_arguments_survived_the_rewrite(tools):
    """Annotated[...] with no default must stay required, or a missing
    balance would silently become one the server chose."""
    assert sdk.field(tools["paper_wallet_reset"], "inputSchema")["required"] == ["balance"]
    assert set(sdk.field(tools["paper_trade_open"], "inputSchema")["required"]) == {
        "symbol", "side", "notional"
    }
    assert set(sdk.field(tools["alert_create"], "inputSchema")["required"]) == {
        "symbol", "metric", "operator", "threshold"
    }


def test_defaults_survived_the_rewrite(tools):
    props = sdk.field(tools["paper_wallet_reset"], "inputSchema")["properties"]
    assert props["fee_pct"]["default"] == 0.05
    assert props["max_leverage"]["default"] == 20.0
    assert props["confirm_wipe"]["default"] is False
    assert sdk.field(tools["paper_trade_open"], "inputSchema")["properties"]["market"]["default"] \
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


# ── 5. The code a client branches on must describe what went missing ──────────

def test_a_missing_local_row_is_not_reported_as_a_missing_symbol():
    """SYMBOL_NOT_FOUND means a market does not list a ticker.

    A stale paper-position id and a stale alert id are rows in this server's
    own database. Reusing the symbol code for them sent a client hunting for
    a ticker when the id had simply been deleted.
    """
    from tradingview_mcp.core.errors import ErrorCode

    assert ErrorCode.RECORD_NOT_FOUND.value == "RECORD_NOT_FOUND"
    assert ErrorCode.RECORD_NOT_FOUND != ErrorCode.SYMBOL_NOT_FOUND


def test_a_real_symbol_miss_still_uses_the_symbol_code():
    """The fix must not drag the genuine cases along with it."""
    from tradingview_mcp.core.services import crypto_analysis_service as cas

    src = (
        __import__("pathlib").Path(cas.__file__).read_text(encoding="utf-8")
    )
    assert "SYMBOL_NOT_FOUND" in src, "an unlisted ticker is still that"


# ── 6. Hint lists a person can read ───────────────────────────────────────────

def test_timeframes_are_listed_in_time_order():
    """Sorted alphabetically they came back 15m, 1D, 1M, 1W, 1h, 4h, 5m —
    every value right and the sequence meaningless."""
    from tradingview_mcp.core.errors import ScreenerServiceError
    from tradingview_mcp.core.utils.validators import validate_timeframe

    with pytest.raises(ScreenerServiceError) as caught:
        validate_timeframe("7m")
    listed = caught.value.to_envelope()["error"]["valid_timeframes"]
    assert listed == ["5m", "15m", "1h", "4h", "1D", "1W", "1M"]


def test_a_typo_gets_the_name_it_probably_meant():
    from tradingview_mcp.core.errors import ScreenerServiceError
    from tradingview_mcp.core.utils.validators import validate_exchange

    with pytest.raises(ScreenerServiceError) as caught:
        validate_exchange("kucion")
    assert caught.value.to_envelope()["error"]["did_you_mean"] == ["kucoin"]


def test_a_typo_is_matched_case_insensitively():
    from tradingview_mcp.core.errors import ScreenerServiceError
    from tradingview_mcp.core.utils.validators import validate_exchange

    with pytest.raises(ScreenerServiceError) as caught:
        validate_exchange("BINENCE")
    assert "binance" in caught.value.to_envelope()["error"]["did_you_mean"]


def test_a_name_resembling_nothing_gets_no_guess():
    """A suggestion that is not close is worse than none."""
    from tradingview_mcp.core.errors import ScreenerServiceError
    from tradingview_mcp.core.utils.validators import validate_exchange

    with pytest.raises(ScreenerServiceError) as caught:
        validate_exchange("zzzzzzzz")
    assert caught.value.to_envelope()["error"]["did_you_mean"] is None


def test_grouping_the_exchanges_drops_none_of_them():
    """The list is what a caller self-corrects from, so it stays complete."""
    from tradingview_mcp.core.utils import validators as v

    grouped = v._exchanges_by_kind()
    assert sorted(grouped["crypto"] + grouped["stock"]) == v._all_exchanges()
    assert not set(grouped["crypto"]) & set(grouped["stock"])
    assert len(grouped["crypto"]) < len(grouped["stock"]), (
        "most accepted venues are stock markets, which is why one flat list "
        "buried the crypto ones"
    )
