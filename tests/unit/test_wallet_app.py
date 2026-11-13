"""The wallet panel's wiring. Offline.

The first widget was built, shipped, and never drew: it embedded an HTML
resource in a tool result, which is not what a host looks for. A host looks
for three things on the wire, so these tests look for the same three rather
than for the HTML being present somewhere.
"""
import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from mcp import types
from pydantic import AnyUrl

from tradingview_mcp import sdk
from tradingview_mcp import apps, server, wallet_app
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc

HTML = wallet_app.WALLET_HTML


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_PAPER_DB", raising=False)
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(tmp_path))
    store._initialised.clear()
    store.select_wallet(None)
    yield
    store.select_wallet(None)
    store._initialised.clear()


def _wallet(monkeypatch):
    monkeypatch.setattr(svc.px, "mark_price", lambda s, m: (100.0, "test"))
    monkeypatch.setattr(svc.px, "bars_since", lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(svc.px, "funding_cost", lambda *a, **k: (0.0, 0))
    svc.reset_wallet(10000.0)
    svc.open_trade("BTCUSDT", "short", 2000.0, "futures", 2, None, None, "t")


def _tool():
    return next(t for t in asyncio.run(server.mcp.list_tools())
                if t.name == "paper_wallet_widget")


# ── The three things a host looks for ─────────────────────────────────────────

def test_the_tool_points_at_its_view():
    meta = _tool().meta
    assert meta == {"ui": {"resourceUri": wallet_app.WALLET_URI}}
    assert wallet_app.WALLET_URI.startswith("ui://")


def test_the_view_is_listed_with_the_exact_mime_type():
    """A plain text/html resource is what the first attempt served, and what
    a host ignores."""
    listed = {str(r.uri): r for r in asyncio.run(server.mcp.list_resources())}
    entry = listed[wallet_app.WALLET_URI]
    assert sdk.field(entry, "mimeType") == "text/html;profile=mcp-app"
    assert entry.meta == wallet_app.RESOURCE_META


def test_reading_the_view_returns_the_page_and_its_allow_list():
    """Hosts read the csp from the content item first, the listing second."""
    item = list(asyncio.run(server.mcp.read_resource(wallet_app.WALLET_URI)))[0]
    assert item.mime_type == "text/html;profile=mcp-app"
    assert item.content == HTML
    assert item.meta["ui"]["csp"]["resourceDomains"] == ["https://unpkg.com"]


def test_the_server_advertises_the_extension():
    assert sdk.capabilities(server.mcp)["extensions"] == {apps.UI_EXTENSION: {}}


def test_advertising_twice_does_not_wrap_twice():
    before = sdk.lowlevel(server.mcp).get_capabilities
    apps.advertise(server.mcp)
    assert sdk.lowlevel(server.mcp).get_capabilities is before


# ── The page itself ───────────────────────────────────────────────────────────

def test_the_page_loads_one_script_and_the_allow_list_covers_it():
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", HTML))
    assert urls == {wallet_app.BRIDGE_URL}, urls
    origin = "/".join(wallet_app.BRIDGE_URL.split("/")[:3])
    assert origin in wallet_app.RESOURCE_META["ui"]["csp"]["resourceDomains"]


def test_the_bridge_version_is_pinned():
    """An unpinned import would change under every friend at once."""
    assert re.search(r"ext-apps@\d+\.\d+\.\d+/", wallet_app.BRIDGE_URL)


def test_no_wallet_value_is_ever_written_as_markup():
    """A note or a symbol is text someone typed. It goes in with
    textContent, so it can never become a tag."""
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
                 "eval("):
        assert sink not in HTML, sink


def test_every_element_the_script_reaches_for_exists():
    wanted = set(re.findall(r'\$\("([a-z-]+)"\)', HTML))
    wanted |= set(re.findall(r'getElementById\("([a-z-]+)"\)', HTML))
    present = set(re.findall(r'id="([a-z-]+)"', HTML))
    assert wanted and wanted <= present, wanted - present


def test_refresh_calls_a_tool_that_exists_with_an_argument_it_takes():
    called = re.search(r'callServerTool\(\{\s*name: "([a-z_]+)"', HTML).group(1)
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert called in tools
    assert "wallet" in sdk.field(tools[called], "inputSchema")["properties"]


def test_a_blocked_bridge_says_so_instead_of_staying_blank():
    """A blank frame and a blocked script look the same from outside, and
    which of the two happened is the whole point of trying this."""
    assert "could not connect" in HTML
    assert "__walletConnected" in HTML


def test_the_page_surfaces_what_a_glance_could_miss():
    for key in ("unsettled_positions", "orders_unchecked"):
        assert key in HTML, key


# ── What the model is told ────────────────────────────────────────────────────

def _ctx(extensions):
    caps = SimpleNamespace(extensions=extensions)
    return SimpleNamespace(session=SimpleNamespace(
        client_params=SimpleNamespace(capabilities=caps)))


def test_a_client_that_names_the_extension_can_draw():
    assert apps.client_supports_ui(_ctx({apps.UI_EXTENSION: {}})) is True


def test_a_client_that_names_other_extensions_cannot():
    assert apps.client_supports_ui(_ctx({"something/else": {}})) is False
    assert apps.client_supports_ui(_ctx(None)) is False


def test_outside_a_request_nothing_is_concluded():
    class NoSession:
        @property
        def session(self):
            raise ValueError("Context is not available outside of a request")

    assert apps.client_supports_ui(NoSession()) is None
    assert apps.client_supports_ui(None) is None


def test_when_a_panel_is_drawn_the_model_gets_a_summary_and_is_told(monkeypatch):
    _wallet(monkeypatch)
    status = svc.wallet_status()
    told = json.loads(wallet_app.text_for_model(status, "default", True))
    full = wallet_app.text_for_model(status, "default", False)
    assert "do not draw another panel" in told["shown"].lower()
    assert "display_hint" not in told, "a hint to draw would contradict the panel"
    assert told["open_positions"][0]["position_id"] == 1, "enough to act on"
    assert len(json.dumps(told)) < len(full) * 0.6


def test_when_nothing_is_drawn_the_model_gets_everything(monkeypatch):
    """Otherwise a client that cannot draw would be told a panel exists."""
    _wallet(monkeypatch)
    status = svc.wallet_status()
    for draws in (False, None):
        told = json.loads(wallet_app.text_for_model(status, "default", draws))
        assert told["display_hint"] == status["display_hint"]
        assert "shown" not in told


def test_a_failure_is_never_trimmed_from_the_summary(monkeypatch):
    _wallet(monkeypatch)
    status = svc.wallet_status()
    status["unsettled_positions"] = [{"symbol": "BTCUSDT", "reason": "no price"}]
    told = wallet_app.summary_for_model(status, "default")
    assert told["unsettled_positions"] == status["unsettled_positions"]


def test_an_error_reaches_the_model_whole():
    envelope = {"error": {"code": "NO_DATA", "message": "No paper wallet exists yet"}}
    assert json.loads(wallet_app.text_for_model(envelope, "default", True)) == envelope


# ── The tool ──────────────────────────────────────────────────────────────────

def test_the_tool_returns_data_for_the_view_and_text_for_the_model(monkeypatch):
    _wallet(monkeypatch)
    out = asyncio.run(server.mcp.call_tool("paper_wallet_widget", {}))
    assert isinstance(out, types.CallToolResult)
    assert sdk.field(out, "structuredContent")["wallet_name"] == "main", "as the wallet list spells it"
    assert sdk.field(out, "structuredContent")["wallet_is_default"] is True
    assert sdk.field(out, "structuredContent")["open_positions"][0]["symbol"] == "BTCUSDT"
    told = json.loads(out.content[0].text)
    assert "display_hint" in told, "no client said it could draw, so nothing was"


def test_a_named_wallet_is_read_and_named(monkeypatch):
    _wallet(monkeypatch)
    svc.reset_wallet(500.0, wallet="scalps")
    out = asyncio.run(server.mcp.call_tool("paper_wallet_widget", {"wallet": "scalps"}))
    assert sdk.field(out, "structuredContent")["wallet_name"] == "scalps"
    assert sdk.field(out, "structuredContent")["wallet_is_default"] is False
    assert sdk.field(out, "structuredContent")["wallet"]["equity"] == pytest.approx(500.0)
    assert sdk.field(out, "structuredContent")["open_positions"] == []


def test_no_wallet_is_an_error_the_view_can_show():
    out = asyncio.run(server.mcp.call_tool("paper_wallet_widget", {}))
    assert sdk.field(out, "structuredContent")["error"]["code"] == "NO_DATA"
    assert "wallet_name" not in sdk.field(out, "structuredContent")
    assert json.loads(out.content[0].text)["error"]["code"] == "NO_DATA"


def test_the_request_context_is_not_offered_as_an_argument():
    assert set(sdk.field(_tool(), "inputSchema")["properties"]) == {"include_closed", "wallet"}


def test_an_older_sdk_keeps_the_tool_and_draws_nothing(monkeypatch):
    """The dependency floor predates meta on tools. Starting must not fail."""
    monkeypatch.setattr(apps, "supports_meta", lambda: False)
    assert apps.tool_kwargs(wallet_app.WALLET_URI) == {}
    assert wallet_app.register(object()) is False


# ── Found by looking at the panel: a resting order read as a loss ─────────────

def test_money_held_for_a_resting_order_is_still_the_wallets(monkeypatch):
    """Drawing the wallet made this visible: with 300 waiting in a limit
    order the panel said -3.04% on a wallet that had lost 0.04% in fees.
    The margin had left `cash` and was counted nowhere."""
    from tradingview_mcp.core.services import paper_orders as orders

    _wallet(monkeypatch)
    for mod in (orders.px,):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (100.0, "test"))
        monkeypatch.setattr(mod, "bars_since",
                            lambda s, ms, m, now_ms=None: ([], "1m"))
    before = svc.wallet_status()["wallet"]["equity"]
    assert "placed" in orders.place_order("ETHUSDT", "long", 300.0, 90.0)
    after = svc.wallet_status()["wallet"]
    assert after["margin_on_resting_orders"] == pytest.approx(300.0)
    assert after["equity"] == pytest.approx(before), "placing an order costs nothing"
    assert after["cash"] + after["margin_posted"] + after["margin_on_resting_orders"] \
        + after["unrealised_pnl"] == pytest.approx(after["equity"])


def test_the_wallet_and_the_wallet_list_agree_on_what_it_is_worth(monkeypatch):
    """They were computed separately, and only one counted held margin."""
    from tradingview_mcp.core.services import paper_orders as orders

    _wallet(monkeypatch)
    monkeypatch.setattr(orders.px, "mark_price", lambda s, m: (100.0, "test"))
    monkeypatch.setattr(orders.px, "bars_since",
                        lambda s, ms, m, now_ms=None: ([], "1m"))
    orders.place_order("ETHUSDT", "long", 300.0, 90.0)
    status = svc.wallet_status()["wallet"]
    listed = next(w for w in svc.wallet_names()["wallets"]
                  if w["wallet"] == store.DEFAULT_WALLET)
    assert listed["equity_excl_unrealised"] == pytest.approx(
        status["equity"] - status["unrealised_pnl"])


def test_the_panel_shows_held_money_so_the_boxes_add_up():
    assert "margin_on_resting_orders" in HTML
    assert "Held for orders" in HTML


# ── Making the trial conclusive ───────────────────────────────────────────────

def test_the_panel_tells_the_conversation_what_is_on_screen():
    """After a refresh the panel holds newer figures than the assistant was
    given. It also makes a first draw visible from the other side, which is
    the only way to know one happened without asking the person."""
    assert "updateModelContext" in HTML
    assert "refreshed by the person" in HTML
    assert "catch (error) { /* this host does not take context updates */ }" in HTML


def test_the_client_is_named_by_its_software_not_its_user():
    info = SimpleNamespace(name="some-app", version="1.2.3")
    ctx = SimpleNamespace(session=SimpleNamespace(
        client_params=SimpleNamespace(clientInfo=info)))
    assert apps.client_name(ctx) == "some-app/1.2.3"
    assert apps.client_name(None) is None


def test_what_a_client_claims_is_logged_only_when_auditing(monkeypatch, capsys):
    _wallet(monkeypatch)
    monkeypatch.delenv("TRADINGVIEW_MCP_AUDIT_LOG", raising=False)
    asyncio.run(server.mcp.call_tool("paper_wallet_widget", {}))
    assert "[apps]" not in capsys.readouterr().err
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_LOG", "1")
    asyncio.run(server.mcp.call_tool("paper_wallet_widget", {}))
    line = capsys.readouterr().err
    assert "[apps] tool=paper_wallet_widget" in line and "says_it_draws=" in line


def test_the_views_half_never_contradicts_the_text(monkeypatch):
    """Measured through a Claude connector: the assistant was handed
    structuredContent, not the text. It carried a display_hint asking for a
    panel while the text beside it said one was already drawn."""
    _wallet(monkeypatch)
    status = svc.wallet_status()
    drawn = wallet_app.data_for_view(status, "main", True)
    assert "display_hint" not in drawn
    assert drawn["shown"] == wallet_app.PANEL_NOTE
    assert drawn["open_positions"] == status["open_positions"], "the view still gets everything"
    for draws in (False, None):
        plain = wallet_app.data_for_view(status, "main", draws)
        assert plain["display_hint"] == status["display_hint"]
        assert "shown" not in plain
    assert "display_hint" in status, "and the status itself is not modified"
