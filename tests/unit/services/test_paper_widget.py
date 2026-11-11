"""Unit tests for the MCP-UI wallet widget.

The HTML renders in a sandboxed iframe on someone else's page, so the things
worth pinning are: it is self-contained, it escapes what it interpolates, it
survives missing fields, and no information lives only in a button.
"""
from tradingview_mcp.core.services import paper_widget as w


def _status(**over):
    base = {
        "wallet": {
            "currency": "USDT", "starting_balance": 10000.0, "cash": 9666.17,
            "margin_posted": 333.33, "unrealised_pnl": 0.8, "equity": 10000.3,
            "total_return_pct": 0.0, "realised_pnl": 0.0,
        },
        "open_positions": [{
            "position_id": 1, "symbol": "BTCUSDT", "market": "futures",
            "side": "short", "leverage": 3.0, "entry_price": 73809.16,
            "current_price": 73749.91, "stop_loss": 76500.0,
            "take_profit": 70000.0, "liquidation_price": 98412.21,
            "margin": 333.33, "unrealised_pnl": 0.8,
            "unrealised_pct_of_margin": 0.24,
        }],
        "open_count": 1, "closed_count": 0,
    }
    base.update(over)
    return base


def test_page_is_self_contained():
    """It cannot borrow the host's styling — an iframe has no access to it."""
    html = w.wallet_widget_html(_status())
    assert html.startswith("<!doctype html>")
    assert "<style>" in html and "</style>" in html
    assert "prefers-color-scheme" in html, "dark mode must be handled internally"
    assert "var(--surface-1)" not in html, "host CSS variables do not resolve here"
    assert "http://" not in html and "https://" not in html, "no external assets"


def test_numbers_and_labels_appear():
    html = w.wallet_widget_html(_status())
    for fragment in ("10,000.30", "9,666.17", "BTCUSDT", "short 3x",
                     "76,500.00", "98,412.21", "USDT"):
        assert fragment in html, fragment


def test_gain_and_loss_use_different_tones():
    up = w.wallet_widget_html(_status())
    down_status = _status()
    down_status["wallet"]["unrealised_pnl"] = -42.0
    down_status["open_positions"][0]["unrealised_pnl"] = -42.0
    down = w.wallet_widget_html(down_status)
    assert "var(--up)" in up
    assert "var(--down)" in down
    assert "+0.80" in up and "-42.00" in down


def test_missing_levels_render_as_a_dash():
    status = _status()
    status["open_positions"][0]["stop_loss"] = None
    status["open_positions"][0]["take_profit"] = None
    html = w.wallet_widget_html(status)
    assert "—" in html


def test_no_positions_says_so_instead_of_rendering_nothing():
    html = w.wallet_widget_html(_status(open_positions=[], open_count=0))
    assert "No open positions" in html


def test_error_state_offers_the_fix():
    html = w.wallet_widget_html(
        {"error": {"code": "NO_DATA", "message": "No paper wallet exists yet"}}
    )
    assert "No paper wallet exists yet" in html
    assert "Create a wallet" in html


def test_unsettled_positions_are_surfaced_not_hidden():
    """An unchecked position is the one thing a glance must not miss."""
    html = w.wallet_widget_html(_status(unsettled_positions=[
        {"position_id": 1, "unsettled": "klines unavailable"}
    ]))
    assert "could not be checked" in html
    assert "may already have hit" in html


def test_interpolated_text_is_escaped():
    """Symbols and notes reach this from stored rows, so they are escaped."""
    status = _status()
    status["open_positions"][0]["symbol"] = '<img src=x onerror="alert(1)">'
    html = w.wallet_widget_html(status)
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_actions_are_present_but_carry_no_unique_information():
    html = w.wallet_widget_html(_status())
    assert "postMessage" in html
    assert "paper_trade_close" in html
    # Every number a button would act on is also rendered as text, so a host
    # that ignores the action protocol loses nothing but convenience.
    assert "Close" in html and "73,809.16" in html


def test_simulation_notice_is_always_present():
    html = w.wallet_widget_html(_status())
    assert "Simulated wallet" in html
    assert "no real money" in html
    assert "slippage" in html


def test_resource_uri_is_a_ui_scheme():
    assert w.ui_resource_uri().startswith("ui://")
