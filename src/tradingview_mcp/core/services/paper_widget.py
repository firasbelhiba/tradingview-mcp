"""
Paper Widget — an MCP-UI resource for the simulated wallet.

An experiment with a clear hypothesis: a tool can return an
``EmbeddedResource`` whose URI is ``ui://`` and whose mimeType is
``text/html``, and a host that implements MCP Apps / MCP-UI renders it inline
instead of printing JSON. The upstream project's hosted version draws charts
in-conversation this way, so the pattern is known to work for this codebase —
but a local stdio server and a remote connector are not guaranteed to be
treated alike, which is why this ships alongside the JSON tool rather than
replacing it.

Two consequences shape the HTML:

  * **It cannot borrow the host's styling.** The page renders in a sandboxed
    iframe, so claude.ai's CSS variables are out of reach. Everything is
    inline, and light/dark comes from ``prefers-color-scheme`` rather than a
    host token.
  * **Buttons talk back over postMessage**, the MCP-UI action protocol. On a
    host that implements it the button calls the tool; on one that does not it
    does nothing, which is why no information lives only in a button.

Every tool that returns this also returns the plain-text summary, so a host
that ignores the resource still shows something useful.
"""
from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional

_UI_URI = "ui://tradingview-mcp/paper-wallet.html"


def _fmt(value: Any, digits: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _signed(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    return f"{'+' if number >= 0 else ''}{number:,.{digits}f}"


def _tone(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "var(--ink)"
    if number > 0:
        return "var(--up)"
    if number < 0:
        return "var(--down)"
    return "var(--ink)"


_STYLE = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--dim:#52514e;--line:rgba(11,11,11,.12);
--up:#0f6e56;--down:#a32d2d;--chip:#f1efe8}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232322;--ink:#f0efec;
--dim:#b8b7b0;--line:rgba(255,255,255,.14);--up:#5dcaa5;--down:#f09595;--chip:#2c2c2a}}
*{box-sizing:border-box}
body{margin:0;padding:14px;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px;margin-bottom:12px}
.tile{background:var(--chip);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:19px;font-weight:500;margin-top:2px}
.lbl{font-size:12px;color:var(--dim)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:8px}
.hd{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.sym{font-weight:500}
.tag{font-size:11px;padding:2px 8px;border-radius:6px;background:var(--chip);color:var(--dim)}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:3px 0}
td.k{color:var(--dim)}
td.v{text-align:right;font-variant-numeric:tabular-nums}
.acts{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
button{font:inherit;font-size:13px;padding:6px 11px;border-radius:7px;cursor:pointer;
background:transparent;color:var(--ink);border:1px solid var(--line)}
button:hover{background:var(--chip)}
.sim{font-size:11px;color:var(--dim);margin-top:10px;line-height:1.45}
.empty{color:var(--dim);font-size:13px;padding:6px 0}
"""

_SCRIPT = """
function act(tool, params){
  try{
    window.parent.postMessage({type:'tool',messageId:'tv-'+Date.now(),
      payload:{toolName:tool,params:params||{}}},'*');
  }catch(e){}
  try{
    window.parent.postMessage({type:'prompt',messageId:'tvp-'+Date.now(),
      payload:{prompt:'Run '+tool+' '+JSON.stringify(params||{})}},'*');
  }catch(e){}
}
"""


def wallet_widget_html(status: Dict[str, Any]) -> str:
    """Self-contained HTML for the wallet, for an MCP-UI host to render."""
    if "error" in status:
        body = (
            f'<div class="card"><div class="lbl">Paper wallet</div>'
            f'<div class="empty">{escape(str(status["error"].get("message", "unavailable")))}</div>'
            f'<div class="acts"><button onclick="act(\'paper_wallet_reset\',{{}})">'
            f"Create a wallet</button></div></div>"
        )
        return _page(body)

    w = status.get("wallet") or {}
    currency = escape(str(w.get("currency", "")))
    tiles = [
        ("Equity", _fmt(w.get("equity")), f"from {_fmt(w.get('starting_balance'))}"),
        ("Free cash", _fmt(w.get("cash")), f"{_fmt(w.get('margin_posted'))} as margin"),
        ("Unrealised", _signed(w.get("unrealised_pnl")),
         f"{_signed(w.get('total_return_pct'))}% total"),
        ("Realised", _signed(w.get("realised_pnl")),
         f"{status.get('closed_count', 0)} closed"),
    ]
    tile_html = "".join(
        f'<div class="tile"><span class="lbl">{escape(label)}</span>'
        f'<b style="color:{_tone(raw) if label in ("Unrealised", "Realised") else "var(--ink)"}">'
        f"{value}</b><span class=\"lbl\">{escape(sub)}</span></div>"
        for (label, value, sub), raw in zip(
            tiles,
            [None, None, w.get("unrealised_pnl"), w.get("realised_pnl")],
        )
    )

    positions: List[Dict[str, Any]] = status.get("open_positions") or []
    if positions:
        cards = "".join(_position_card(p, currency) for p in positions)
    else:
        cards = (
            '<div class="card"><div class="empty">No open positions. '
            "Ask for a trade to open one.</div></div>"
        )

    unsettled = status.get("unsettled_positions")
    warn = ""
    if unsettled:
        warn = (
            f'<div class="card" style="border-color:var(--down)">'
            f'<div class="lbl" style="color:var(--down)">'
            f"{len(unsettled)} position(s) could not be checked against their "
            f"levels — they stay open and may already have hit a stop or target."
            f"</div></div>"
        )

    actions = (
        '<div class="acts">'
        "<button onclick=\"act('paper_wallet_status',{'include_closed':5})\">Refresh</button>"
        "<button onclick=\"act('paper_trade_history',{})\">History</button>"
        "</div>"
    )
    sim = (
        '<div class="sim">Simulated wallet — no real order, no real money. '
        "Prices are live; fills land exactly on levels, so slippage is not "
        "modelled.</div>"
    )
    return _page(f'<div class="row">{tile_html}</div>{warn}{cards}{actions}{sim}')


def _position_card(p: Dict[str, Any], currency: str) -> str:
    side = escape(str(p.get("side", "")))
    pnl = p.get("unrealised_pnl")
    rows = [
        ("Entry", _fmt(p.get("entry_price"))),
        ("Mark", _fmt(p.get("current_price"))),
        ("Stop", _fmt(p.get("stop_loss"))),
        ("Target", _fmt(p.get("take_profit"))),
        ("Liquidation", _fmt(p.get("liquidation_price"))),
        ("Margin", _fmt(p.get("margin"))),
    ]
    body = "".join(
        f'<tr><td class="k">{escape(k)}</td><td class="v">{v}</td></tr>'
        for k, v in rows
    )
    pid = p.get("position_id")
    return (
        f'<div class="card"><div class="hd">'
        f'<span class="sym">{escape(str(p.get("symbol", "")))}</span>'
        f'<span class="tag">{side} {_fmt(p.get("leverage"), 0)}x</span>'
        f'<span class="tag">{escape(str(p.get("market", "")))}</span>'
        f'<span class="tag" style="margin-left:auto">#{escape(str(pid))}</span>'
        f"</div>"
        f'<div style="font-size:21px;font-weight:500;color:{_tone(pnl)};margin-bottom:6px">'
        f"{_signed(pnl)} <span class=\"lbl\">{currency} "
        f"({_signed(p.get('unrealised_pct_of_margin'))}% of margin)</span></div>"
        f"<table>{body}</table>"
        f'<div class="acts">'
        f"<button onclick=\"act('paper_trade_close',{{'position_id':{pid}}})\">Close</button>"
        f"</div></div>"
    )


def _page(body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<style>{_STYLE}</style></head><body>{body}"
        f"<script>{_SCRIPT}</script></body></html>"
    )


def ui_resource_uri() -> str:
    return _UI_URI
