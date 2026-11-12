"""The practice wallet as a panel the host draws, not one the assistant writes.

Until now every wallet panel was written from scratch by the assistant from
a ``display_hint``: slow, different each time, and its buttons could only
send a chat message, which the host flags with a security warning. A view
served by the server is the same every time and its buttons call tools
directly.

Two audiences read one tool result, and they are given different things:

  * ``structuredContent`` goes to the view and is not added to the model's
    context, so it carries the whole wallet status.
  * ``content`` is what the model reads. When the client says it can draw
    views it gets a short summary and is told a panel is on screen, so it
    does not repeat the numbers or draw a second panel. When the client did
    not say so, it gets the full status with its display_hint, exactly as
    paper_wallet_status returns it - because then nothing was drawn.

The page loads one script: the MCP Apps bridge, pinned to the version the
upstream project found current hosts to speak. Everything else is inline,
and every value from the wallet is written with textContent, never as HTML.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from tradingview_mcp import apps
from tradingview_mcp.core.services.paper_store import DEFAULT_WALLET

WALLET_URI = "ui://tradingview-mcp/wallet.html"
BRIDGE_URL = "https://unpkg.com/@modelcontextprotocol/ext-apps@1.0.1/app-with-deps"

# The host frames the view with a content-security policy built from this.
# One origin, for the bridge script; no network access is asked for, so the
# view can reach nothing but its host.
RESOURCE_META = {
    "ui": {
        "csp": {"resourceDomains": ["https://unpkg.com"]},
        "prefersBorder": True,
    }
}

_PANEL_NOTE = (
    "An interactive panel showing this wallet is on the person's screen, with "
    "its own refresh button. Do not repeat these figures in prose and do not "
    "draw another panel. Add only what the panel cannot say."
)


def wallet_label(wallet: Optional[str]) -> str:
    """The name a wallet argument refers to, as the wallet list spells it."""
    return (wallet or "").strip().lower() or DEFAULT_WALLET


def summary_for_model(status: dict, wallet_name: str) -> dict:
    """The few figures the model needs when the panel is doing the showing."""
    wallet = status.get("wallet") or {}
    summary: dict = {
        "wallet": wallet_name,
        "currency": wallet.get("currency"),
        "equity": wallet.get("equity"),
        "cash": wallet.get("cash"),
        "unrealised_pnl": wallet.get("unrealised_pnl"),
        "total_return_pct": wallet.get("total_return_pct"),
        "open_positions": [
            {
                "position_id": p.get("position_id"),
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "leverage": p.get("leverage"),
                "entry_price": p.get("entry_price"),
                "current_price": p.get("current_price"),
                "unrealised_pnl": p.get("unrealised_pnl"),
                "stop_loss": p.get("stop_loss"),
                "take_profit": p.get("take_profit"),
            }
            for p in status.get("open_positions") or []
        ],
        "pending_orders": [
            {
                "order_id": o.get("order_id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "limit_price": o.get("limit_price"),
                "notional": o.get("notional"),
            }
            for o in status.get("pending_orders") or []
        ],
        "as_of": status.get("as_of"),
        "simulation": "Simulated wallet. Nothing here reaches an exchange.",
        "shown": _PANEL_NOTE,
    }
    # Whatever a glance at healthy numbers could miss is never trimmed away.
    for key in ("unsettled_positions", "orders_unchecked", "settled_this_check",
                "orders_resolved_this_check"):
        if status.get(key):
            summary[key] = status[key]
    return summary


def text_for_model(status: dict, wallet_name: str, client_draws: Optional[bool]) -> str:
    if "error" in status:
        return json.dumps(status, default=str)
    if client_draws:
        return json.dumps(summary_for_model(status, wallet_name), default=str)
    return json.dumps(status, default=str)


def register(mcp: Any) -> bool:
    """Serve the view and advertise the extension. False if the SDK cannot."""
    if not apps.supports_meta():
        return False

    @mcp.resource(
        WALLET_URI,
        name="paper_wallet_panel",
        title="Paper wallet panel",
        description="The view paper_wallet_widget is drawn with.",
        mime_type=apps.APP_MIME,
        meta=RESOURCE_META,
    )
    def paper_wallet_panel() -> str:
        return WALLET_HTML

    apps.advertise(mcp)
    return True


WALLET_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<title>Paper wallet</title>
<style>
  :root {
    color-scheme: light dark;
    --fg: var(--color-text-primary, light-dark(#1c1c1c, #ededed));
    --muted: var(--color-text-secondary, light-dark(#6a6a6a, #a0a0a0));
    --line: var(--color-border-primary, light-dark(rgba(0,0,0,.13), rgba(255,255,255,.15)));
    --card: var(--color-background-secondary, light-dark(rgba(0,0,0,.035), rgba(255,255,255,.055)));
    --up: light-dark(#14805a, #3ecf8e);
    --down: light-dark(#c2392f, #ff6b61);
    --warn: light-dark(#8a5a00, #f0b429);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; }
  body { font: 13px/1.45 var(--font-sans, system-ui, -apple-system, "Segoe UI", sans-serif);
         color: var(--fg); padding: 14px 16px 12px; }
  .num { font-variant-numeric: tabular-nums; }
  .up { color: var(--up); } .down { color: var(--down); } .muted { color: var(--muted); }
  header { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  h1 { font-size: 13px; font-weight: 600; margin: 0; }
  .tag { font-size: 10.5px; letter-spacing: .04em; text-transform: uppercase;
         border: 1px solid var(--line); border-radius: 999px; padding: 1px 7px; color: var(--muted); }
  #equity { font-size: 26px; font-weight: 650; margin: 6px 0 0; }
  #ret { font-size: 13px; margin-left: 8px; font-weight: 500; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
           gap: 8px; margin: 12px 0 4px; }
  .stat { background: var(--card); border-radius: 8px; padding: 7px 10px; }
  .stat b { display: block; font-weight: 600; font-size: 14px; }
  .stat span { font-size: 11px; color: var(--muted); }
  h2 { font-size: 11px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
       color: var(--muted); margin: 14px 0 6px; }
  .row { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin-bottom: 6px;
         display: grid; grid-template-columns: 1fr auto; gap: 2px 12px; }
  .row .sym { font-weight: 600; }
  .pill { font-size: 10.5px; font-weight: 600; text-transform: uppercase; border-radius: 4px;
          padding: 1px 6px; margin-left: 6px; }
  .pill.long { color: var(--up); background: color-mix(in srgb, var(--up) 14%, transparent); }
  .pill.short { color: var(--down); background: color-mix(in srgb, var(--down) 14%, transparent); }
  .row .pnl { font-weight: 600; text-align: right; }
  .row .sub { grid-column: 1 / -1; font-size: 11.5px; color: var(--muted); }
  .notice { border-left: 3px solid var(--warn); background: var(--card); padding: 6px 10px;
            border-radius: 0 6px 6px 0; margin: 10px 0 0; font-size: 12px; }
  .notice b { color: var(--warn); }
  .empty { color: var(--muted); padding: 4px 0 2px; }
  footer { display: flex; align-items: center; justify-content: space-between; gap: 10px;
           margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--line); }
  #asof { font-size: 11.5px; color: var(--muted); }
  button { font: inherit; font-size: 12px; color: var(--fg); background: transparent;
           border: 1px solid var(--line); border-radius: 6px; padding: 4px 12px; cursor: pointer; }
  button:hover:not(:disabled) { background: var(--card); }
  button:focus-visible { outline: 2px solid var(--fg); outline-offset: 2px; }
  button:disabled { opacity: .55; cursor: default; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<div id="wait" class="muted" role="status">Loading the wallet…</div>

<main id="panel" hidden>
  <header>
    <h1 id="name">Practice wallet</h1>
    <span class="tag">Simulated</span>
  </header>
  <p id="equity" class="num"><span id="equity-value"></span><span id="ret"></span></p>
  <div class="stats" id="stats"></div>
  <div id="notices"></div>
  <h2 id="pos-title">Open positions</h2>
  <div id="positions"></div>
  <div id="orders-wrap" hidden>
    <h2>Resting orders</h2>
    <div id="orders"></div>
  </div>
  <footer>
    <span id="asof" role="status" aria-live="polite"></span>
    <button id="refresh" type="button">Refresh</button>
  </footer>
</main>

<script>
  // If the bridge below never loads, say so instead of staying blank: a blank
  // frame and a blocked script look identical from the outside.
  window.__walletConnected = false;
  setTimeout(function () {
    if (window.__walletConnected) return;
    var wait = document.getElementById("wait");
    if (wait && !wait.hidden) {
      wait.textContent = "This panel could not connect to the app showing it. " +
        "The wallet figures are in the message above.";
    }
  }, 6000);
</script>

<script type="module">
import { App } from "https://unpkg.com/@modelcontextprotocol/ext-apps@1.0.1/app-with-deps";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};
const isNum = (v) => typeof v === "number" && isFinite(v);
const money = (v) => isNum(v)
  ? v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—";
const signed = (v) => isNum(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + money(Math.abs(v)) : "—";
const price = (v) => {
  if (!isNum(v)) return "—";
  const a = Math.abs(v);
  const digits = a >= 1000 ? 1 : a >= 1 ? 2 : a >= 0.01 ? 4 : 8;
  return v.toLocaleString("en-US", { minimumFractionDigits: digits > 2 ? 2 : digits,
                                     maximumFractionDigits: digits });
};
const tone = (v) => !isNum(v) || v === 0 ? "" : v > 0 ? "up" : "down";
const lev = (v) => isNum(v) && v !== 1 ? " · " + (+v.toFixed(2)) + "x" : "";

let walletArg = null;        // which wallet a refresh should ask for
let walletName = null;       // set only for a named wallet; the default needs no label

function unwrap(result) {
  // The bridge delivers the tool result directly or wrapped in {toolResult}.
  const res = (result && result.toolResult) || result || {};
  let data = res.structuredContent;
  if (!data && Array.isArray(res.content)) {
    const text = res.content.find((c) => c.type === "text");
    if (text) { try { data = JSON.parse(text.text); } catch (e) { /* not JSON */ } }
  }
  return data;
}

function stat(label, value, cls) {
  const box = el("div", "stat");
  box.append(el("b", "num " + (cls || ""), value), el("span", "", label));
  return box;
}

function notice(strong, rest) {
  const box = el("div", "notice");
  box.append(el("b", "", strong), document.createTextNode(" " + rest));
  return box;
}

function describe(items) {
  // Best effort: these lists carry whatever the settlement step recorded.
  if (!Array.isArray(items)) return "";
  return items.map((i) => {
    if (i === null || typeof i !== "object") return String(i);
    const what = i.exit_reason || i.reason || i.status || i.outcome || "";
    return [i.symbol, what].filter(Boolean).join(" ");
  }).filter(Boolean).join(", ");
}

function render(data) {
  if (!data) return;
  $("wait").hidden = true;
  $("panel").hidden = false;

  if (data.error) {
    $("name").textContent = "Practice wallet";
    $("equity-value").textContent = "";
    $("ret").textContent = "";
    $("stats").replaceChildren();
    $("positions").replaceChildren(el("div", "empty",
      (data.error && data.error.message) || String(data.error)));
    $("notices").replaceChildren();
    $("orders-wrap").hidden = true;
    $("pos-title").hidden = true;
    return;
  }
  $("pos-title").hidden = false;

  if (data.wallet_name) walletName = data.wallet_is_default ? null : data.wallet_name;
  const w = data.wallet || {};
  const cur = w.currency || "";
  $("name").textContent = walletName
    ? "Practice wallet · " + walletName : "Practice wallet";
  $("equity-value").textContent = money(w.equity) + " " + cur;
  const ret = $("ret");
  ret.textContent = isNum(w.total_return_pct)
    ? (w.total_return_pct > 0 ? "+" : "") + w.total_return_pct.toFixed(2) + "%" : "";
  ret.className = tone(w.total_return_pct);

  const stats = [
    stat("Cash", money(w.cash)),
    stat("In positions", money(w.margin_posted)),
  ];
  // Money waiting in a resting order is neither cash nor a position, and
  // without its own box the three would not add up to the total above.
  if (isNum(w.margin_on_resting_orders) && w.margin_on_resting_orders > 0) {
    stats.push(stat("Held for orders", money(w.margin_on_resting_orders)));
  }
  stats.push(
    stat("Open profit / loss", signed(w.unrealised_pnl), tone(w.unrealised_pnl)),
    stat("Closed profit / loss", signed(w.realised_pnl), tone(w.realised_pnl)),
  );
  $("stats").replaceChildren(...stats);

  const notices = [];
  if (data.unsettled_positions) {
    notices.push(notice("Could not be priced:",
      (describe(data.unsettled_positions) || "some positions") +
      ". Their figures here may be out of date."));
  }
  if (data.orders_unchecked) {
    notices.push(notice("Orders not checked:",
      (describe(data.orders_unchecked) || "some resting orders") +
      ". They may have filled without it showing yet."));
  }
  if (data.settled_this_check) {
    notices.push(notice("Closed since last look:", describe(data.settled_this_check) || "see history"));
  }
  if (data.orders_resolved_this_check) {
    notices.push(notice("Orders resolved since last look:",
      describe(data.orders_resolved_this_check) || "see orders"));
  }
  $("notices").replaceChildren(...notices);

  const positions = data.open_positions || [];
  $("pos-title").textContent = "Open positions (" + positions.length + ")";
  if (!positions.length) {
    $("positions").replaceChildren(el("div", "empty", "Nothing open."));
  } else {
    $("positions").replaceChildren(...positions.map((p) => {
      const row = el("div", "row");
      const head = el("div");
      head.append(el("span", "sym", p.symbol),
                  el("span", "pill " + (p.side === "short" ? "short" : "long"), p.side),
                  el("span", "muted", lev(p.leverage)));
      const pnl = el("div", "pnl num " + tone(p.unrealised_pnl), signed(p.unrealised_pnl));
      const bits = [
        "Entry " + price(p.entry_price) + " → now " + price(p.current_price),
        "Size " + money(p.notional),
        "Stop " + (isNum(p.stop_loss) ? price(p.stop_loss) : "none"),
        "Target " + (isNum(p.take_profit) ? price(p.take_profit) : "none"),
      ];
      if (isNum(p.liquidation_price)) bits.push("Liquidation " + price(p.liquidation_price));
      row.append(head, pnl, el("div", "sub num", bits.join("  ·  ")));
      return row;
    }));
  }

  const orders = data.pending_orders || [];
  $("orders-wrap").hidden = !orders.length;
  $("orders").replaceChildren(...orders.map((o) => {
    const row = el("div", "row");
    const head = el("div");
    head.append(el("span", "sym", o.symbol),
                el("span", "pill " + (o.side === "short" ? "short" : "long"), o.side),
                el("span", "muted", lev(o.leverage)));
    row.append(head, el("div", "pnl num muted", "at " + price(o.limit_price)),
               el("div", "sub num", "Size " + money(o.notional) + "  ·  waiting for price"));
    return row;
  }));

  const at = data.as_of ? new Date(data.as_of) : null;
  $("asof").textContent = at && !isNaN(at)
    ? "Prices as of " + at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "";
}

function applyHost(context) {
  if (!context) return;
  if (context.theme === "light" || context.theme === "dark") {
    document.documentElement.style.colorScheme = context.theme;
  }
  const vars = context.styles && context.styles.variables;
  if (vars && typeof vars === "object") {
    for (const [name, value] of Object.entries(vars)) {
      if (typeof name === "string" && name.startsWith("--") && typeof value === "string") {
        document.documentElement.style.setProperty(name, value);
      }
    }
  }
}

const app = new App({ name: "paper-wallet", version: "0.1.0" });

app.ontoolinput = (params) => {
  const args = (params && params.arguments) || {};
  if (typeof args.wallet === "string" && args.wallet) walletArg = args.wallet;
};
app.ontoolresult = (result) => render(unwrap(result));
app.onhostcontextchanged = (context) => applyHost(context);

$("refresh").addEventListener("click", async () => {
  const button = $("refresh");
  button.disabled = true;
  button.textContent = "Refreshing…";
  try {
    const result = await app.callServerTool({
      name: "paper_wallet_status",
      arguments: walletArg ? { wallet: walletArg } : {},
    });
    render(unwrap(result));
  } catch (error) {
    $("asof").textContent = "Could not refresh. Try again in a moment.";
  } finally {
    button.disabled = false;
    button.textContent = "Refresh";
  }
});

await app.connect();
window.__walletConnected = true;
applyHost(app.getHostContext && app.getHostContext());
</script>
</body>
</html>
"""
