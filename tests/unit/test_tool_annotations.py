"""Every MCP tool must carry directory-grade annotations.

Both official marketplaces (Anthropic's Connectors Directory and OpenAI's
ChatGPT plugin directory) require each tool to declare a human-readable
`title` and the applicable safety hint. Submissions with unannotated tools
are rejected at the Tools step of the portal, so this test makes the
requirement permanent: a newly added tool without annotations fails CI here
before it can fail a directory review.

Almost every tool in this server is read-only market analysis, so
readOnlyHint must be True across the board — that read-only posture is an
approval advantage for a finance connector: no financial-transaction risk.

The alert tools are the one exception and are named in _STATEFUL_TOOLS below.
They persist rows to a local SQLite file so a condition can outlive a single
request, which is what an alert is. They still send nothing, place no order
and touch no external service, so the finance-connector argument survives —
but their annotations have to say they write, and an allowlist is how that
stays deliberate: a NEW tool that quietly declares itself non-read-only still
fails here until it is added to the list on purpose.
"""
from tradingview_mcp.server import mcp

# Tools permitted to declare readOnlyHint=False, with the reason.
_STATEFUL_TOOLS = {
    "alert_create": "writes one alert row to the local SQLite store",
    "alert_manage": "deletes or re-arms a stored alert row",
    "paper_wallet_reset": "wipes and recreates the simulated wallet",
    "paper_trade_open": "writes a simulated position row",
    "paper_trade_close": "closes a simulated position row",
    "paper_trade_modify": "updates a simulated position's stop/target",
    "paper_order_place": "writes a resting order row and holds its margin",
    "paper_order_cancel": "takes a resting order off the book",
    "paper_copy_trader": "opens simulated positions mirroring a lead trader",
}


def _all_tools():
    tools = mcp._tool_manager.list_tools()
    assert len(tools) >= 30, f"tool count suspiciously low: {len(tools)}"
    return tools


def test_every_tool_has_annotations_with_title():
    missing = [t.name for t in _all_tools() if t.annotations is None or not (t.annotations.title or "").strip()]
    assert not missing, f"tools missing annotations/title (directory submissions reject these): {missing}"


def test_every_tool_is_declared_read_only():
    not_ro = [
        t.name for t in _all_tools()
        if t.annotations is None or t.annotations.readOnlyHint is not True
    ]
    unexpected = sorted(set(not_ro) - set(_STATEFUL_TOOLS))
    assert unexpected == [], (
        "tools not declared read-only and not on the stateful allowlist: "
        f"{unexpected}"
    )


def test_stateful_tools_actually_declare_that_they_write():
    """The allowlist is a permission, not an expectation.

    If a tool on it is later made read-only, the entry is stale and should go
    — otherwise the list quietly grows into a hole in the invariant.
    """
    by_name = {t.name: t for t in _all_tools()}
    for name in _STATEFUL_TOOLS:
        assert name in by_name, f"allowlisted tool {name} no longer exists"
        assert by_name[name].annotations.readOnlyHint is False, (
            f"{name} is on the stateful allowlist but declares itself read-only"
        )


def test_every_tool_explicitly_declares_destructiveness():
    # OpenAI's plugin scanner requires an EXPLICIT true/false for
    # destructiveHint on every tool — omitting it fails the MCP scan step.
    missing = [
        t.name for t in _all_tools()
        if t.annotations is None or t.annotations.destructiveHint is None
    ]
    assert missing == [], f"tools without an explicit destructiveHint: {missing}"

    # Only a tool that really removes something may claim to be destructive.
    destructive = {
        t.name for t in _all_tools()
        if t.annotations and t.annotations.destructiveHint is True
    }
    assert destructive <= {"alert_manage", "paper_wallet_reset",
                           "paper_order_cancel"}, (
        f"unexpected tools declaring destructiveHint=True: {sorted(destructive)}"
    )


def test_titles_are_unique_and_human_readable():
    tools = _all_tools()
    titles = [t.annotations.title for t in tools]
    assert len(set(titles)) == len(titles), "duplicate tool titles confuse directory listings"
    for title in titles:
        assert title != title.lower(), f"title looks like an identifier, not a human title: {title!r}"
        assert "_" not in title, f"title contains underscores: {title!r}"
