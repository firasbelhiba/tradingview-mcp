"""Who the HTTP transport will answer. Offline.

The server advertises `--host 0.0.0.0` for remote deployment and then refused
every request that arrived with a real Host header: the SDK switches on
DNS-rebinding protection whenever FastMCP is constructed with a localhost
host, and this module constructs it at import time, before `--host` is
parsed. So `allowed_hosts` was pinned to localhost for every launch.

Found by hosting it behind Caddy, which had to rewrite Host to
`localhost:8000` to get one call through. These tests drive the SDK's own
matcher rather than only inspecting our settings, because the bug was never
in what we stored — it was in what the middleware accepted.
"""
import pytest

from tradingview_mcp.server import _host_patterns, resolve_transport_security

PUBLIC = "mcp.example.com"


def _middleware(settings):
    from mcp.server.transport_security import TransportSecurityMiddleware

    return TransportSecurityMiddleware(settings)


def _accepts(settings, host):
    return _middleware(settings)._validate_host(host)


# ── The regression itself ─────────────────────────────────────────────────────

def test_a_public_host_is_accepted_when_it_was_named():
    settings, warning = resolve_transport_security("0.0.0.0", [PUBLIC])
    assert warning is None
    assert _accepts(settings, f"{PUBLIC}:10443"), "the deployed case"
    assert _accepts(settings, PUBLIC), "the same name reached on 443, no port"


def test_the_sdk_default_would_still_refuse_it():
    """Proof the bug was real and not a misreading of the error."""
    from mcp.server.transport_security import TransportSecuritySettings

    sdk_default = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
    )
    assert not _accepts(sdk_default, f"{PUBLIC}:10443")


def test_a_host_nobody_named_is_still_refused():
    settings, _ = resolve_transport_security("0.0.0.0", [PUBLIC])
    assert not _accepts(settings, "evil.example.com")
    assert not _accepts(settings, f"evil.example.com:{10443}")


def test_localhost_keeps_working_so_a_health_probe_does_too():
    """The container HEALTHCHECK calls http://localhost:8000/health."""
    settings, _ = resolve_transport_security("0.0.0.0", [PUBLIC])
    for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000"):
        assert _accepts(settings, host), host


# ── Both forms of a name, because one pattern misses the other ────────────────

def test_a_pattern_covers_the_name_with_and_without_a_port():
    assert _host_patterns("example.com") == ["example.com", "example.com:*"]


def test_the_wildcard_alone_would_miss_the_portless_name():
    """Why _host_patterns emits two entries and not just the wildcard.

    The SDK tests `host.startswith(base + ":")`, so `example.com:*` cannot
    match a bare `example.com` — which is what a client reaching the server
    on 443 sends.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    wildcard_only = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=["example.com:*"]
    )
    assert _accepts(wildcard_only, "example.com:443")
    assert not _accepts(wildcard_only, "example.com")


@pytest.mark.parametrize("junk", ["", "   ", "/"])
def test_an_empty_name_contributes_nothing(junk):
    assert _host_patterns(junk) == []


# ── Origins, which an empty list silently refuses ─────────────────────────────

def test_origins_are_populated_when_protection_is_on():
    """With protection on and allowed_origins empty, the SDK refuses any
    request carrying an Origin header at all — breaking a browser client."""
    settings, _ = resolve_transport_security("0.0.0.0", [PUBLIC])
    mw = _middleware(settings)
    assert mw._validate_origin(f"https://{PUBLIC}:10443")
    assert mw._validate_origin(f"https://{PUBLIC}")
    assert mw._validate_origin(None), "absent Origin is same-origin"
    assert not mw._validate_origin("https://evil.example.com")


# ── The two defaults ──────────────────────────────────────────────────────────

def test_a_public_bind_with_no_names_turns_protection_off_and_says_so():
    """Refusing every request would be worse, but it must not be quiet."""
    settings, warning = resolve_transport_security("0.0.0.0", [])
    assert settings.enable_dns_rebinding_protection is False
    assert warning is not None
    assert "--allowed-host" in warning, "the warning must name the fix"
    assert "TRADINGVIEW_MCP_ALLOWED_HOSTS" in warning
    assert "no authentication of its own" in warning, "say the rest out loud"


def test_a_local_bind_still_protects_itself():
    settings, warning = resolve_transport_security("127.0.0.1", [])
    assert warning is None
    assert settings.enable_dns_rebinding_protection is True
    assert _accepts(settings, "localhost:8000")
    assert not _accepts(settings, f"{PUBLIC}:10443")


def test_the_local_case_is_stated_not_left_to_none():
    """None is not "use the SDK default" — the middleware reads it as off.

    Returning None here would have fixed the remote case by quietly
    disabling protection for the local one.
    """
    settings, _ = resolve_transport_security("localhost", [])
    assert settings is not None
    assert _middleware(None).settings.enable_dns_rebinding_protection is False


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_every_localhost_spelling_counts_as_local(host):
    _, warning = resolve_transport_security(host, [])
    assert warning is None, f"{host} is not a public bind"


def test_blank_names_do_not_count_as_naming_a_host():
    """An env var set to "" or "," must not look like an explicit allow-list."""
    settings, warning = resolve_transport_security("0.0.0.0", ["", "  "])
    assert settings.enable_dns_rebinding_protection is False
    assert warning is not None


def test_several_names_are_all_accepted():
    settings, _ = resolve_transport_security(
        "0.0.0.0", [PUBLIC, "mcp.example.com"]
    )
    assert _accepts(settings, f"{PUBLIC}:10443")
    assert _accepts(settings, "mcp.example.com")


# ── The command line ──────────────────────────────────────────────────────────

def test_the_flag_is_repeatable_and_falls_back_to_the_environment(monkeypatch):
    import subprocess
    import sys

    code = (
        "import os,sys;"
        "sys.argv=['x','streamable-http']+sys.argv[1:];"
        "from tradingview_mcp.server import resolve_transport_security as r;"
        "import argparse;"
        "p=argparse.ArgumentParser();"
        "p.add_argument('transport',nargs='?');"
        "p.add_argument('--allowed-host',action='append');"
        "a=p.parse_args(sys.argv[1:]);"
        "allowed=a.allowed_host or [h for h in "
        "os.environ.get('TRADINGVIEW_MCP_ALLOWED_HOSTS','').split(',') "
        "if h.strip()];"
        "print(','.join(allowed))"
    )
    env = {**__import__("os").environ,
           "TRADINGVIEW_MCP_ALLOWED_HOSTS": "from.env,second.env"}
    out = subprocess.run(
        [sys.executable, "-c", code, "--allowed-host", "a.com",
         "--allowed-host", "b.com"],
        capture_output=True, text=True, env=env,
    )
    assert out.stdout.strip() == "a.com,b.com", "the flag wins over the env"

    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.stdout.strip() == "from.env,second.env"
