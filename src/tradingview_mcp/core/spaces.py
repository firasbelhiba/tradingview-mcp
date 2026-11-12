"""A private space per person, on a server several people share.

A hosted server had one data directory, so everyone holding its address
shared one wallet and one list of alerts: a friend asking to see "my wallet"
was shown the owner's open short, and could close it. Named wallets were
separation by courtesy - anyone could read or wipe any of them.

A space is a directory of its own, chosen by the link a request arrives on:

    https://host/<base path>/mcp                the owner's, as before
    https://host/<base path>/u/<key>/mcp        one person's

The key is random, handed out by the operator and listed in a registry file
in the data directory. An unknown key is a 404 with nothing to tell it apart
from any other wrong address. The registry is re-read when it changes, so
adding a person needs no restart, and removing their entry closes their link
at once while leaving their data on disk.

Nothing here applies to a local stdio install: with no HTTP request there is
no space, and every path is exactly what it was.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

HEADER = b"x-tv-space"
PREFIX = "/u/"
REGISTRY_NAME = "spaces.json"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

# The space the current tool call belongs to. None is the owner's.
CURRENT: ContextVar[Optional[str]] = ContextVar("tradingview_mcp_space", default=None)


def scoped(base: Path) -> Path:
    """``base`` for the owner, ``base/spaces/<name>`` inside a space."""
    name = CURRENT.get()
    return base / "spaces" / name if name else base


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


# ── The registry ──────────────────────────────────────────────────────────────

def data_dir() -> Path:
    configured = os.environ.get("TRADINGVIEW_MCP_DATA_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".tradingview-mcp"


def registry_path() -> Path:
    return data_dir() / REGISTRY_NAME


_cache_lock = Lock()
_cache: Dict[str, object] = {"path": None, "mtime": None, "keys": {}}


def _load(path: Path) -> Dict[str, str]:
    """key -> space name. A missing or unreadable registry is an empty one:
    no link works, which is the safe way to be wrong."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keys: Dict[str, str] = {}
    for key, entry in (raw.get("spaces") or {}).items():
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(key, str) and len(key) >= 16 and valid_name(name or ""):
            keys[key] = name
    return keys


def lookup(key: str) -> Optional[str]:
    """The space a key opens, or None. Re-reads the registry when it changes."""
    path = registry_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    with _cache_lock:
        if _cache["path"] != str(path) or _cache["mtime"] != mtime:
            _cache.update(path=str(path), mtime=mtime,
                          keys=_load(path) if mtime is not None else {})
        return _cache["keys"].get(key)  # type: ignore[union-attr]


def read_registry(path: Optional[Path] = None) -> dict:
    path = path or registry_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    raw.setdefault("spaces", {})
    return raw


def _write_registry(raw: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def add(name: str, path: Optional[Path] = None) -> str:
    """Create a space and return its key. A name can be used once."""
    name = (name or "").strip().lower()
    if not valid_name(name):
        raise ValueError(
            "a name is 1-32 characters: lowercase letters, digits, - and _, "
            "starting with a letter or digit"
        )
    path = path or registry_path()
    raw = read_registry(path)
    if any(entry.get("name") == name for entry in raw["spaces"].values()):
        raise ValueError(f"there is already a space called {name!r}")
    key = secrets.token_urlsafe(18)
    raw["spaces"][key] = {
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_registry(raw, path)
    return key


def remove(name: str, path: Optional[Path] = None) -> bool:
    """Close a space's link. Its data stays on disk. False if no such name."""
    path = path or registry_path()
    raw = read_registry(path)
    doomed = [k for k, entry in raw["spaces"].items() if entry.get("name") == name]
    for key in doomed:
        del raw["spaces"][key]
    if doomed:
        _write_registry(raw, path)
    return bool(doomed)


def link_suffix(key: str) -> str:
    """What to append to the server's base address for this key."""
    return f"{PREFIX}{key}/mcp"


# ── Reading the space off a request ───────────────────────────────────────────

class SpaceRouter:
    """ASGI middleware: turn ``/u/<key>/...`` into ``/...`` plus a trusted header.

    The header is how the space reaches a tool call, so one sent by a client
    is always removed first - otherwise anyone on the owner's link could
    name the space they wanted.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != HEADER]
        path = scope.get("path", "")
        if path.startswith(PREFIX):
            key, _, rest = path[len(PREFIX):].partition("/")
            name = lookup(key)
            if not name:
                await send({"type": "http.response.start", "status": 404,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Not Found"})
                return
            rewritten = "/" + rest
            scope = dict(scope, path=rewritten, raw_path=rewritten.encode(),
                         headers=headers + [(HEADER, name.encode())])
        else:
            scope = dict(scope, headers=headers)
        await self.app(scope, receive, send)


def from_request(request) -> Optional[str]:
    """The space a Starlette request was routed into, or None."""
    try:
        name = request.headers.get(HEADER.decode())
    except Exception:  # noqa: BLE001
        return None
    return name if name and valid_name(name) else None


# ── The operator's command ────────────────────────────────────────────────────

def main(argv=None) -> int:
    """``python -m tradingview_mcp.core.spaces add NAME | remove NAME | list``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tradingview_mcp.core.spaces",
        description="Give a person their own private wallets and alerts on a "
                    "shared server, by giving them their own link.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="create a space and print its link").add_argument("name")
    commands.add_parser("remove", help="close a space's link; its data is kept").add_argument("name")
    commands.add_parser("list", help="show every space and its link")
    args = parser.parse_args(argv)

    if args.command == "add":
        try:
            key = add(args.name)
        except ValueError as exc:
            print(f"Not created: {exc}")
            return 1
        print(f"Created the space {args.name.strip().lower()!r}.")
        print("Their link is your server's address with this in place of /mcp:")
        print(f"  {link_suffix(key)}")
        print("Anyone holding that link sees this space, so send it privately.")
        return 0
    if args.command == "remove":
        if not remove(args.name):
            print(f"There is no space called {args.name!r}.")
            return 1
        print(f"Closed the link for {args.name!r}. Its wallets and alerts are "
              f"still in {data_dir() / 'spaces' / args.name}.")
        return 0
    entries = read_registry()["spaces"]
    if not entries:
        print("No spaces yet. Everyone on this server shares the owner's data.")
        return 0
    for key, entry in sorted(entries.items(), key=lambda kv: kv[1].get("name", "")):
        print(f"{entry.get('name'):<20} created {entry.get('created', '?')}   {link_suffix(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
