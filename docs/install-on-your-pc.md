# Install it on your own PC

This gives Claude a set of crypto tools: market overview, coin analysis,
trade plans, alerts, top traders, and a **practice wallet** for trying
trades with pretend money. Everything runs on your computer. Your wallet
and alerts are yours alone and stay on your machine.

It takes about five minutes. You need the **Claude desktop app** and nothing
else: no Python, no git, no account, no API key.

> Nothing here places a real order or touches real money. It is a tool for
> analysis and practice, not financial advice.

## Windows

### 1. Install uv

uv is a small program that downloads and runs the tools for you. Open
**PowerShell** (press Start, type `powershell`, press Enter) and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

When it finishes, **close PowerShell and open it again**.

### 2. Download the tools once

Paste this into the new PowerShell window:

```powershell
uvx --python 3.13 --from https://github.com/firasbelhiba/tradingview-mcp/archive/refs/heads/crypto-swing-tools.zip tradingview-mcp --help
```

The first time it downloads and builds for about a minute. It is done when
you see a short help text. If you see that, it works.

### 3. Tell Claude about it

In the Claude desktop app: **Settings → Developer → Edit Config**. A file
called `claude_desktop_config.json` opens. Replace what is in it with this
and save:

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "uvx",
      "args": [
        "--python", "3.13",
        "--from", "https://github.com/firasbelhiba/tradingview-mcp/archive/refs/heads/crypto-swing-tools.zip",
        "tradingview-mcp"
      ]
    }
  }
}
```

If the file already has other things in `mcpServers`, keep them and add the
`"tradingview": { ... }` part next to them, separated by a comma.

### 4. Restart Claude completely

Closing the window is not enough. Right-click the Claude icon near the
clock (bottom right of the screen) and choose **Quit**, then open Claude
again.

### 5. Try it

Start a new chat and type:

> I'm new here, what can you do with the tradingview tools?

Then try:

- `how's the market today?`
- `analyse BTC for me`
- `create me a practice wallet with 5000 dollars`
- `show me the bubble chart of the top 30 coins this week`
- `give me the top traders`

## Mac

Same steps, with two differences. Install uv from the **Terminal** app with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

and quit Claude with **Cmd + Q** instead of the tray icon.

## If something goes wrong

- **Claude shows no tradingview tools.** Make sure you really quit Claude
  (step 4), and that step 2 showed the help text.
- **Claude says it cannot find `uvx`.** In the config file, replace `"uvx"`
  with its full path. On Windows that is
  `"C:\\Users\\YOUR-NAME\\.local\\bin\\uvx.exe"` with your Windows user name,
  and the double backslashes are needed.
- **The config file will not save or Claude complains about it.** One
  missing comma or bracket breaks it. Paste the block from step 3 again,
  exactly.

## Getting updates

New features arrive often. To get them: quit Claude completely, run this in
PowerShell, and open Claude again. The next start takes about a minute.

```powershell
uv cache clean
```

## Where your data lives

Your practice wallet and alerts are in a folder called `.tradingview-mcp`
inside your user folder. Delete that folder to start from nothing. Nothing
in it is sent anywhere.
