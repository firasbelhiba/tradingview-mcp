# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **A yardstick for whether an assistant uses the tools well**
  (`evals/tool_selection/`). 35 things a trader actually types, in English
  and French, each with the tool that should be reached for first, and 8
  multi-step requests with the things a good run must and must not do
  (ask for an amount rather than invent one, never wipe a wallet unasked,
  preview a copy before confirming it). `score.py export` writes the tool
  list and the questions without the answers; `score` and `flows` mark a
  model's replies. First measurement, two models: 35/35 on the first tool
  with the full list, 34/35 and 35/35 from tool names alone, 18/18 on the
  flows. The descriptions were therefore left as they are. Running the
  calls the models chose is what found the three fixes below.
- **`crypto_bubbles`** returns the top N coins by market cap with the change
  over 1h, 24h, 7d, 30d and 1y in a single call, so a rendered view can
  switch window without refetching. Breadth comes with it. Source is
  CoinGecko's public markets endpoint, cached 60s — the exchange-pair
  screener returns `market_cap_calc` as null, so market-cap ranking cannot
  come from there. Stablecoins are dropped by default and the dropped ones
  named; the result says that `change_pct` is price change, not market-cap
  change.
- **`paper_copy_trader`** mirrors a lead trader's open positions into a
  simulated wallet. It previews by default and writes nothing until
  `confirm=true`. It is not copy trading and says so: you enter at today's
  price, not theirs, so every row reports `already_moved_pct` and the result
  leads with the position furthest along. Copied positions carry no stop,
  because OKX publishes what a trader holds and never where they would get
  out. Sizing is the wallet's, not the trader's, and leverage above the
  wallet's cap is capped with a note saying that makes it a different trade
  rather than a smaller one. A trader whose positions could not be read is
  refused outright rather than copied as an empty book.
- **Named wallets** (`paper_wallet_list`, and a `wallet` argument on every
  paper tool). The wallet table is `CHECK (id = 1)` and positions carry no
  wallet column, so a second wallet cannot share the database; each name gets
  its own SQLite file instead, which makes isolation total rather than a
  WHERE clause someone could forget. The default wallet keeps its original
  filename, so an existing installation opens exactly the database it always
  did. The name becomes a filename on a server that answers from the open
  internet, so it is validated to `[a-z0-9_-]{1,32}` before anything touches
  the filesystem. Separation, not privacy: every caller can name every
  wallet, and the result says so.
- **Resting limit orders for the paper wallet** (`paper_order_place`,
  `paper_order_list`, `paper_order_cancel`). The wallet could only fill at
  the current market price, so "buy at support" could not be expressed.
  Orders resolve retroactively the way stops already do: the next check scans
  the bars since placement, and a reached order fills at that bar's time with
  its stop and target settling from there. Margin is held from placement, as
  an exchange holds it. A limit fill at a touched level is reported as a best
  case — the opposite bias to a stop, and the results say so.

### Fixed
- **A mistyped trader code was described as a trader who hides.** OKX
  answers error 59269 both for a trader with private positions and for a
  code nobody has, so `lead_trader_profile` returned a profile with a null
  name and "position-level data is hidden" for any 16 characters typed at
  it. The published aggregates separate the two: measured live, they answer
  for a hidden trader and are refused for a made-up code.
  `lead_trader_profile`, `lead_trader_trades` and `paper_copy_trader` now
  return `INVALID_PARAMETER` saying there is no such trader and where codes
  come from. If that check cannot be made, nothing is concluded.
- **A scan that matched nothing answered with nothing.** Six screeners
  return a bare list (`top_gainers`, `top_losers`, `bollinger_scan`,
  `rating_filter`, `volume_breakout_scanner`, `smart_volume_scanner`) and
  an empty list reached the client as no content at all. An empty result
  is now an object: `matches: 0`, what was asked including defaults, and
  which setting to loosen. Rows are passed through untouched.
- **Refusing to wipe a wallet named only the destructive way forward.**
  Asked to create a wallet when one exists, `paper_wallet_reset` said to
  pass `confirm_wipe=true`. It now also says that `wallet='<name>'` makes
  a separate wallet and leaves this one alone.
- **Traders whose instruments OKX blanks were presented as ordinary
  positions.** Measured against the live leaderboard: 5 of the top 20
  return open positions with side, leverage and margin but no instrument,
  price or size. `lead_trader_profile` now marks each such position
  `instrument_hidden`, reports the state `instruments_hidden` when all of
  them are, and counts them in `open_positions_instrument_hidden` when
  only some are. `paper_copy_trader` refuses the first case with the real
  reason instead of "not a USDT pair this wallet can price", and skips
  only the blanked rows in the second. `lead_trader_trades` no longer
  repeats OKX's "this lead trader doesn't exist" for a trader whose
  history is merely hidden: it returns `NO_DATA` with
  `trade_history_state: hidden` and `retryable: false`, where it used to
  return a retryable `UPSTREAM_ERROR`.
- **`crypto_bubbles` failed on 2 calls in 5 from a hosted server.**
  CoinGecko's free tier answers 429 to a datacenter address straight
  after another CoinGecko-backed tool has run, and works seconds later.
  A 429 is now retried after 2.5s and 6s before being surfaced; any other
  status is not retried. That alone was measured and was not enough: five
  boards of different sizes back to back still failed three times, because
  each size was its own cache entry and so its own request. Every size now
  reads one shared page fetched at most once a minute, and when a refresh
  is refused a page up to 15 minutes old is shown with `data_age_s` and a
  `stale` note instead of an error. A failed stablecoin lookup is
  remembered for two minutes rather than repeated on every call, and a
  board of 100 now returns 100 (the over-fetch was capped at 100, so the
  stablecoins came out of it). `bitcoin_market_pulse` caches its two
  CoinGecko requests for 60s so it stops spending the allowance the next
  tool needs.
- **`--host 0.0.0.0` refused every request it received.** The MCP SDK enables
  DNS-rebinding protection whenever `FastMCP` is constructed with a localhost
  host, and `server.py` constructs it at import time — before `--host` is
  parsed — so `allowed_hosts` was pinned to `127.0.0.1/localhost/[::1]` for
  every launch and any request carrying a real `Host` header was answered
  `Invalid Host header`. That is the exact deployment the flag exists for.
  New `--allowed-host` (repeatable) and `TRADINGVIEW_MCP_ALLOWED_HOSTS`
  (comma separated) name the hostnames to accept; a non-localhost bind with
  neither runs with the protection off and warns on startup, since refusing
  everything silently is worse. A localhost bind is unchanged.

### Changed
- **What a caller can create is now bounded.** A hosted server protected
  only by its URL let any caller make wallets, positions, orders and
  alerts without end, and a wallet is a file on disk. Defaults: 25
  wallets, 50 open positions and 50 resting orders per wallet, 200
  alerts, each overridable by a `TRADINGVIEW_MCP_MAX_...` variable. Past
  the cap a tool returns the new `LIMIT_REACHED` and says what to remove.
  Resetting a wallet that already exists is never refused.
- **Optional call budget and audit line for shared deployments**, both
  off unless asked for, so a stdio install is untouched.
  `TRADINGVIEW_MCP_RATE_LIMIT_PER_MIN` refuses calls past that rate with
  the new `RATE_LIMITED` and a `retry_after_s`;
  `TRADINGVIEW_MCP_AUDIT_LOG=1` writes tool, duration and outcome to
  stderr and never the arguments. The budget is per process, not per
  caller, because the server cannot tell callers apart.
- **`lead_trader_trades` returns 40 rows by default, not all of them.**
  The full set measured 95-150 KB in one answer. Statistics still cover
  every trade read; `trades_shown` and `trades_omitted` say what was left
  out and `trade_rows` raises the cap.
- **A missing paper position or alert now returns `RECORD_NOT_FOUND`, not
  `SYMBOL_NOT_FOUND`**: both are rows in this server's own database, and the
  symbol code sent clients hunting for a ticker when an id had simply been
  deleted. Both envelopes also now state `retryable: false`, which they left
  unset. If you branch on `SYMBOL_NOT_FOUND` for `paper_trade_close`,
  `paper_trade_modify` or `alert_manage`, that branch needs the new code.
  Genuine unlisted-symbol errors are unchanged.
- **`valid_exchanges` in an `INVALID_EXCHANGE` envelope is now grouped**
  `{"crypto": [...], "stock": [...]}` rather than one flat list of 67, and a
  near-miss gets `did_you_mean`. No accepted value was dropped. Code reading
  that field as a flat list needs updating.
- **`valid_timeframes` is now in time order** (`5m, 15m, 1h, 4h, 1D, 1W,
  1M`) instead of alphabetical (`15m, 1D, 1M, 1W, 1h, 4h, 5m`).

### Fixed
- **`compare_strategies` could name a strategy with no trades the winner**:
  the leaderboard was sorted by return alone, and a strategy that never
  trades reports 0%. When buy-and-hold and every strategy that traded lost
  money, that 0% came first, so rank 1 and `winner` went to a strategy with
  no trades (APP on 1y daily: `rsi_pullback`, 0 trades, buy-and-hold −37%).
  Strategies that traded now rank first, by return; strategies without
  trades rank last and are never the winner. `winner` is null when no
  strategy traded, and `warnings` names the strategies without trades.
- **No volume baseline in the TA-based tools** (`coin_analysis`,
  `multi_timeframe_analysis`, the stock score and trade quality, and the EGX
  tools): the indicator helpers divided by a `volume.SMA20` column that
  tradingview_ta never requests and TradingView's scanner doesn't have. The
  effects:
  - `volume_analysis` always had a null ratio and a "Normal" signal.
  - The timeframe context never mentioned volume.
  - The stock score's volume points, breakout bonus and low-volume penalty
    never applied, nor did trade quality's volume and liquidity points.
  - `egx_sector_scanner` showed every sector's flow as "Inflow".
  - `egx_stock_screener`'s liquidity filter never fired.
  - `egx_fibonacci_retracement` always returned a null volume ratio.

  The TA fetch now also requests `relative_volume_10d_calc` (the bar's volume
  over the mean of the previous 10 bars) and `average_volume_10d_calc` (mean
  volume of the last 10 bars), and the helpers use them. Stock scores and trade
  quality scores change accordingly. `volume_analysis` gains `average_10`, and
  `average_20` stays as a null key for compatibility.
- **Volume tools never saw a volume baseline** (`volume_breakout_scanner`,
  `smart_volume_scanner`, `volume_confirmation_analysis`): the volume ratio
  was computed from a `volume.SMA20` column that was never fetched.
  tradingview_ta doesn't request it, and TradingView's scanner has no
  volume-SMA field. Since the baseline fallback was removed, the breakout
  scans skipped every symbol and always returned an empty list, and
  `volume_confirmation_analysis` always reported a 1.0x ratio. The ratio is
  now TradingView's relative volume (`relative_volume_10d_calc`): the bar's
  volume over the mean of the previous 10 bars at that timeframe. The scans
  also move to a single scanner request for the whole exchange. They used to
  run 100-symbol batches capped at the first 500 symbols, so on big
  exchanges only the alphabetical head was scanned.
- **US-stock scans failed or timed out** (`top_gainers`, `top_losers`,
  `rating_filter`, `bollinger_scan`): the scans called `get_multiple_analysis`
  in 200-symbol batches, each asking for ~90 indicator columns, so NASDAQ
  (~5,000 tickers) took 25 requests and NYSE 14. The burst ran past the 30s
  scan budget (`PARTIAL_DATA`) or tripped TradingView's empty-body throttling
  (`ALL_BATCHES_FAILED`), so most NASDAQ/NYSE scans failed, and
  `bollinger_scan`, which has no budget, could run for minutes. Each scan is
  now a single scanner request for the eight columns it reads (about a second
  for NASDAQ), with the same retry, 60s cache and stale-while-error fallback.
  Results are unchanged.
- **TradingView interval codes rejected as timeframes**: `"60"`, `"240"`,
  `"15"`, `"5"`, `"1440"`, `"D"` and `"W"` are now accepted and mapped to
  `1h`, `4h`, `15m`, `5m` and `1D` / `1W`. Integrations sending TradingView's
  native codes used to silently get the tool's default timeframe instead
  (a `"60"` request returned 15m data), and since 0.9.0's strict validation
  they got `INVALID_TIMEFRAME`. Unsupported intervals (`30m`, `120`, ...)
  still fail loudly.
- **Futures watchlist returned nothing for the main US index futures** (#98):
  `FUTURES_WATCHLIST` used exchange prefixes TradingView's scanner doesn't
  index (`CME:ES1!` instead of `CME_MINI:ES1!`, likewise NQ, RTY, YM and EMD,
  plus livestock under `CBOT` instead of `CME`). A wrong prefix returns 0
  rows without an error, so `futures_watchlist`, `futures_category_snapshot`
  and `futures_market_overview` were silently empty for those contracts.
- **Stale US stock coinlists** (#96): `nasdaq.txt` and `nyse.txt` hadn't been
  refreshed since 2025. About 1,500 entries were no longer listed on that
  venue, about 1,600 newer listings were missing, and 18 tickers had moved
  venue (WMT, AZN, KHC, ETSY, QBTS, QS and others), which broke error
  suggestions, venue fallback and exchange-level scan coverage. Regenerated
  from the scanner.

### Added
- **India and Indonesia**: NSE, BSE and IDX route to their TradingView markets
  and ship scanner-built coinlists (stocks and depositary receipts) for
  exchange-level scans. The benchmark indices resolve by name whatever exchange
  the caller passes: `NIFTY` / `NIFTY50` / `^NSEI`, `BANKNIFTY` / `^NSEBANK`,
  `SENSEX` / `^BSESN`, and `JKSE` / `^JKSE` / `IHSG` for the Jakarta Composite.
- **European and Canadian stock exchanges** (#95, thanks @Zulimon): Euronext
  Paris, Amsterdam, Brussels and Lisbon (`EPA`, `AMS`, `BRU`, `LIS`), Borsa
  Italiana (`MIL`), London (`LSE`), SIX (`SIX`), Madrid (`BME`), Xetra
  (`XETRA`), Frankfurt (`FWB`), Toronto (`TSX`) and TSX Venture (`TSXV`), plus
  common aliases. Each routes to its TradingView market and symbol prefix and
  ships a scanner-generated coinlist (stocks and depositary receipts) for
  exchange-level scans. Euronext's four markets share one `EURONEXT` prefix,
  so the market follows the venue the caller named. Venue fallback now stays
  in the requested venue's market first, so a ticker that exists on both
  sides of the Atlantic (MC is Moelis on NYSE, LVMH in Paris) doesn't jump
  continents.
- `scripts/refresh_coinlists.py` rebuilds the stock coinlists (US, European
  and Canadian venues) from the scanner. `--check` reports drift without
  writing (exit 1 if stale), and a guard refuses to write a list that shrank
  suspiciously.

## [0.9.0] - 2026-08-26

### Changed (behavior)
- **Strict input validation**: tools now return `INVALID_EXCHANGE` /
  `INVALID_TIMEFRAME` error envelopes (listing valid values) instead of
  silently substituting the default — `exchange="KRAKEN"` used to return
  KUCOIN data with no warning. Aliases (`1d`→`1D`) and omitted parameters
  still resolve silently.
- **`PARTIAL_DATA` envelopes**: batched scans that abort mid-flight
  (wall-clock budget / consecutive-failure bail) return a `PARTIAL_DATA`
  envelope that still carries the collected rows, instead of a plain list
  indistinguishable from a complete scan.

### Fixed
- **`top_losers` returned gainers**: the service sorted descending and
  truncated to `limit` before the tool re-sorted ascending, so it returned
  the smallest of the top gainers. Sorting now happens before truncation.
- **Volume breakout scanner fabricated breakouts**: symbols missing a
  `volume.SMA20` baseline got a fallback ratio that was always exactly 2.0
  and passed the default gate on price change alone. They are now skipped.
- **Walk-forward robustness inverted for losing strategies**: a strategy
  losing MORE out-of-sample scored a capped 2.0 ("maximally robust") in the
  both-negative branch. Warmup-starved test folds are flagged
  `insufficient_data` and excluded instead of reading as OVERFITTED.
- **Multi-timeframe alignment misattribution**: a failed timeframe shifted
  every later score onto the wrong timeframe key in `scores_by_tf`.
- **`fetch_multi_timeframe_patterns` ignored its `symbols` argument**
  (whole-exchange scan) while caching on it.
- **Backtest metrics**: drawdown/Calmar/Sharpe now come from a per-bar
  mark-to-market equity series (intra-trade dips count); open positions at
  data end are force-closed and flagged `forced_exit`; the buy-and-hold
  benchmark pays the same round-trip costs as the strategy;
  `compare_strategies` validates `period`.
- **Thread safety**: the Yahoo options session handshake is lock-guarded;
  the screener cache is bounded (256 entries); cached indicator dicts are
  copied before ATR backfill; the resilience layer's stale-while-error
  cache is actually populated on success for `cache_key` callers.
- **Timeouts everywhere**: all `get_scanner_data` calls pass explicit
  timeouts (`requests` has no default).
- **Docker HEALTHCHECK**: the server now serves `GET /health` under the
  streamable-http transport, so containers stop cycling to `unhealthy`.
- Marketaux sentiment: whole-word keyword scoring and exact entity-symbol
  matching; multi-agent analysis no longer crashes on explicit-null
  indicators; proxy credentials are URL-encoded.

### Removed
- Dead code (~800 lines): the unwired paper-trading `portfolio.py`, the
  RSS `news_service.py` and Reddit `sentiment_service.py` (both replaced by
  Marketaux in 0.6), unused fetchers in `screener_provider.py` /
  `screener_service.py`, and the stray `PR_BODY.md`.

### Infrastructure
- **CI now runs the test suite** (Python 3.10–3.13 matrix) on every
  push/PR, and the Docker image publish is gated on it.
- `pandas` declared as a direct dependency (server.py imports it at module
  top; it previously arrived only via the tradingview-screener pin).
- `docker-compose.yml` points at the GHCR image CI actually builds.
- Real `SECURITY.md` (GitHub Security Advisories; 0.9.x supported).

## [0.8.1] - 2026-08-02

### Fixed
- Raised the `mcp` lower bound to `>=1.14.0` (still `<2`): with
  `from __future__ import annotations`, `Tool.from_function` in <=1.13.x
  dies at import time on string annotations, so the server never started
  on those SDK versions.

### Added
- Official MCP Registry manifest (`server.json`) + OIDC publish workflow.

## [0.8.0] - 2026-07-29

### Fixed
- **CRITICAL — pinned `mcp[cli]>=1.12.0,<2`**: `mcp` 2.0.0 (released 2026-07-28
  alongside the MCP 2026-07-28 stateless spec) removes the `mcp.server.fastmcp`
  module this server imports. 0.7.1's published metadata carries the open
  `>=1.12.0` range, so every fresh `pip install` / `uvx` since 2026-07-28
  resolved mcp 2.0.0 and died on startup with
  `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`. Upgrade to
  0.8.0 to get a working install. Migrating to the 2.x SDK
  (FastMCP → MCPServer) will be a deliberate follow-up release.
- **`volume_confirmation_analysis` bare-symbol failures**: bare crypto symbols
  (no exchange prefix) failed ~99% of the time; symbols are now resolved
  before analysis.
- **Donchian breakout off-by-one**: the breakout strategy read a window that
  included the current bar, inflating backtest results; window is now strictly
  historical (regression-tested).
- **Backtest input validation**: initial capital and cost parameters are now
  validated instead of silently producing nonsense results.
- **`coin_analysis` ATR null bug**: `tradingview_ta` omits the `ATR` column from
  its analysis payload, which left `indicators["ATR"]` (and every downstream
  consumer — stop-loss sizing, trade-quality score, volatility metrics) at
  `None` on every call. `analyze_coin` now falls back to a direct
  `scanner.tradingview.com/<market>/scan` request via the new
  `fetch_atr_for_ticker()` helper in `screener_provider.py`. The lookup is
  best-effort (silent `None` on any network / parse failure) and only fires
  when `tradingview_ta` returned no ATR, so it does not regress healthy
  payloads. Timeframe → resolution mapping is shared with the existing
  screener column logic (`5m→5`, `15m→15`, `1h→60`, `4h→240`, `1D/1W/1M`
  unchanged); unknown timeframes degrade to the unsuffixed `ATR` column.

### Added
- **Markets**: Taiwan (TWSE, TPEX), Saudi Arabia (TADAWUL), US futures across
  CME/COMEX/NYMEX/CBOT (equity index, energy, metals, agriculture, rates, FX,
  crypto futures), AMEX/NYSEARCA aliases, auto-venue fallback for unlisted
  symbols, precious-metal futures resolution via TVC.
- **Tools**: `stock_screener` + `stock_prices` (bulk fetches up to 1,000 rows,
  `exclude_otc`, server-side sorting, daily OHLC), `stock_options_chain` +
  `stock_options_unusual_activity`, `stock_extended_hours`,
  `bitcoin_market_pulse`.
- **Directory-grade tool metadata**: every tool now ships `title`,
  `readOnlyHint`, and an explicit `destructiveHint=False` annotation.
- **Multi-arch Docker image CI**: GHCR images built for linux/amd64 +
  linux/arm64 on every push to main and on version tags.

### Changed
- **Every tool now runs off the event loop** (blanket async offload) — a slow
  upstream call can no longer block the MCP server's event loop.
- **Structured error envelopes everywhere**: screener/scanner failures return
  machine-readable envelopes with retryability signals and symbol
  suggestions instead of bare strings; note that tool return types are now
  `list[dict] | dict` (consumers that assumed a bare list should handle the
  envelope shape).
- **News pipeline**: RSS/Reddit scraping replaced with the licensed Marketaux
  API. Set `MARKETAUX_API_TOKEN` (see `.env.example`) to enable
  `financial_news` and news-driven sentiment; without a token the news tools
  return empty results while everything else works normally.
- **Reliability**: bounded HTTP timeouts with stale-while-error fallback,
  retry + 60s TTL response cache, `tradingview_ta` request throttling,
  fast-fail on upstream outages.
- Python support capped below 3.14 (`requires-python >=3.10,<3.14`) until the
  dependency stack publishes 3.14 wheels.
- `requests` is now an explicit dependency in `pyproject.toml`. It was already
  pulled in transitively by `tradingview-screener` / `tradingview-ta`, but the
  new ATR injection path uses it directly, so it is no longer safe to rely on
  the transitive resolution.

## [0.7.1] - 2026-04-14

### Added
- **MEXC Exchange Support** (`exchange="MEXC"`):
  - 420 MEXC trading pairs added — including many unique pairs not available on Binance, KuCoin, or Bybit
  - Fully supported in all tools: `top_gainers`, `top_losers`, `bollinger_scan`, `coin_analysis`, `multi_agent_analysis`, `volume_breakout_scanner`, `consecutive_candles_scan`, `advanced_candle_pattern`, and more
  - MEXC correctly categorized as a crypto screener market

---

## [0.7.0] - 2026-03-29

### Added
- **Walk-Forward Backtesting** (`walk_forward_backtest_strategy`):
  - Splits data into N folds (train/test) to validate strategy on unseen forward data
  - Per-fold in-sample vs out-of-sample return comparison
  - **Robustness score** (test/train ratio): ROBUST ≥ 0.8 | MODERATE ≥ 0.5 | WEAK ≥ 0.2 | OVERFITTED < 0.2
  - Aggregate out-of-sample metrics: Sharpe, win rate, max drawdown, total return
  - Supports 2–10 splits, configurable train ratio, both 1d and 1h intervals
- **Full Trade Log** (`include_trade_log=True`):
  - Per-trade breakdown: entry/exit date & price, holding days, gross/net return %, cost %
  - Running capital and cumulative return at each trade
- **Equity Curve** (`include_equity_curve=True`):
  - Capital value + drawdown % at each trade exit — ready for charting
- **1h (Hourly) Timeframe** (`interval="1h"`):
  - All strategies and compare now support intraday hourly data
  - Sharpe ratio annualization corrected for 1h bars (252 × 6 trading hours)
  - Works on `backtest_strategy`, `compare_strategies`, and `walk_forward_backtest_strategy`

### Changed
- `backtest_strategy` tool: added `interval`, `include_trade_log`, `include_equity_curve` params
- `compare_strategies` tool: added `interval` param; now documents all 6 strategies (was 4)
- `run_backtest()` now returns last 5 trades always (`recent_trades`) for quick inspection
- Sharpe ratio calculation now uses interval-aware annualization factor

---

## [0.6.0] - 2026-03-29

### Added
- **Backtesting Engine v2** (`backtest_strategy`, `compare_strategies`):
  - 6 trading strategies: RSI, Bollinger Band, MACD, EMA Cross, **Supertrend** (🔥 trending 2025), **Donchian Channel** (Turtle Trader classic)
  - Institutional-grade metrics: Sharpe Ratio, Calmar Ratio, Expectancy, Profit Factor, Max Drawdown
  - Transaction cost simulation: per-trade commission + slippage
  - Buy-and-hold benchmark comparison
  - Single OHLCV fetch for `compare_strategies` (all 6 strategies in ~0.3s)
- **Yahoo Finance Integration** (`yahoo_price`, `market_snapshot`):
  - Real-time quotes for stocks, crypto, ETFs, indices (S&P500, NASDAQ, VIX), FX
  - Global market snapshot with 14 instruments across 4 asset classes
  - Turkish stocks supported (THYAO.IS, SASA.IS...)
- **Webshare Rotating Proxy Manager**:
  - 250 sticky sessions for rate-limit bypass
  - Direct-first + proxy-fallback architecture for reliability
  - Zero-config for users (optional env-based configuration)
- **Technical Indicators (pure Python, zero deps)**:
  - ATR (Average True Range)
  - Supertrend
  - Donchian Channel

### Changed
- `compare_strategies` now fetches OHLCV once and runs all strategies on cached data (5x faster)
- Yahoo Finance data fetching uses direct connection first, proxy fallback only on failure

## [0.5.0] - 2026-03-29

### Added
- **Real-Time Market Sentiment (Agent-Reach Integration)**: Integrated Reddit JSON API to track symbol sentiment across finance communities (`market_sentiment`).
- **Live Financial News RSS**: Added `fetch_news` service via `feedparser` to track real-time headlines across Reuters, CoinDesk, and CoinTelegraph (`financial_news`).
- **Combined Analysis Power Tool**: The new `combined_analysis` tool merges TradingView technicals, Reddit sentiment, and live news into a single confluence analysis (signals agree/conflict, confidence score, full recommendation).
- Added `feedparser` dependency to `pyproject.toml`.

## [0.4.0] - 2026-03-29

### Added
- **EGX (Egyptian Exchange) Full Support**: Complete trading infrastructure for the Egyptian Stock Market.
  - `egx_market_overview`: Top gainers, losers, most active stocks, and market breadth stats (advancing/declining/unchanged).
  - `egx_sector_scan`: Scan across 18 EGX sectors (banks, real_estate, healthcare_and_pharma, technology, etc.).
  - `egx_stock_screener`: Cross-sectional ranking with auto trade plan generation.
  - `egx_trade_plan`: Single-stock detailed trade setup (entry, stop-loss, targets, R:R).
  - `egx_fibonacci_retracement`: Fibonacci retracement + extension levels with golden pocket detection.
  - 6 EGX indices: EGX30, EGX70, EGX100, SHARIAH33, EGX35LV, TAMAYUZ (200+ symbols).
- **3-Layer Stock Decision Engine**:
  - Layer A: 100-point stock ranking model (trend, momentum, risk, fundamentals).
  - Layer B: Trade setup engine (entry points, stop-loss, targets, support/resistance, R:R).
  - Layer C: Trade quality scoring (structure, risk/reward, volume, liquidity).
- **Liquidity-Aware Scoring**: Hard grade caps prevent illiquid stocks from ever receiving "Strong" or "Elite" grades.
- **23 Technical Indicators**: Expanded from 5 → 23: CCI, Williams %R, Awesome Oscillator, Momentum, Parabolic SAR, Ichimoku, Stoch RSI, ADX +DI/-DI, Hull MA, VWMA, Ultimate Oscillator, full EMA/SMA suites.
- **Multi-Timeframe Alignment**: Weekly→Daily→4H→1H→15m bias analysis with timeframe-specific advice.

### Fixed
- Hardcoded `"crypto"` market type in `_fetch_multi_changes`, `_fetch_multi_timeframe_patterns`, and `screener_provider` — now dynamically resolved per exchange (egypt, turkey, america, etc.).
- `volume_confirmation_analysis` was appending `"USDT"` to stock symbols — now exchange-aware with proper `EGX:SYMBOL` formatting.

### Changed
- MCP server name updated to "TradingView Multi-Market Screener".
- All tool descriptions updated to reference both crypto and stock exchanges.
- Added `is_stock_exchange()`, `get_market_type()`, and `STOCK_EXCHANGES` helpers to `validators.py`.

## [0.3.0] - 2026-03-24

### Added
- **Docker Support**: Official Dockerfile and docker-compose.yml for easy 1-click self-hosting.
- **PyPI Release**: Added proper metadata and structuring for PyPI distribution (`pip install tradingview-mcp`).

## [0.2.0] - 2026-03-24

### Added
- **Multi-Agent Trading Framework**: Introduced `multi_agent_analysis` MCP tool.
  - **Technical Analyst Agent**: Analyzes RSI, MACD, and Bollinger Bands.
  - **Sentiment Analyst Agent**: Calculates momentum and produces a sentiment score.
  - **Risk Manager Agent**: Evaluates volatility (BBW) and mean reversion risk.
- **Debate System**: Agents combine their scores to provide a single, logical Framework Decision (Strong Buy, Buy, Hold, Sell, Strong Sell) with confidence levels.

### Changed
- Repositioned the project from a "screener" to an "AI Trading Intelligence Framework".
- Updated `README.md` to reflect the new architecture.

## [0.1.0] - Initial Release
- Basic MCP Server setup.
- Bollinger Band squeeze detection.
- Consecutive candle pattern detection.
- Real-time market screening (gainers, losers).
