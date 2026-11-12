"""
Correlation Service — is this its own trade, or leveraged BTC?

Two computations over candles the package already fetches. No new upstream.

  * **Correlation and beta against a benchmark.** The macro tools keep
    reporting ALT_RISK and high BTC dominance; this makes the implication
    concrete per symbol. An alt with beta 1.6 and r-squared 0.9 is not a
    diversifying position — it is a leveraged bet on the benchmark, and
    holding five of them is one trade in five tickets.
  * **Seasonality.** Average return by weekday and by hour. Crypto trades
    24/7, so unlike equities it actually has an hour-of-day distribution to
    look at, and nothing in the server looked.

Both report sample counts beside every figure. A weekday bucket with six
observations is not evidence, and a mean that does not say how many bars it
averaged invites exactly that mistake.
"""
from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.crypto_analysis_service import (
    binance_symbol_for,
    fetch_crypto_candles,
)

_MAX_SYMBOLS = 20
_MIN_OVERLAP = 30          # paired observations below this is not a correlation
_MIN_BUCKET_SAMPLES = 10   # below this a seasonality bucket is labelled thin

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")


def _returns_by_date(candles: Sequence[dict]) -> Dict[str, float]:
    """Bar-over-bar returns keyed by the bar's own timestamp.

    Keyed rather than positional so two symbols are aligned by TIME. Index
    alignment would silently pair Monday on one symbol with Tuesday on
    another whenever their histories start at different points or one has a
    gap, which inflates or destroys the correlation for no visible reason.
    """
    out: Dict[str, float] = {}
    for prev, cur in zip(candles, candles[1:]):
        prev_close = prev.get("close")
        cur_close = cur.get("close")
        if not prev_close or cur_close is None:
            continue
        out[cur["date"]] = (cur_close - prev_close) / prev_close
    return out


# A return series flatter than this has no dispersion worth correlating: a
# pegged stablecoin pair, a halted market, or floating-point residue. Testing
# the denominator against exactly zero is not enough — residue of order 1e-18
# survives it and then the "correlation" describes rounding error, which comes
# back as a confident-looking number like 0.178.
_MIN_DISPERSION = 1e-12


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sd_x = sqrt(sum(v * v for v in dx) / (n - 1))
    sd_y = sqrt(sum(v * v for v in dy) / (n - 1))
    if sd_x < _MIN_DISPERSION or sd_y < _MIN_DISPERSION:
        return None
    denom = sqrt(sum(v * v for v in dx)) * sqrt(sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _beta(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Slope of symbol returns (ys) on benchmark returns (xs)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if sqrt(var_x / (n - 1)) < _MIN_DISPERSION:
        return None  # a flat benchmark has no slope to measure against
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / var_x


def _label(correlation: Optional[float], beta: Optional[float]) -> str:
    if correlation is None:
        return "unknown"
    if correlation >= 0.85:
        if beta is not None and beta >= 1.25:
            return "leveraged benchmark — same trade, bigger size"
        if beta is not None and beta <= 0.75:
            return "dampened benchmark — same trade, smaller size"
        return "tracks the benchmark"
    if correlation >= 0.5:
        return "partly independent"
    if correlation > -0.5:
        return "largely independent"
    return "inversely correlated"


def analyze_correlation(
    symbols: List[str],
    benchmark: str = "BTCUSDT",
    timeframe: str = "4h",
    lookback: int = 180,
) -> dict:
    """Correlation, beta and r-squared of each symbol against a benchmark.

    Args:
        symbols:   Pairs to measure, max 20.
        benchmark: Pair to measure against.
        timeframe: Candle interval.
        lookback:  Bars to use, 40-1000.
    """
    if not symbols:
        return make_error(ErrorCode.INVALID_PARAMETER, "symbols must not be empty")
    if not 40 <= lookback <= 1000:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"lookback must be 40-1000, got {lookback}"
        )

    requested = [s.strip().upper() for s in symbols if s and s.strip()][:_MAX_SYMBOLS]
    if not requested:
        return make_error(ErrorCode.INVALID_PARAMETER, "symbols must not be empty")

    bench = benchmark.strip().upper()
    try:
        bench_candles, bench_source = fetch_crypto_candles(bench, timeframe)
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"Could not fetch the benchmark {bench}: {str(exc)[:100]}",
            benchmark=bench, retryable=True,
        )
    bench_returns = _returns_by_date(bench_candles[-lookback:])
    if len(bench_returns) < _MIN_OVERLAP:
        return make_error(
            ErrorCode.NO_DATA,
            f"Only {len(bench_returns)} benchmark returns available — "
            f"need at least {_MIN_OVERLAP}",
            benchmark=bench,
        )

    rows: List[dict] = []
    unavailable: List[str] = []
    sources = {bench_source: 1}
    for symbol in requested:
        if symbol == bench:
            continue
        try:
            candles, source = fetch_crypto_candles(symbol, timeframe)
        except Exception:
            unavailable.append(symbol)
            continue
        sources[source] = sources.get(source, 0) + 1
        series = _returns_by_date(candles[-lookback:])
        shared = sorted(set(series) & set(bench_returns))
        if len(shared) < _MIN_OVERLAP:
            unavailable.append(symbol)
            continue
        ys = [series[d] for d in shared]
        xs = [bench_returns[d] for d in shared]
        corr = _pearson(xs, ys)
        beta = _beta(xs, ys)
        rows.append({
            "symbol": symbol,
            # The pair the candles came from, when it is not what was typed:
            # "sol" is measured as SOLUSDT, and the row should say so.
            **({"measured_as": binance_symbol_for(symbol)}
               if binance_symbol_for(symbol) != symbol else {}),
            "correlation": None if corr is None else round(corr, 3),
            "r_squared": None if corr is None else round(corr * corr, 3),
            "beta": None if beta is None else round(beta, 3),
            "observations": len(shared),
            "reading": _label(corr, beta),
        })

    if not rows:
        return make_error(
            ErrorCode.NO_DATA,
            "No symbol had enough overlapping history with the benchmark",
            benchmark=bench, unavailable=unavailable or None,
        )

    rows.sort(key=lambda r: -(r["correlation"] or -1))
    return {
        "benchmark": bench,
        "timeframe": timeframe,
        "lookback_bars": lookback,
        "benchmark_observations": len(bench_returns),
        "candle_sources": sources,
        "alignment_note": (
            "Returns are paired by bar timestamp, not by position, so a gap or "
            "a different history start cannot mis-pair two symbols."
        ),
        "rows": rows,
        "unavailable": sorted(unavailable) or None,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


# ── Seasonality ───────────────────────────────────────────────────────────────

def _bucket_stats(values: Sequence[float]) -> dict:
    """Mean, dispersion and a t-statistic for one bucket.

    The mean alone is the trap here. A weekday averaging +0.1% sounds like an
    edge until you see the bar-to-bar standard deviation is 1.5% — the mean is
    then a rounding error on noise. ``t_stat`` is mean / standard error; by
    convention |t| under ~2 is indistinguishable from zero, and
    ``distinguishable_from_zero`` says so outright so a reader does not have
    to know that convention.
    """
    n = len(values)
    if not n:
        return {"samples": 0, "mean_return_pct": None, "win_rate_pct": None,
                "thin_sample": True}
    mean = sum(values) / n
    positive = sum(1 for v in values if v > 0)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = sqrt(variance)
    std_error = std / sqrt(n) if n and std else 0.0
    t_stat = (mean / std_error) if std_error else None
    return {
        "samples": n,
        "mean_return_pct": round(mean * 100, 4),
        "std_dev_pct": round(std * 100, 4),
        "win_rate_pct": round(positive / n * 100, 1),
        "t_stat": None if t_stat is None else round(t_stat, 2),
        "distinguishable_from_zero": (
            None if t_stat is None else abs(t_stat) >= 2.0
        ),
        "thin_sample": n < _MIN_BUCKET_SAMPLES,
    }


def analyze_seasonality(
    symbol: str,
    timeframe: str = "4h",
    lookback: int = 500,
) -> dict:
    """Average return by weekday and by hour of day (UTC).

    Crypto trades 24/7, so an hour-of-day distribution genuinely exists —
    unlike equities, where it would just describe the session.

    Every bucket carries its sample count and a ``thin_sample`` flag. A
    weekday with six observations is not evidence, and the flag is there so a
    reader does not act on one.

    Args:
        symbol:    Pair to analyse.
        timeframe: Candle interval. Hourly buckets need an intraday interval.
        lookback:  Bars to use, 50-1500.
    """
    if not 50 <= lookback <= 1500:
        return make_error(
            ErrorCode.INVALID_PARAMETER, f"lookback must be 50-1500, got {lookback}"
        )
    try:
        candles, source = fetch_crypto_candles(symbol, timeframe)
    except ValueError as exc:
        return make_error(ErrorCode.INVALID_TIMEFRAME, str(exc))
    except Exception as exc:  # noqa: BLE001
        return make_error(
            ErrorCode.UPSTREAM_ERROR, f"Candle fetch failed: {str(exc)[:100]}",
            symbol=symbol, retryable=True,
        )

    candles = candles[-lookback:]
    if len(candles) < 50:
        return make_error(
            ErrorCode.NO_DATA,
            f"Only {len(candles)} candles available — too few for seasonality",
            symbol=symbol, candle_source=source,
        )

    from datetime import datetime

    by_weekday: Dict[str, List[float]] = {d: [] for d in _WEEKDAYS}
    by_hour: Dict[int, List[float]] = {}
    undated = 0

    for prev, cur in zip(candles, candles[1:]):
        prev_close, cur_close = prev.get("close"), cur.get("close")
        if not prev_close or cur_close is None:
            continue
        ret = (cur_close - prev_close) / prev_close
        stamp = cur.get("date") or ""
        try:
            fmt = "%Y-%m-%d %H:%M" if " " in stamp else "%Y-%m-%d"
            when = datetime.strptime(stamp, fmt)
        except ValueError:
            undated += 1
            continue
        by_weekday[_WEEKDAYS[when.weekday()]].append(ret)
        if " " in stamp:
            by_hour.setdefault(when.hour, []).append(ret)

    weekday_stats = {
        day: _bucket_stats(vals) for day, vals in by_weekday.items() if vals
    }
    hour_stats = {
        f"{hour:02d}:00": _bucket_stats(vals)
        for hour, vals in sorted(by_hour.items())
    }

    def _best(stats: Dict[str, dict], reverse: bool) -> Optional[dict]:
        usable = [
            (k, v) for k, v in stats.items()
            if not v["thin_sample"] and v["mean_return_pct"] is not None
        ]
        if not usable:
            return None
        key, value = sorted(usable, key=lambda kv: kv[1]["mean_return_pct"],
                            reverse=reverse)[0]
        return {"bucket": key, **value}

    from tradingview_mcp.core.utils.validators import crypto_pair

    return {
        "symbol": crypto_pair(symbol, "BINANCE"),
        "timeframe": timeframe,
        "candle_source": source,
        "bars_analysed": len(candles),
        "undated_bars_skipped": undated or None,
        "by_weekday": weekday_stats,
        "by_hour_utc": hour_stats or None,
        "hour_note": (
            None if hour_stats else
            "hourly buckets need an intraday timeframe — a daily candle has no hour"
        ),
        "best_weekday": _best(weekday_stats, reverse=True),
        "worst_weekday": _best(weekday_stats, reverse=False),
        "best_hour_utc": _best(hour_stats, reverse=True),
        "worst_hour_utc": _best(hour_stats, reverse=False),
        "sample_note": (
            f"Buckets with fewer than {_MIN_BUCKET_SAMPLES} observations are "
            "flagged thin_sample and excluded from best/worst. Check t_stat "
            "before acting on any bucket: with returns this noisy a mean of a "
            "tenth of a percent is usually indistinguishable from zero, and "
            "distinguishable_from_zero says which buckets clear that bar."
        ),
        "buckets_distinguishable_from_zero": sorted(
            [k for k, v in {**weekday_stats, **hour_stats}.items()
             if v.get("distinguishable_from_zero")]
        ) or None,
        "disclaimer": "Past patterns in a finite sample are not a forecast. For "
                      "educational/informational purposes only. Not financial advice.",
    }
