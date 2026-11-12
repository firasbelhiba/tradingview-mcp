"""A price far below a dollar keeps its digits. Offline.

Measured on PEPE and SHIB before this: coin_analysis returned 23 price
fields as 0.0 and the price itself as 0.000005; every Fibonacci level and
every timeframe's price in multi_timeframe_analysis was 0.0; and
crypto_trade_plan refused with "ATR unavailable" although ATR was there,
because its entry had been rounded to 0.0. The trade-setup code rounded to
two decimals, so even DOGE came out wrong: an entry of 0.065988 became 0.07
and its stop 0.06.
"""
import pytest

from tradingview_mcp.core.services.indicators import (
    compute_fibonacci_levels,
    compute_trade_setup,
)
from tradingview_mcp.core.utils.rounding import keep_digits, round_price


# ── keep_digits: round(), minus the erasure ──────────────────────────────────

@pytest.mark.parametrize("value, ndigits", [
    (73849.512345, 4), (2059.19, 6), (83.5123456, 6), (0.5123, 4), (0.0312345, 4),
    (1.0, 4), (-420.69, 4), (0, 4), (0.0, 6), (12345, 4),
])
def test_every_number_fixed_rounding_handled_is_unchanged(value, ndigits):
    assert keep_digits(value, ndigits) == round(value, ndigits)


@pytest.mark.parametrize("value, ndigits, expected", [
    (0.0000049123, 4, 0.0000049123),      # was 0.0
    (0.0000049123, 6, 0.0000049123),      # was 0.000005
    (0.000012345678, 4, 0.000012346),
    (-0.00000123456, 6, -0.0000012346),
    (0.00312345, 4, 0.0031234),           # two digits survived; now five
])
def test_a_tiny_price_keeps_five_significant_digits(value, ndigits, expected):
    assert keep_digits(value, ndigits) == pytest.approx(expected, rel=1e-9)
    assert keep_digits(value, ndigits) != 0


@pytest.mark.parametrize("value, ndigits", [
    (0.004, 2), (0.5, 0), (0.0449, 1), (0.00004, 3),
])
def test_a_call_that_rounds_coarsely_is_left_coarse(value, ndigits):
    """A change of 0.004% shown to two decimals is 0.0, and should be."""
    assert keep_digits(value, ndigits) == round(value, ndigits)


def test_it_behaves_like_round_with_one_argument_and_odd_inputs():
    assert keep_digits(2.5) == round(2.5)
    assert keep_digits(True, 4) == round(True, 4)
    assert keep_digits(float("inf"), 4) == float("inf")


# ── round_price: two decimals where that is a price, more where it is not ────

@pytest.mark.parametrize("value, expected", [
    (73849.512345, 73849.51), (2059.186, 2059.19), (83.5149, 83.51), (2.3512, 2.35),
    (1.0, 1.0),
])
def test_a_price_of_one_or_more_is_exactly_what_it_always_was(value, expected):
    assert round_price(value) == expected == round(value, 2)


@pytest.mark.parametrize("value, expected", [
    (0.065988, 0.065988),          # DOGE: was 0.07
    (0.5123456, 0.51235),          # was 0.51
    (0.0000043301, 0.0000043301),  # PEPE: was 0.0
])
def test_a_price_below_one_is_not_rounded_into_another_price(value, expected):
    assert round_price(value) == pytest.approx(expected, rel=1e-9)


def test_odd_prices_do_not_raise():
    assert round_price(None) is None
    assert round_price("n/a") is None
    assert round_price(0) == 0


# ── The trade setup ───────────────────────────────────────────────────────────

def _levels(setup):
    return [setup["stop_loss"], *setup["entry_points"].values(), *setup["targets"].values()]


def test_a_coin_under_a_dollar_gets_levels_that_differ_from_each_other():
    """At two decimals a DOGE entry, stop and target were 0.07, 0.06 and
    0.07: an entry equal to its target."""
    setup = compute_trade_setup({
        "close": 0.065988, "ATR": 0.0019, "EMA20": 0.0655, "EMA50": 0.0668,
        "BB.lower": 0.0641, "BB.upper": 0.0684,
    })
    levels = _levels(setup)
    assert all(0.05 < level < 0.08 for level in levels), levels
    assert setup["stop_loss"] < min(setup["entry_points"].values())
    assert round(setup["stop_loss"], 2) != setup["stop_loss"], "more than cents"


def test_a_coin_far_below_a_cent_still_gets_a_plan():
    """It used to return None, reported upstream as 'ATR unavailable'."""
    setup = compute_trade_setup({
        "close": 4.4e-06, "ATR": 1.9e-07, "EMA20": 4.35e-06, "EMA50": 4.5e-06,
        "BB.lower": 4.1e-06, "BB.upper": 4.7e-06,
    })
    assert setup is not None
    assert all(level > 0 for level in _levels(setup))
    assert setup["stop_loss"] < setup["entry_points"]["pullback_entry"] \
        < setup["targets"]["target_1"]


def test_bitcoin_sized_levels_are_still_whole_cents():
    setup = compute_trade_setup({
        "close": 73849.5, "ATR": 836.97, "EMA20": 73500.123456, "EMA50": 74900.0,
        "BB.lower": 72000.0, "BB.upper": 75500.0,
    })
    for level in _levels(setup):
        assert level == round(level, 2), level


# ── Fibonacci ─────────────────────────────────────────────────────────────────

def test_fibonacci_levels_of_a_tiny_price_are_not_all_zero():
    out = compute_fibonacci_levels(5.2e-06, 3.9e-06, "uptrend")
    levels = list(out["retracement_levels"].values())
    assert all(level > 0 for level in levels)
    assert len(set(levels)) == len(levels), "each ratio is its own price"
    assert out["swing_high"] == pytest.approx(5.2e-06)


def test_fibonacci_levels_of_a_large_price_are_unchanged():
    out = compute_fibonacci_levels(80000.0, 60000.0, "uptrend")
    for level in out["retracement_levels"].values():
        assert level == round(level, 2)
