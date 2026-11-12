"""Rounding that does not erase a small price.

Prices here were rounded to a fixed four or six decimals, which is right for
bitcoin and wrong for anything priced far below a cent. Measured on PEPE and
SHIB: coin_analysis returned 23 price fields as 0.0 and the price itself as
0.000005, one digit of a number that had six; every level in a Fibonacci
retracement and every timeframe's price in multi_timeframe_analysis was 0.0;
the wallet showed a liquidation price of 0.0.

``keep_digits`` is ``round`` with one difference. When a caller asks for
price-like precision (four decimals or more) and that would leave fewer than
three significant digits, it keeps five significant digits instead. Every
number that fixed rounding already handled comes out exactly as before, and
so does every call that rounds to fewer than four places: a percentage
rounded to two decimals is meant to be coarse.

The price-handling modules import it under the name ``round``, so the rule
holds at every site in them, including ones added later.
"""
from __future__ import annotations

import builtins
import math

_PRICE_PRECISION = 4      # decimals at or above which a call is rounding a price
_MIN_SIGNIFICANT = 3      # fewer than this survive -> fixed rounding is lossy
_KEEP_SIGNIFICANT = 5     # what is kept instead


def keep_digits(value, ndigits=None):
    """``round(value, ndigits)``, except a tiny price keeps its digits."""
    if ndigits is None:
        return builtins.round(value)
    if ndigits < _PRICE_PRECISION or isinstance(value, bool):
        return builtins.round(value, ndigits)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return builtins.round(value, ndigits)
    if number == 0 or not math.isfinite(number):
        return builtins.round(value, ndigits)
    magnitude = math.floor(math.log10(abs(number)))    # 0.0000049 -> -6
    if ndigits + magnitude + 1 >= _MIN_SIGNIFICANT:
        return builtins.round(value, ndigits)
    return builtins.round(number, _KEEP_SIGNIFICANT - 1 - magnitude)


def round_price(value, decimals: int = 2):
    """A price to ``decimals`` places, or to five significant digits if finer.

    The trade-setup and Fibonacci code rounded every level to two decimals,
    which suits a share price and nothing below a dollar: a level of 0.0934
    became 0.09, further from the truth than a typical stop is from its
    entry, and anything under half a cent became 0.0 - after which the trade
    plan gave up and reported that ATR was unavailable. A price at or above
    one unit is returned exactly as two-decimal rounding always gave it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0 or not math.isfinite(number):
        return builtins.round(number, decimals)
    magnitude = math.floor(math.log10(abs(number)))
    if decimals + magnitude + 1 >= _MIN_SIGNIFICANT:
        return builtins.round(number, decimals)
    return builtins.round(number, _KEEP_SIGNIFICANT - 1 - magnitude)
