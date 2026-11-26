"""
engine/market.py
================
Optional blend of the model with the sportsbook line (ESPN pickcenter / DraftKings).

    final = (1 - w) * model + w * market         for margin and total
    P(home win) = Phi(final_margin / sigma)

The model is computed first and independently; the market only nudges the final
number. The pure model outputs are kept alongside (model_* fields), so disagreement
with the market stays visible.

Why a small fixed weight: on the 2026-27 season (model out-of-sample, 270 games) every
metric improved monotonically with w (margin MAE 11.01 -> 10.93 at w=0.25 -> 10.83
market only; log-loss 0.5815 -> 0.5789 -> 0.5777). The market is the better single
forecast; w = 0.25 is the owner's choice of "some, but limited" market influence.
models/train.py re-reports these numbers on every training run.
"""

from __future__ import annotations

MARKET_WEIGHT = 0.25


def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def no_vig_home_prob(ml_home: float | None, ml_away: float | None) -> float | None:
    """Market-implied home win probability with the bookmaker margin removed."""
    if ml_home is None or ml_away is None:
        return None
    ph, pa = 1 / american_to_decimal(ml_home), 1 / american_to_decimal(ml_away)
    return ph / (ph + pa)


def blend(model_value: float, market_value: float | None, w: float = MARKET_WEIGHT) -> float:
    if market_value is None:
        return model_value
    return (1 - w) * model_value + w * market_value
