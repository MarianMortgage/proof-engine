"""Shared inputs for the example scripts. Illustrative, not Marian terms.

Every number here is an arbitrary example input chosen to show how the package works. None is any
lender's rate, floor, trigger, spread, fee or behavior assumption.
"""

from proof_engine import DownOnlyLoan, FixedFloor, FloorBelowStart, HullWhiteMarket, PrepaymentModel

LABEL = "Illustrative, not Marian terms."

MARKET = HullWhiteMarket(flat_rate=3.7, mean_reversion=0.11, volatility=1.3, index_tenor_years=10)

DESIGNS = {
    "One tier, no floor": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.55,), check_every_days=1),
    "Two tiers, fixed floor": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.35, 0.6), check_every_days=1, floor=FixedFloor(2.2)),
    "Three tiers, reset limit": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.3, 0.45, 0.75), check_every_days=1, max_resets=4),
    "Prepayment penalty, floor below start": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.55,), check_every_days=1, floor=FloorBelowStart(1.8),
        prepayment_penalty=(3.0, 2.0, 1.0)),
}

PREPAYMENT = PrepaymentModel(turnover_cpr=0.08, refi_max_cpr=0.55, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9)
PRICING = dict(term_months=360, coupon_strip=0.55, oas=0.4, search=(-2.0, 6.0))


def paths(n: int = 500):
    """Simulated rate paths: 30 years on a daily grid (21 steps per month)."""
    return MARKET.simulate(months=360, days_per_month=21, paths=n, seed=21, antithetic=True)
