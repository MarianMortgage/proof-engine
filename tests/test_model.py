"""Invariant tests. All parameter values here are arbitrary test inputs, not any product's terms."""

import numpy as np
import pytest

from proof_engine import (
    DownOnlyLoan,
    HullWhiteMarket,
    MarketPaths,
    PrepaymentModel,
    RefinancePolicy,
    StandardLoan,
    fair_rate_spread,
    par_spread_down_only,
    par_spread_standard,
    price_down_only,
    price_standard,
    simulate,
)

MARKET = HullWhiteMarket(flat_rate=3.7, mean_reversion=0.11, volatility=1.3, index_tenor_years=10)
CALM = HullWhiteMarket(flat_rate=3.7, mean_reversion=0.11, volatility=0.0, index_tenor_years=10)
DAYS = 3
PATHS = MARKET.simulate(months=120, days_per_month=DAYS, paths=300, seed=21, antithetic=True)
FLAT = CALM.simulate(months=120, days_per_month=DAYS, paths=4, seed=21, antithetic=True)
DOWN = DownOnlyLoan(rate_spread=1.6, floor=2.2, first_step=0.35, subsequent_step=0.6, check_every_days=1)
NEVER = StandardLoan(rate_spread=1.6, refinance=RefinancePolicy.never())
PROMPT = StandardLoan(rate_spread=1.6, refinance=RefinancePolicy(0.6, 0, 1.0, 1.0, 0.025))
PREPAY = PrepaymentModel(turnover_cpr=0.08, refi_max_cpr=0.55, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9)
NO_PREPAY = PrepaymentModel(turnover_cpr=0.0, refi_max_cpr=0.0, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9)
BORROWER = dict(balance=250_000, term_years=10, horizon_years=10, discount_rate=3.1, annual_turnover=0.0, seed=5)


def manual_paths(index_by_day, days=DAYS, short_rate=3.7):
    idx = np.asarray(index_by_day, dtype=float)[None, :]
    return MarketPaths(short_rate=np.full_like(idx, short_rate), index=idx, days_per_month=days)


# ---- Rates ------------------------------------------------------------------------------------------


def test_zero_volatility_reproduces_the_flat_curve():
    np.testing.assert_allclose(FLAT.short_rate, 3.7, atol=1e-9)
    np.testing.assert_allclose(FLAT.index, 3.7, atol=1e-9)


def test_antithetic_paths_mirror_each_other():
    half = PATHS.paths // 2
    drift = PATHS.short_rate[:half] + PATHS.short_rate[half:]
    np.testing.assert_allclose(drift - drift[:1], 0.0, atol=1e-9)  # x + (-x) leaves 2 * alpha(t) on every pair


def test_grid_shape_matches_days_per_month():
    assert PATHS.short_rate.shape == (300, 120 * DAYS + 1)
    assert PATHS.months == 120


# ---- Down-only reset rule ---------------------------------------------------------------------------


def test_rate_never_rises_and_respects_floor():
    r = simulate(PATHS, down_only=DOWN, standard=NEVER, record_rates=True, **BORROWER).rates_down_only
    assert np.all(np.diff(r, axis=1) <= 1e-12)
    assert np.all(r >= DOWN.floor - 1e-12)


def test_two_tier_steps():
    # Index starts at 3.0 (rate 4.6). Falls 0.4: >= first_step 0.35, resets. Falls a further 0.5:
    # < subsequent_step 0.6, no reset. Falls a further 0.22 (0.72 total since the reset): resets.
    days = [3.0] * 4 + [2.6] * 4 + [2.1] * 4 + [1.88] * 4 + [1.88] * 9
    p = manual_paths(days)
    loan = DownOnlyLoan(1.6, 0.0, 0.35, 0.6, 1)
    r = simulate(p, balance=1.0, term_years=10, horizon_years=8 / 12, down_only=loan, standard=NEVER,
                 discount_rate=3.1, annual_turnover=0.0, seed=1, record_rates=True)
    assert r.resets[0] == 2
    np.testing.assert_allclose(r.rates_down_only[0, -1], 1.88 + 1.6)


def test_floor_within_one_step_blocks_the_reset():
    # Rate 4.6, floor 4.4. Index collapses, but max(candidate, floor) = 4.4 is not a full step below 4.6.
    p = manual_paths([3.0] * 3 + [0.5] * 22)
    loan = DownOnlyLoan(1.6, 4.4, 0.35, 0.6, 1)
    r = simulate(p, balance=1.0, term_years=10, horizon_years=8 / 12, down_only=loan, standard=NEVER,
                 discount_rate=3.1, annual_turnover=0.0, seed=1)
    assert r.resets[0] == 0


def test_check_frequency_can_miss_a_brief_drop():
    # A one-day dip between checks is missed when checking every 3 days, caught when checking daily.
    days = [3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    p = manual_paths(days)
    kw = dict(balance=1.0, term_years=10, horizon_years=2 / 12, standard=NEVER, discount_rate=3.1,
              annual_turnover=0.0, seed=1)
    daily = simulate(p, down_only=DownOnlyLoan(1.6, 0.0, 0.35, 0.6, 1), **kw)
    sparse = simulate(p, down_only=DownOnlyLoan(1.6, 0.0, 0.35, 0.6, 3), **kw)
    assert daily.resets[0] == 1 and sparse.resets[0] == 0


# ---- Pricing lens -----------------------------------------------------------------------------------


def test_prepayment_curve_is_bounded_and_increasing():
    inc = np.linspace(-2, 4, 50)
    c = PREPAY.cpr(inc)
    assert np.all(np.diff(c) >= 0)
    assert c.min() >= PREPAY.turnover_cpr - 1e-12 and c.max() <= PREPAY.max_total_cpr + 1e-12


def test_calm_market_prices_near_par_at_zero_spread():
    v = price_standard(FLAT, rate_spread=0.0, term_months=120, coupon_strip=0.0, oas=0.0, prepayment=NO_PREPAY)
    assert v == pytest.approx(1.0, abs=2e-3)
    s = par_spread_standard(FLAT, term_months=120, coupon_strip=0.0, oas=0.0, prepayment=NO_PREPAY, search=(-2, 3))
    assert s == pytest.approx(0.0, abs=0.025)


def test_strip_and_oas_raise_the_par_spread():
    base = par_spread_standard(PATHS, term_months=120, coupon_strip=0.0, oas=0.0, prepayment=PREPAY, search=(-2, 6))
    more = par_spread_standard(PATHS, term_months=120, coupon_strip=0.55, oas=0.4, prepayment=PREPAY, search=(-2, 6))
    assert more > base


def test_down_only_premium_is_zero_in_a_calm_market_and_positive_when_volatile():
    kw = dict(term_months=120, coupon_strip=0.55, oas=0.4, search=(-2, 6))
    calm_std = par_spread_standard(FLAT, prepayment=PrepaymentModel(0.08, 0.0, 0.9, 0.28, 0.9), **kw)
    calm_down = par_spread_down_only(FLAT, loan=DOWN, turnover_cpr=0.08, **kw)
    assert calm_down - calm_std == pytest.approx(0.0, abs=0.025)
    vol_std = par_spread_standard(PATHS, prepayment=PrepaymentModel(0.08, 0.0, 0.9, 0.28, 0.9), **kw)
    vol_down = par_spread_down_only(PATHS, loan=DOWN, turnover_cpr=0.08, **kw)
    assert vol_down > vol_std


def test_price_falls_as_spread_falls():
    hi = price_down_only(PATHS, loan=DOWN.with_spread(2.0), term_months=120, coupon_strip=0.55, oas=0.4, turnover_cpr=0.08)
    lo = price_down_only(PATHS, loan=DOWN.with_spread(1.0), term_months=120, coupon_strip=0.55, oas=0.4, turnover_cpr=0.08)
    assert hi > lo


# ---- Borrower-cost lens -----------------------------------------------------------------------------


def test_never_refinance_policy_never_refinances():
    assert simulate(PATHS, down_only=DOWN, standard=NEVER, **BORROWER).refinances.sum() == 0


def test_calm_market_same_spread_means_no_difference():
    r = simulate(FLAT, down_only=DOWN, standard=NEVER, **BORROWER)
    assert r.resets.sum() == 0
    np.testing.assert_allclose(r.savings, 0.0, atol=1e-6)


def test_down_only_never_costs_more_at_the_same_spread():
    r = simulate(PATHS, down_only=DOWN, standard=NEVER, **BORROWER)
    assert np.all(r.savings >= -1e-6) and r.savings.mean() > 0


def test_failing_to_requalify_blocks_refinancing():
    stuck = StandardLoan(1.6, RefinancePolicy(0.1, 0, 1.0, 0.0, 0.015))
    assert simulate(PATHS, down_only=DOWN, standard=stuck, **BORROWER).refinances.sum() == 0


def test_seed_makes_results_reproducible():
    a = simulate(PATHS, down_only=DOWN, standard=PROMPT, **BORROWER).summary()
    b = simulate(PATHS, down_only=DOWN, standard=PROMPT, **BORROWER).summary()
    assert a == b


def test_fair_spread_is_above_the_standard_spread_against_a_never_refinancer():
    s = fair_rate_spread(PATHS, down_only=DOWN, standard=NEVER, search=(-2, 6), **BORROWER)
    assert s > NEVER.rate_spread


# ---- Validation -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HullWhiteMarket(3.7, 0.0, 1.3, 10),
        lambda: HullWhiteMarket(3.7, 0.11, -1.0, 10),
        lambda: DownOnlyLoan(1.6, -1.0, 0.35, 0.6, 1),
        lambda: DownOnlyLoan(1.6, 2.2, 0.0, 0.6, 1),
        lambda: DownOnlyLoan(1.6, 2.2, 0.35, 0.6, 0),
        lambda: PrepaymentModel(0.08, 1.2, 0.9, 0.28, 0.9),
        lambda: PrepaymentModel(0.08, 0.55, 0.9, 0.0, 0.9),
        lambda: RefinancePolicy(0.6, 0, 1.5, 1.0, 0.0),
        lambda: MARKET.simulate(months=12, days_per_month=3, paths=5, seed=1, antithetic=True),
    ],
)
def test_invalid_inputs_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_floor_above_starting_rate_is_rejected():
    with pytest.raises(ValueError):
        simulate(PATHS, down_only=DownOnlyLoan(1.6, 9.0, 0.35, 0.6, 1), standard=NEVER, **BORROWER)
