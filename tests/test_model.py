"""Invariant tests. All parameter values here are arbitrary test inputs, not any product's terms."""

import numpy as np
import pytest

from proof_engine import (
    BorrowerBehavior,
    DownOnlyLoan,
    FixedFloor,
    FloorBelowStart,
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
    select_behavior,
    simulate,
)

MARKET = HullWhiteMarket(flat_rate=3.7, mean_reversion=0.11, volatility=1.3, index_tenor_years=10)
CALM = HullWhiteMarket(flat_rate=3.7, mean_reversion=0.11, volatility=0.0, index_tenor_years=10)
DAYS = 3
PATHS = MARKET.simulate(months=120, days_per_month=DAYS, paths=300, seed=21, antithetic=True)
FLAT = CALM.simulate(months=120, days_per_month=DAYS, paths=4, seed=21, antithetic=True)
DOWN = DownOnlyLoan(rate_spread=1.6, triggers=(0.35, 0.6), check_every_days=1, floor=FixedFloor(2.2))
NEVER = StandardLoan(rate_spread=1.6, refinance=RefinancePolicy.never())
PROMPT = StandardLoan(rate_spread=1.6, refinance=RefinancePolicy(0.6, 0, 1.0, 1.0, 0.025))
PREPAY = PrepaymentModel(turnover_cpr=0.08, refi_max_cpr=0.55, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9)
NO_PREPAY = PrepaymentModel(turnover_cpr=0.0, refi_max_cpr=0.0, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9)
BORROWER = dict(balance=250_000, term_years=10, horizon_years=10, discount_rate=3.1, annual_turnover=0.0, seed=5)

# Contrasting example designs (arbitrary values).
DESIGNS = {
    "one tier, no floor": DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1),
    "two tiers, fixed floor": DOWN,
    "three tiers, reset limit": DownOnlyLoan(1.6, triggers=(0.3, 0.45, 0.75), check_every_days=1, max_resets=4),
    "prepayment penalty, floor below start": DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1,
                                                          floor=FloorBelowStart(1.8), prepayment_penalty=(3.0, 2.0, 1.0)),
    "averaged index, notice lag, spacing": DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=5, index_average_days=20,
                                                        notice_lag_days=10, min_days_between_resets=60),
}


def manual_paths(index_by_day, days=DAYS, short_rate=3.7, months=None):
    """Paths from a daily index sequence, padded with its last value to whole months."""
    idx = list(index_by_day)
    cols = (months * days + 1) if months else ((len(idx) - 1 + days - 1) // days) * days + 1
    idx = (idx + [idx[-1]] * cols)[:cols]
    arr = np.asarray(idx, dtype=float)[None, :]
    return MarketPaths(short_rate=np.full_like(arr, short_rate), index=arr, days_per_month=days)


def run(p, loan, months, standard=NEVER, **kw):
    args = dict(balance=1.0, term_years=10, horizon_years=months / 12, down_only=loan, standard=standard,
                discount_rate=3.1, annual_turnover=0.0, seed=1, record_rates=True)
    args.update(kw)
    return simulate(p, **args)


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


def test_named_series_from_other_tenors():
    p = MARKET.simulate(months=12, days_per_month=DAYS, paths=6, seed=3, antithetic=True, series_tenors={"2y": 2})
    assert p.series["2y"].shape == p.index.shape
    assert not np.allclose(p.series["2y"], p.index)
    loan = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, index_series="2y")
    r = simulate(p, balance=1.0, term_years=1, horizon_years=1, down_only=loan, standard=NEVER, discount_rate=3.1,
                 annual_turnover=0.0, seed=1, record_rates=True)
    np.testing.assert_allclose(r.rates_down_only[:, 0], p.series["2y"][:, 0] + 1.6)
    with pytest.raises(ValueError):
        simulate(p, balance=1.0, term_years=1, horizon_years=1, standard=NEVER, discount_rate=3.1, annual_turnover=0.0,
                 seed=1, down_only=DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, index_series="missing"))


# ---- Down-only reset rule ---------------------------------------------------------------------------


def test_rate_never_rises_and_respects_floor():
    r = simulate(PATHS, down_only=DOWN, standard=NEVER, record_rates=True, **BORROWER).rates_down_only
    assert np.all(np.diff(r, axis=1) <= 1e-12)
    assert np.all(r >= DOWN.floor.level - 1e-12)


def test_two_tier_steps():
    # Index starts at 3.0 (rate 4.6). Falls 0.4: >= first tier 0.35, resets. Falls a further 0.5:
    # < second tier 0.6, no reset. Falls a further 0.22 (0.72 total since the reset): resets.
    days = [3.0] * 4 + [2.6] * 4 + [2.1] * 4 + [1.88] * 4 + [1.88] * 9
    loan = DownOnlyLoan(1.6, triggers=(0.35, 0.6), check_every_days=1)
    r = run(manual_paths(days), loan, 8)
    assert r.resets[0] == 2
    np.testing.assert_allclose(r.rates_down_only[0, -1], 1.88 + 1.6)


def test_multi_tier_triggers_and_last_tier_repeats():
    # Tiers 0.2, 0.4, 0.6. Each index level is held for a month.
    levels = [3.0, 2.75, 2.45, 2.30, 1.80, 1.65, 1.0, 0.35]
    #          r1 (0.25) no (0.30<0.4) r2 (0.45) no (0.5<0.6) r3 (0.65) r4 (0.65, tier 3 again) r5
    days = [x for x in levels for _ in range(DAYS)]
    loan = DownOnlyLoan(1.6, triggers=(0.2, 0.4, 0.6), check_every_days=1)
    r = run(manual_paths(days, months=len(levels)), loan, len(levels))
    assert r.resets[0] == 5
    np.testing.assert_allclose(r.rates_down_only[0, -1], 0.35 + 1.6)
    # A single 0.6 tier resets only at 2.30, 1.65, 1.0 and 0.35.
    single = run(manual_paths(days, months=len(levels)), DownOnlyLoan(1.6, triggers=(0.6,), check_every_days=1), len(levels))
    assert single.resets[0] == 4


def test_floor_within_one_step_blocks_the_reset():
    # Rate 4.6, floor 4.4. Index collapses, but max(candidate, floor) = 4.4 is not a full step below 4.6.
    p = manual_paths([3.0] * 3 + [0.5] * 22)
    loan = DownOnlyLoan(1.6, triggers=(0.35, 0.6), check_every_days=1, floor=FixedFloor(4.4))
    assert run(p, loan, 8).resets[0] == 0


@pytest.mark.parametrize(
    "floor, expected",
    [(None, 0.5 + 1.6), (FixedFloor(3.2), 3.2), (FloorBelowStart(1.2), 4.6 - 1.2)],
)
def test_optional_floor(floor, expected):
    # Index collapses from 3.0 to 0.5: no floor follows it all the way; each floor stops the reset there.
    p = manual_paths([3.0] * 3 + [0.5] * 22)
    loan = DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, floor=floor)
    r = run(p, loan, 8)
    np.testing.assert_allclose(r.rates_down_only[0, -1], expected)
    assert r.resets[0] == 1


def test_fixed_floor_above_starting_rate_is_rejected():
    with pytest.raises(ValueError):
        simulate(PATHS, down_only=DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, floor=FixedFloor(9.0)),
                 standard=NEVER, **BORROWER)


def test_max_resets_caps_the_number_of_resets():
    days = [3.0 - 0.5 * k for k in range(6) for _ in range(DAYS)]  # five 0.5 drops, one a month
    p = manual_paths(days, months=6)
    unlimited = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1), 6)
    limited = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1, max_resets=2), 6)
    assert unlimited.resets[0] == 5 and limited.resets[0] == 2
    np.testing.assert_allclose(limited.rates_down_only[0, -1], 2.0 + 1.6)


def test_min_days_between_resets():
    # Drops on day 3 and day 6. With 9 days required between resets, the second can't happen before day 12.
    days = [3.0] * 3 + [2.5] * 3 + [2.0] * 30
    p = manual_paths(days, months=12)
    free = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1), 3)
    spaced = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1, min_days_between_resets=9), 3)
    later = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1, min_days_between_resets=9), 5)
    assert free.resets[0] == 2 and spaced.resets[0] == 1 and later.resets[0] == 2


def test_notice_lag_delays_the_effective_date():
    # Drop on day 3 (month 2). With a 4-day lag the new rate takes effect on day 7 (month 3).
    days = [3.0] * 3 + [2.5] * 30
    p = manual_paths(days, months=4)
    now = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1), 4)
    lag = run(p, DownOnlyLoan(1.6, triggers=(0.4,), check_every_days=1, notice_lag_days=4), 4)
    np.testing.assert_allclose(now.rates_down_only[0, 2], 4.1)  # after month 2
    np.testing.assert_allclose(lag.rates_down_only[0, 2], 4.6)
    np.testing.assert_allclose(lag.rates_down_only[0, 3], 4.1)  # after month 3
    assert now.resets[0] == lag.resets[0] == 1


def test_index_averaging_smooths_a_brief_drop():
    # A one-day dip of 1.0 is a reset on the daily index, but only 0.25 on a 4-day average (< 0.35 tier).
    days = [3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 3.0]
    p = manual_paths(days, months=3)
    daily = run(p, DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1), 3)
    averaged = run(p, DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, index_average_days=4), 3)
    assert daily.resets[0] == 1 and averaged.resets[0] == 0


def test_check_frequency_can_miss_a_brief_drop():
    # A one-day dip between checks is missed when checking every 4 days, caught when checking daily.
    p = manual_paths([3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    daily = run(p, DownOnlyLoan(1.6, triggers=(0.35, 0.6), check_every_days=1), 2)
    sparse = run(p, DownOnlyLoan(1.6, triggers=(0.35, 0.6), check_every_days=4), 2)
    assert daily.resets[0] == 1 and sparse.resets[0] == 0


# ---- Prepayment penalty -----------------------------------------------------------------------------


def test_penalty_schedule_by_loan_year():
    loan = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, prepayment_penalty=(3.0, 2.0))
    assert loan.penalty_fraction(0) == loan.penalty_fraction(11) == pytest.approx(0.03)
    assert loan.penalty_fraction(12) == pytest.approx(0.02)
    assert loan.penalty_fraction(24) == 0.0
    assert DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1).penalty_fraction(0) == 0.0


def test_penalty_raises_the_pricing_value_only_with_prepayments():
    base = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1)
    pen = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, prepayment_penalty=(3.0, 2.0, 1.0))
    zero = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, prepayment_penalty=(0.0,))
    kw = dict(term_months=120, coupon_strip=0.55, oas=0.4)
    v = lambda loan, cpr: price_down_only(PATHS, loan=loan, turnover_cpr=cpr, **kw)  # noqa: E731
    assert v(pen, 0.08) > v(base, 0.08)
    assert v(pen, 0.0) == pytest.approx(v(base, 0.0))
    assert v(zero, 0.08) == pytest.approx(v(base, 0.08))


def test_penalty_charged_on_early_sales_in_the_borrower_lens():
    base = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1)
    pen = DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, prepayment_penalty=(3.0, 2.0))
    kw = {**BORROWER, "annual_turnover": 0.3}
    a = simulate(PATHS, down_only=base, standard=NEVER, **kw)
    b = simulate(PATHS, down_only=pen, standard=NEVER, **kw)
    extra = b.cost_down_only - a.cost_down_only
    assert np.all(extra >= -1e-9) and extra.max() > 0
    np.testing.assert_allclose(b.cost_standard, a.cost_standard)  # the standard loan has no penalty
    no_sales = simulate(PATHS, down_only=pen, standard=NEVER, **BORROWER)
    np.testing.assert_allclose(no_sales.cost_down_only, simulate(PATHS, down_only=base, standard=NEVER, **BORROWER).cost_down_only)


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


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_fair_cost_solvers_work_for_every_design(name):
    loan = DESIGNS[name]
    kw = dict(term_months=120, coupon_strip=0.55, oas=0.4, search=(-2, 6))
    down = par_spread_down_only(PATHS, loan=loan, turnover_cpr=0.08, **kw)
    assert price_down_only(PATHS, loan=loan.with_spread(down), term_months=120, coupon_strip=0.55, oas=0.4,
                           turnover_cpr=0.08) == pytest.approx(1.0, abs=1e-3)
    fair = fair_rate_spread(PATHS, down_only=loan, standard=NEVER, search=(-2, 6), **BORROWER)
    assert fair > NEVER.rate_spread  # the feature is worth something against a borrower who never refinances


def test_more_generous_designs_cost_more():
    kw = dict(term_months=120, coupon_strip=0.55, oas=0.4, turnover_cpr=0.08, search=(-2, 6))
    unfloored = par_spread_down_only(PATHS, loan=DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1), **kw)
    limited = par_spread_down_only(PATHS, loan=DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1, max_resets=1), **kw)
    floored = par_spread_down_only(PATHS, loan=DownOnlyLoan(1.6, triggers=(0.55,), check_every_days=1,
                                                            floor=FloorBelowStart(0.5)), **kw)
    assert unfloored > limited and unfloored > floored


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


def test_fixed_refinance_costs_add_to_the_standard_loan():
    pct = StandardLoan(1.6, RefinancePolicy(0.6, 0, 1.0, 1.0, 0.01))
    both = StandardLoan(1.6, RefinancePolicy(0.6, 0, 1.0, 1.0, 0.01, fixed_costs=1500.0))
    a = simulate(PATHS, down_only=DOWN, standard=pct, **BORROWER)
    b = simulate(PATHS, down_only=DOWN, standard=both, **BORROWER)
    np.testing.assert_array_equal(a.refinances, b.refinances)
    assert b.cost_standard.mean() > a.cost_standard.mean()
    np.testing.assert_allclose(b.cost_standard[a.refinances == 0], a.cost_standard[a.refinances == 0])


def test_seed_makes_results_reproducible():
    a = simulate(PATHS, down_only=DOWN, standard=PROMPT, **BORROWER).summary()
    b = simulate(PATHS, down_only=DOWN, standard=PROMPT, **BORROWER).summary()
    assert a == b


def test_fair_spread_is_above_the_standard_spread_against_a_never_refinancer():
    s = fair_rate_spread(PATHS, down_only=DOWN, standard=NEVER, search=(-2, 6), **BORROWER)
    assert s > NEVER.rate_spread


# ---- Borrower type ----------------------------------------------------------------------------------


def test_borrower_type_only_selects_the_supplied_inputs():
    owner = BorrowerBehavior("owner_occupant", PREPAY, RefinancePolicy(0.6, 0, 1.0, 1.0, 0.025), 0.08)
    investor = BorrowerBehavior("investor", NO_PREPAY, RefinancePolicy(0.4, 1, 0.5, 0.9, 0.02), 0.12)
    assert select_behavior([owner, investor], "owner_occupant") is owner
    assert select_behavior([owner, investor], "investor") is investor
    with pytest.raises(ValueError):
        select_behavior([owner], "investor")  # no built-in behavior for a type you didn't supply
    with pytest.raises(ValueError):
        select_behavior([owner, investor], "second_home")
    with pytest.raises(ValueError):
        BorrowerBehavior("landlord", PREPAY, RefinancePolicy.never(), 0.1)


# ---- Validation -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HullWhiteMarket(3.7, 0.0, 1.3, 10),
        lambda: HullWhiteMarket(3.7, 0.11, -1.0, 10),
        lambda: FixedFloor(-0.25),
        lambda: FloorBelowStart(0.0),
        lambda: DownOnlyLoan(1.6, triggers=(), check_every_days=1),
        lambda: DownOnlyLoan(1.6, triggers=(0.35, 0.0), check_every_days=1),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=0),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, floor=2.2),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, max_resets=0),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, min_days_between_resets=0),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, notice_lag_days=-1),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, index_average_days=0),
        lambda: DownOnlyLoan(1.6, triggers=(0.35,), check_every_days=1, prepayment_penalty=(-1.0,)),
        lambda: PrepaymentModel(0.08, 1.2, 0.9, 0.28, 0.9),
        lambda: PrepaymentModel(0.08, 0.55, 0.9, 0.0, 0.9),
        lambda: RefinancePolicy(0.6, 0, 1.5, 1.0, 0.0),
        lambda: RefinancePolicy(0.6, 0, 1.0, 1.0, 0.0, fixed_costs=-1.0),
        lambda: MARKET.simulate(months=12, days_per_month=3, paths=5, seed=1, antithetic=True),
    ],
)
def test_invalid_inputs_raise(factory):
    with pytest.raises(ValueError):
        factory()
