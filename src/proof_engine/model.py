"""Monte Carlo comparison of a down-only mortgage with a standard fixed-rate mortgage.

Every product, market and behavioral parameter is a required input. Nothing here encodes any
lender's pricing.

Conventions
-----------
* Rates, spreads, steps and volatilities are in percentage points (``5.0`` means 5%).
* Prepayment speeds (CPR) are annual fractions (``0.1`` means 10% a year).
* Rates are simulated on a daily grid (``days_per_month`` steps per month); loan cash flows are monthly.

Two lenses
----------
* **Pricing** (``price_down_only``, ``price_standard``, ``par_spread_*``): the value, per 1 of par, of a
  loan's cash flows to whoever holds it, net of a coupon strip, discounted along each simulated
  short-rate path plus an option-adjusted spread. The standard loan prepays through an S-curve
  refinancing model plus turnover; the down-only loan prepays through turnover only. The difference
  between the two loans' par spreads is the cost of the down-only feature.
* **Borrower cost** (``simulate``, ``fair_rate_spread``): the present value of what one borrower pays
  under each loan on each path, with an explicit refinancing decision rule and closing costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "HullWhiteMarket",
    "MarketPaths",
    "DownOnlyLoan",
    "PrepaymentModel",
    "RefinancePolicy",
    "StandardLoan",
    "SimulationResult",
    "price_down_only",
    "price_standard",
    "par_spread_down_only",
    "par_spread_standard",
    "simulate",
    "fair_rate_spread",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _smm(cpr):
    """Annual prepayment rate (CPR) to a single-month rate (SMM)."""
    return 1 - (1 - cpr) ** (1 / 12)


# --------------------------------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketPaths:
    """Simulated paths on a daily grid, shape ``(paths, months * days_per_month + 1)``, in points.

    ``short_rate`` drives discounting. ``index`` is the long-tenor yield that mortgage rates follow.
    Column ``m * days_per_month`` is the start of month ``m``.
    """

    short_rate: np.ndarray
    index: np.ndarray
    days_per_month: int

    def __post_init__(self) -> None:
        _require(self.short_rate.shape == self.index.shape, "short_rate and index must have the same shape")
        _require(self.days_per_month >= 1, "days_per_month must be >= 1")
        _require(self.short_rate.ndim == 2 and (self.short_rate.shape[1] - 1) % self.days_per_month == 0,
                 "paths must have months * days_per_month + 1 columns")

    @property
    def paths(self) -> int:
        return self.short_rate.shape[0]

    @property
    def months(self) -> int:
        return (self.short_rate.shape[1] - 1) // self.days_per_month

    def month_start(self, m: int) -> int:
        return m * self.days_per_month


@dataclass(frozen=True)
class HullWhiteMarket:
    """One-factor Hull-White short-rate model fitted to a flat initial forward curve.

    ``dr = (theta(t) - mean_reversion * r) dt + volatility dW``, with ``theta`` chosen so the model
    reproduces a flat forward curve at ``flat_rate``. The index is the model-implied zero-coupon yield
    with maturity ``index_tenor_years``, in closed form on each path. Mortgage rates are the index plus
    a spread.
    """

    flat_rate: float
    mean_reversion: float
    volatility: float
    index_tenor_years: float

    def __post_init__(self) -> None:
        _require(self.mean_reversion > 0, "mean_reversion must be > 0")
        _require(self.volatility >= 0, "volatility must be >= 0")
        _require(self.index_tenor_years > 0, "index_tenor_years must be > 0")

    def simulate(self, *, months: int, days_per_month: int, paths: int, seed: int | None, antithetic: bool) -> MarketPaths:
        """Simulate on a grid of ``days_per_month`` steps per month (exact OU discretization)."""
        _require(months > 0 and paths > 0 and days_per_month >= 1, "months, paths and days_per_month must be positive")
        _require(not antithetic or paths % 2 == 0, "antithetic sampling needs an even number of paths")
        a, sig, f = self.mean_reversion, self.volatility / 100, self.flat_rate / 100
        steps = months * days_per_month
        dt = 1 / (12 * days_per_month)
        t = np.arange(steps + 1) * dt

        rng = np.random.default_rng(seed)
        z = rng.standard_normal((paths // 2 if antithetic else paths, steps))
        if antithetic:
            z = np.vstack([z, -z])

        decay = np.exp(-a * dt)
        sd = sig * np.sqrt((1 - decay**2) / (2 * a))
        x = np.zeros((paths, steps + 1))
        for i in range(steps):
            x[:, i + 1] = x[:, i] * decay + sd * z[:, i]
        r = x + f + sig**2 / (2 * a**2) * (1 - np.exp(-a * t)) ** 2

        tau = self.index_tenor_years
        b = (1 - np.exp(-a * tau)) / a
        ln_a = -f * tau + b * f - sig**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * b**2
        y = -(ln_a[None, :] - b * r) / tau
        return MarketPaths(short_rate=r * 100, index=y * 100, days_per_month=days_per_month)


# --------------------------------------------------------------------------------------------------
# Loans and behavior
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DownOnlyLoan:
    """A loan whose rate only moves down, and stops at ``floor``.

    It starts at ``index + rate_spread``. Every ``check_every_days`` grid steps it compares its rate with
    a candidate, ``max(index + rate_spread, floor)``. It resets to the candidate when the candidate is
    at least ``first_step`` below the current rate (for the first reset) or ``subsequent_step`` below
    it (for every later reset). The rate never increases.
    """

    rate_spread: float
    floor: float
    first_step: float
    subsequent_step: float
    check_every_days: int

    def __post_init__(self) -> None:
        _require(self.floor >= 0, "floor must be >= 0")
        _require(self.first_step > 0 and self.subsequent_step > 0, "first_step and subsequent_step must be > 0")
        _require(self.check_every_days >= 1, "check_every_days must be >= 1")

    def with_spread(self, rate_spread: float) -> "DownOnlyLoan":
        return DownOnlyLoan(rate_spread, self.floor, self.first_step, self.subsequent_step, self.check_every_days)


@dataclass(frozen=True)
class PrepaymentModel:
    """Expected prepayment speed of a standard loan, for the pricing lens.

    ``cpr = min(turnover_cpr + refi_max_cpr / (1 + exp(-(incentive - refi_midpoint) / refi_width)), max_total_cpr)``

    ``incentive`` is the loan's rate minus the current rate for a new standard loan, in points.
    """

    turnover_cpr: float
    refi_max_cpr: float
    refi_midpoint: float
    refi_width: float
    max_total_cpr: float

    def __post_init__(self) -> None:
        for name in ("turnover_cpr", "refi_max_cpr", "max_total_cpr"):
            _require(0 <= getattr(self, name) < 1, f"{name} must be in [0, 1)")
        _require(self.refi_width > 0, "refi_width must be > 0")

    def cpr(self, incentive: np.ndarray) -> np.ndarray:
        refi = self.refi_max_cpr / (1 + np.exp(-(incentive - self.refi_midpoint) / self.refi_width))
        return np.minimum(self.turnover_cpr + refi, self.max_total_cpr)


@dataclass(frozen=True)
class RefinancePolicy:
    """One borrower's refinancing decision rule, for the borrower-cost lens (checked monthly).

    A refinance happens when the rate exceeds the new-loan rate by at least ``threshold`` for more than
    ``delay_months`` consecutive months and draws succeed with ``monthly_probability`` (attention) and
    ``requalify_probability`` (qualifying again). It costs ``closing_costs`` times the balance and
    restarts the term. ``RefinancePolicy.never()`` never refinances.
    """

    threshold: float
    delay_months: int
    monthly_probability: float
    requalify_probability: float
    closing_costs: float

    def __post_init__(self) -> None:
        _require(self.threshold >= 0, "threshold must be >= 0")
        _require(self.delay_months >= 0, "delay_months must be >= 0")
        _require(0 <= self.monthly_probability <= 1, "monthly_probability must be in [0, 1]")
        _require(0 <= self.requalify_probability <= 1, "requalify_probability must be in [0, 1]")
        _require(self.closing_costs >= 0, "closing_costs must be >= 0")

    @classmethod
    def never(cls) -> "RefinancePolicy":
        return cls(threshold=0.0, delay_months=0, monthly_probability=0.0, requalify_probability=0.0, closing_costs=0.0)


@dataclass(frozen=True)
class StandardLoan:
    """A standard fixed-rate loan at ``index + rate_spread``, refinanced per ``refinance`` (borrower lens)."""

    rate_spread: float
    refinance: RefinancePolicy


def _payment(balance, annual_rate_pct, months_left):
    r = annual_rate_pct / 1200.0
    n = np.maximum(months_left, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, balance * r / (1 - (1 + r) ** (-n)), balance / n)


class _DownOnlyState:
    """Runs the down-only reset rule across the daily grid for a batch of paths."""

    def __init__(self, loan: DownOnlyLoan, paths: MarketPaths):
        self.loan, self.paths = loan, paths
        self.rate = paths.index[:, 0] + loan.rate_spread
        _require(bool(np.all(self.rate >= loan.floor)), "floor must be at or below the down-only starting rate")
        self.resets = np.zeros(paths.paths, dtype=int)

    def month(self, m: int, active: np.ndarray):
        """Advance through month ``m``; return (rate at month start, average rate over the month, reset mask)."""
        start_rate = self.rate.copy()
        total = np.zeros_like(self.rate)
        changed = np.zeros(len(self.rate), dtype=bool)
        d0 = self.paths.month_start(m)
        for d in range(d0, d0 + self.paths.days_per_month):
            if d > 0 and d % self.loan.check_every_days == 0:
                candidate = np.maximum(self.paths.index[:, d] + self.loan.rate_spread, self.loan.floor)
                step = np.where(self.resets == 0, self.loan.first_step, self.loan.subsequent_step)
                ok = active & (candidate <= self.rate - step)
                self.rate = np.where(ok, candidate, self.rate)
                self.resets += ok
                changed |= ok
            total += self.rate
        return start_rate, total / self.paths.days_per_month, changed


# --------------------------------------------------------------------------------------------------
# Pricing lens
# --------------------------------------------------------------------------------------------------


def _discount_factors(paths: MarketPaths, oas: float, term_months: int) -> np.ndarray:
    """Cumulative monthly discount factors along each path, shape (paths, term_months)."""
    starts = [paths.month_start(i) for i in range(term_months)]
    r = paths.short_rate[:, starts]
    return np.exp(-np.cumsum((r + oas) / 100 / 12, axis=1))


def price_standard(paths: MarketPaths, *, rate_spread: float, term_months: int, coupon_strip: float, oas: float,
                   prepayment: PrepaymentModel) -> float:
    """Value per 1 of par of a standard loan's cash flows, net of ``coupon_strip``, at ``oas``."""
    _require(term_months <= paths.months, "paths must cover term_months")
    df = _discount_factors(paths, oas, term_months)
    rate = paths.index[:, 0] + rate_spread
    bal = np.ones(paths.paths)
    pv = np.zeros(paths.paths)
    for i in range(term_months):
        market = paths.index[:, paths.month_start(i)] + rate_spread
        cpr = prepayment.cpr(rate - market)
        pmt = _payment(bal, rate, term_months - i)
        interest = bal * rate / 1200
        sched = pmt - interest
        prep = (bal - sched) * _smm(cpr)
        pv += (bal * (rate - coupon_strip) / 1200 + sched + prep) * df[:, i]
        bal = bal - sched - prep
    return float(pv.mean())


def price_down_only(paths: MarketPaths, *, loan: DownOnlyLoan, term_months: int, coupon_strip: float, oas: float,
                    turnover_cpr: float) -> float:
    """Value per 1 of par of a down-only loan's cash flows, net of ``coupon_strip``, at ``oas``.

    The loan prepays only through ``turnover_cpr``: a borrower whose rate already drops has no rate
    reason to refinance. Interest accrues at the rate in effect on each day of the month.
    """
    _require(term_months <= paths.months, "paths must cover term_months")
    _require(0 <= turnover_cpr < 1, "turnover_cpr must be in [0, 1)")
    df = _discount_factors(paths, oas, term_months)
    state = _DownOnlyState(loan, paths)
    active = np.ones(paths.paths, dtype=bool)
    bal = np.ones(paths.paths)
    pv = np.zeros(paths.paths)
    smm = _smm(turnover_cpr)
    for i in range(term_months):
        start_rate, avg_rate, _ = state.month(i, active)
        pmt = _payment(bal, start_rate, term_months - i)
        interest = bal * avg_rate / 1200
        sched = np.minimum(pmt - interest, bal)
        prep = (bal - sched) * smm
        pv += (bal * np.maximum(avg_rate - coupon_strip, 0) / 1200 + sched + prep) * df[:, i]
        bal = bal - sched - prep
    return float(pv.mean())


def _bisect_to_par(value: Callable[[float], float], lo: float, hi: float, tol: float) -> float:
    v_lo, v_hi = value(lo), value(hi)
    _require((v_lo - 1) * (v_hi - 1) < 0, "search interval does not bracket par; widen `search`")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        v = value(mid)
        if (v - 1) * (v_lo - 1) > 0:
            lo, v_lo = mid, v
        else:
            hi = mid
    return (lo + hi) / 2


def par_spread_standard(paths: MarketPaths, *, term_months: int, coupon_strip: float, oas: float,
                        prepayment: PrepaymentModel, search: tuple[float, float], tol: float = 1e-4) -> float:
    """The standard loan's spread over the index at which it prices at par.

    The refinancing incentive is measured against ``index + spread``, so the solved spread is also the
    market rate for new standard loans (a self-consistent fixed point).
    """
    return _bisect_to_par(
        lambda s: price_standard(paths, rate_spread=s, term_months=term_months, coupon_strip=coupon_strip, oas=oas,
                                 prepayment=prepayment),
        *search, tol)


def par_spread_down_only(paths: MarketPaths, *, loan: DownOnlyLoan, term_months: int, coupon_strip: float, oas: float,
                         turnover_cpr: float, search: tuple[float, float], tol: float = 1e-4) -> float:
    """The down-only loan's spread over the index at which it prices at par (``loan.rate_spread`` is ignored).

    The loan starts at, and resets to, ``index + spread`` (floored), so the premium over the standard
    par spread is carried into every reset. Premium = this spread minus ``par_spread_standard``.
    """
    lo = max(search[0], float(np.max(loan.floor - paths.index[:, 0])))
    return _bisect_to_par(
        lambda s: price_down_only(paths, loan=loan.with_spread(s), term_months=term_months, coupon_strip=coupon_strip,
                                  oas=oas, turnover_cpr=turnover_cpr),
        lo, search[1], tol)


# --------------------------------------------------------------------------------------------------
# Borrower-cost lens
# --------------------------------------------------------------------------------------------------


@dataclass
class SimulationResult:
    cost_down_only: np.ndarray
    cost_standard: np.ndarray
    resets: np.ndarray
    refinances: np.ndarray
    horizon_months: int
    #: Month-start rate histories, shape (paths, months + 1), when ``simulate(record_rates=True)``.
    rates_down_only: np.ndarray | None = None
    rates_standard: np.ndarray | None = None

    @property
    def paths(self) -> int:
        return len(self.cost_down_only)

    @property
    def savings(self) -> np.ndarray:
        """Per-path ``cost_standard - cost_down_only``. Positive means the down-only loan cost less."""
        return self.cost_standard - self.cost_down_only

    def summary(self) -> dict:
        s = self.savings
        q = np.percentile(s, [5, 25, 50, 75, 95])
        return {
            "paths": self.paths,
            "horizon_months": self.horizon_months,
            "savings_mean": float(s.mean()),
            "savings_std": float(s.std(ddof=1)) if self.paths > 1 else 0.0,
            "savings_p5": float(q[0]),
            "savings_p25": float(q[1]),
            "savings_p50": float(q[2]),
            "savings_p75": float(q[3]),
            "savings_p95": float(q[4]),
            "prob_down_only_cheaper": float((s > 0).mean()),
            "mean_resets": float(self.resets.mean()),
            "mean_refinances": float(self.refinances.mean()),
        }


def simulate(paths: MarketPaths, *, balance: float, term_years: int, horizon_years: float, down_only: DownOnlyLoan,
             standard: StandardLoan, discount_rate: float, annual_turnover: float, seed: int | None,
             record_rates: bool = False) -> SimulationResult:
    """Borrower-cost lens: present value of what one borrower pays under each loan, on each path.

    Cost is scheduled payments, refinancing closing costs, and the balance repaid at the horizon or an
    earlier sale, discounted at ``discount_rate``. ``annual_turnover`` is the yearly probability of a
    sale, applied identically to both loans on a path. ``seed`` drives the behavioral draws.
    """
    _require(balance > 0, "balance must be > 0")
    _require(term_years > 0, "term_years must be > 0")
    _require(0 < horizon_years <= term_years, "horizon_years must be in (0, term_years]")
    _require(0 <= annual_turnover < 1, "annual_turnover must be in [0, 1)")
    term, months = int(round(term_years * 12)), int(round(horizon_years * 12))
    _require(months <= paths.months, "paths must cover the horizon")
    n = paths.paths
    rng = np.random.default_rng(seed)
    disc = (1 + discount_rate / 100) ** (-np.arange(months + 1) / 12)

    sold_at = np.full(n, months)
    if annual_turnover > 0:
        hit = rng.random((n, months)) < _smm(annual_turnover)
        sold_at = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, months)

    d = _DownOnlyState(down_only, paths)
    d_bal, d_cost = np.full(n, float(balance)), np.zeros(n)

    pol = standard.refinance
    s_rate = paths.index[:, 0] + standard.rate_spread
    s_bal, s_left = np.full(n, float(balance)), np.full(n, term)
    s_cost, run, refis = np.zeros(n), np.zeros(n, dtype=int), np.zeros(n, dtype=int)

    active = np.ones(n, dtype=bool)
    hist_d = hist_s = None
    if record_rates:
        hist_d, hist_s = np.empty((n, months + 1)), np.empty((n, months + 1))
        hist_d[:, 0], hist_s[:, 0] = d.rate, s_rate

    for t in range(1, months + 1):
        # Down-only: daily checks through the month, interest at the rate in effect each day.
        start_rate, avg_rate, _ = d.month(t - 1, active)
        d_pmt = _payment(d_bal, start_rate, term - (t - 1))
        d_int = d_bal * avg_rate / 1200
        d_sched = np.minimum(d_pmt - d_int, d_bal)
        d_cost += np.where(active, disc[t] * (d_int + d_sched), 0.0)
        d_bal = np.where(active, d_bal - d_sched, d_bal)

        # Standard: monthly refinancing decision.
        avail = paths.index[:, paths.month_start(t)] + standard.rate_spread
        in_money = active & (s_rate - avail >= pol.threshold) & (pol.monthly_probability > 0)
        run = np.where(in_money, run + 1, 0)
        ready = in_money & (run > pol.delay_months)
        if ready.any():
            go = ready & (rng.random(n) < pol.monthly_probability) & (rng.random(n) < pol.requalify_probability)
            s_cost += np.where(go, disc[t] * s_bal * pol.closing_costs, 0.0)
            s_rate = np.where(go, avail, s_rate)
            s_left = np.where(go, term, s_left)
            refis += go
            run = np.where(go, 0, run)
        s_pmt = _payment(s_bal, s_rate, s_left)
        s_int = s_bal * s_rate / 1200
        s_cost += np.where(active, disc[t] * s_pmt, 0.0)
        s_bal = np.where(active, np.maximum(s_bal - (s_pmt - s_int), 0.0), s_bal)
        s_left = np.where(active, s_left - 1, s_left)

        if record_rates:
            hist_d[:, t], hist_s[:, t] = d.rate, s_rate

        ending = active & ((sold_at == t) | (t == months))
        d_cost += np.where(ending, disc[t] * d_bal, 0.0)
        s_cost += np.where(ending, disc[t] * s_bal, 0.0)
        active &= ~ending

    return SimulationResult(d_cost, s_cost, d.resets, refis, months, hist_d, hist_s)


def fair_rate_spread(paths: MarketPaths, *, balance: float, term_years: int, horizon_years: float,
                     down_only: DownOnlyLoan, standard: StandardLoan, discount_rate: float, annual_turnover: float,
                     seed: int, search: tuple[float, float], tol: float = 1e-4) -> float:
    """Borrower-cost lens: the down-only spread at which its expected cost equals the standard loan's.

    Uses one fixed set of paths and behavioral draws (common random numbers). ``down_only.rate_spread`` is ignored.
    """

    def gap(s: float) -> float:
        r = simulate(paths, balance=balance, term_years=term_years, horizon_years=horizon_years,
                     down_only=down_only.with_spread(s), standard=standard, discount_rate=discount_rate,
                     annual_turnover=annual_turnover, seed=seed)
        return float(r.savings.mean())

    lo = max(search[0], float(np.max(down_only.floor - paths.index[:, 0])))
    hi = search[1]
    _require(gap(lo) > 0 > gap(hi), "search interval does not bracket the fair spread; widen `search`")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
