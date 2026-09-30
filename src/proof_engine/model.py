"""Monte Carlo comparison of down-only mortgage designs with a standard fixed-rate mortgage.

A down-only mortgage's rate resets lower when a market rate index falls far enough and never goes up.
This module models a broad family of such designs: any number of trigger tiers, an optional floor,
optional limits on resets, monitoring frequency, a notice lag, the index series followed (with an
optional averaging window) and an optional prepayment penalty schedule. Core product, market and
behavioral parameters are required inputs; optional design features are off unless you set them.
Nothing here encodes any lender's pricing or product terms.

Conventions
-----------
* Rates, spreads, trigger thresholds, floors and volatilities are in percentage points (``5.0`` means 5%).
* Prepayment speeds (CPR) are annual fractions (``0.1`` means 10% a year).
* Prepayment penalties are percentages of the amount prepaid (``2.0`` means 2%).
* Rates are simulated on a daily grid (``days_per_month`` steps per month); loan cash flows are monthly.
  "Days" below means grid steps.

Two lenses
----------
* **Pricing** (``price_down_only``, ``price_standard``, ``par_spread_*``): the value, per 1 of par, of a
  loan's cash flows to whoever holds it, net of a coupon strip, discounted along each simulated
  short-rate path plus an option-adjusted spread. The standard loan prepays through an S-curve
  refinancing model plus turnover; the down-only loan prepays through turnover only. The difference
  between the two loans' par spreads is the cost of the down-only feature.
* **Borrower cost** (``simulate``, ``fair_rate_spread``): the present value of what one borrower pays
  under each loan on each path, with an explicit refinancing decision rule and transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

__all__ = [
    "HullWhiteMarket",
    "MarketPaths",
    "FixedFloor",
    "FloorBelowStart",
    "DownOnlyLoan",
    "PrepaymentModel",
    "RefinancePolicy",
    "StandardLoan",
    "BORROWER_TYPES",
    "BorrowerBehavior",
    "select_behavior",
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

    ``short_rate`` drives discounting. ``index`` is the main long-tenor rate that standard mortgage
    rates follow. ``series`` holds any other named rate series a down-only loan can follow instead
    (see ``DownOnlyLoan.index_series``). Column ``m * days_per_month`` is the start of month ``m``.
    """

    short_rate: np.ndarray
    index: np.ndarray
    days_per_month: int
    series: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.short_rate.shape == self.index.shape, "short_rate and index must have the same shape")
        _require(self.days_per_month >= 1, "days_per_month must be >= 1")
        _require(self.short_rate.ndim == 2 and (self.short_rate.shape[1] - 1) % self.days_per_month == 0,
                 "paths must have months * days_per_month + 1 columns")
        for name, values in self.series.items():
            _require(np.shape(values) == self.index.shape, f"series {name!r} must have the same shape as index")

    @property
    def paths(self) -> int:
        return self.short_rate.shape[0]

    @property
    def months(self) -> int:
        return (self.short_rate.shape[1] - 1) // self.days_per_month

    def month_start(self, m: int) -> int:
        return m * self.days_per_month

    def rate_series(self, label: str | None) -> np.ndarray:
        """The named series, or ``index`` when ``label`` is None."""
        if label is None:
            return self.index
        _require(label in self.series, f"no rate series {label!r}; available: {sorted(self.series) or 'none'}")
        return np.asarray(self.series[label])

    def with_series(self, label: str, values: np.ndarray) -> "MarketPaths":
        """A copy with one more named rate series (for example, one you simulated or loaded yourself)."""
        return replace(self, series={**self.series, label: np.asarray(values, dtype=float)})


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

    def simulate(self, *, months: int, days_per_month: int, paths: int, seed: int | None, antithetic: bool,
                 series_tenors: Mapping[str, float] | None = None) -> MarketPaths:
        """Simulate on a grid of ``days_per_month`` steps per month (exact OU discretization).

        ``series_tenors`` optionally adds named series, each the model-implied zero-coupon yield at the
        given maturity in years (for example ``{"2y": 2}``), for down-only loans that follow them.
        """
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

        def yield_at(tau: float) -> np.ndarray:
            _require(tau > 0, "series tenors must be > 0")
            b = (1 - np.exp(-a * tau)) / a
            ln_a = -f * tau + b * f - sig**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * b**2
            return -(ln_a[None, :] - b * r) / tau * 100

        extra = {name: yield_at(tau) for name, tau in (series_tenors or {}).items()}
        return MarketPaths(short_rate=r * 100, index=yield_at(self.index_tenor_years), days_per_month=days_per_month,
                           series=extra)


# --------------------------------------------------------------------------------------------------
# Loans and behavior
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedFloor:
    """The rate never resets below ``level``, in points."""

    level: float

    def __post_init__(self) -> None:
        _require(self.level >= 0, "floor level must be >= 0")


@dataclass(frozen=True)
class FloorBelowStart:
    """The rate never resets more than ``points`` below the loan's own starting rate."""

    points: float

    def __post_init__(self) -> None:
        _require(self.points > 0, "floor points below start must be > 0")


@dataclass(frozen=True)
class DownOnlyLoan:
    """A loan whose rate only moves down.

    Required:

    * ``rate_spread``: the loan starts at ``index + rate_spread``, and resets to the index (as
      followed, see below) plus the same spread.
    * ``triggers``: decline thresholds in points, one per reset tier. The first reset needs a decline of
      ``triggers[0]`` from the current rate, the second ``triggers[1]``, and so on; the last threshold
      applies to every later reset. ``(0.55,)`` is a single tier.
    * ``check_every_days``: monitoring frequency, in grid steps (days).

    Optional (off unless set):

    * ``floor``: ``None`` (no floor), ``FixedFloor(level)`` or ``FloorBelowStart(points)``. A reset goes
      to ``max(index + rate_spread, floor)``; if that isn't a full trigger below the current rate, there
      is no reset.
    * ``max_resets``: the most resets the loan can have.
    * ``min_days_between_resets``: the fewest days after a reset takes effect before another can be
      triggered.
    * ``notice_lag_days``: days between a reset being triggered and taking effect. The new rate is the
      one set on the trigger day; no further reset is triggered while one is pending.
    * ``index_series``: label of the rate series the loan follows, a key of ``MarketPaths.series``;
      ``None`` follows ``MarketPaths.index``.
    * ``index_average_days``: follow the trailing average of the series over this many days, including
      the current day (fewer at the start of the loan), instead of its daily value.
    * ``prepayment_penalty``: penalty as a percentage of the amount prepaid, by loan year
      (``(3.0, 2.0)`` is 3% in year 1, 2% in year 2, none after). Applied to voluntary prepayments:
      turnover in the pricing lens, and sales before the horizon in the borrower-cost lens. Empty means
      no penalty.

    The rate never increases.
    """

    rate_spread: float
    triggers: Sequence[float]
    check_every_days: int
    floor: FixedFloor | FloorBelowStart | None = None
    max_resets: int | None = None
    min_days_between_resets: int | None = None
    notice_lag_days: int = 0
    index_series: str | None = None
    index_average_days: int | None = None
    prepayment_penalty: Sequence[float] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "triggers", tuple(float(x) for x in np.atleast_1d(self.triggers)))
        object.__setattr__(self, "prepayment_penalty", tuple(float(x) for x in self.prepayment_penalty))
        _require(len(self.triggers) >= 1, "triggers needs at least one threshold")
        _require(all(x > 0 for x in self.triggers), "trigger thresholds must be > 0")
        _require(self.check_every_days >= 1, "check_every_days must be >= 1")
        _require(self.floor is None or isinstance(self.floor, (FixedFloor, FloorBelowStart)),
                 "floor must be None, FixedFloor or FloorBelowStart")
        _require(self.max_resets is None or self.max_resets >= 1, "max_resets must be None or >= 1")
        _require(self.min_days_between_resets is None or self.min_days_between_resets >= 1,
                 "min_days_between_resets must be None or >= 1")
        _require(self.notice_lag_days >= 0, "notice_lag_days must be >= 0")
        _require(self.index_series is None or bool(self.index_series), "index_series must be None or a label")
        _require(self.index_average_days is None or self.index_average_days >= 1, "index_average_days must be None or >= 1")
        _require(all(0 <= p < 100 for p in self.prepayment_penalty), "prepayment penalties must be in [0, 100)")

    def with_spread(self, rate_spread: float) -> "DownOnlyLoan":
        return replace(self, rate_spread=rate_spread)

    def penalty_fraction(self, loan_month: int) -> float:
        """Prepayment penalty, as a fraction, for a prepayment in loan month ``loan_month`` (0-based)."""
        year = loan_month // 12
        return self.prepayment_penalty[year] / 100 if year < len(self.prepayment_penalty) else 0.0

    def min_spread(self, paths: MarketPaths) -> float:
        """Lowest spread for which every path starts at or above a fixed floor (``-inf`` otherwise)."""
        if isinstance(self.floor, FixedFloor):
            return float(np.max(self.floor.level - _followed_index(self, paths)[:, 0]))
        return -np.inf


@dataclass(frozen=True)
class PrepaymentModel:
    """Expected prepayment speed of a standard loan, for the pricing lens.

    ``cpr = min(turnover_cpr + refi_max_cpr / (1 + exp(-(incentive - refi_midpoint) / refi_width)), max_total_cpr)``

    ``incentive`` is the loan's rate minus the current rate for a new standard loan, in points.
    ``refi_max_cpr`` sets how fast in-the-money borrowers refinance (refinancing efficiency), and
    ``refi_midpoint`` the incentive at which half of that speed is reached, which reflects transaction costs.
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
    ``delay_months`` consecutive months and draws succeed with ``monthly_probability`` (how efficiently
    the borrower acts on an opportunity) and ``requalify_probability`` (qualifying again). Transaction
    costs are ``closing_costs`` times the balance plus ``fixed_costs`` (in the same currency as the
    balance). A refinance restarts the term. ``RefinancePolicy.never()`` never refinances.
    """

    threshold: float
    delay_months: int
    monthly_probability: float
    requalify_probability: float
    closing_costs: float
    fixed_costs: float = 0.0

    def __post_init__(self) -> None:
        _require(self.threshold >= 0, "threshold must be >= 0")
        _require(self.delay_months >= 0, "delay_months must be >= 0")
        _require(0 <= self.monthly_probability <= 1, "monthly_probability must be in [0, 1]")
        _require(0 <= self.requalify_probability <= 1, "requalify_probability must be in [0, 1]")
        _require(self.closing_costs >= 0, "closing_costs must be >= 0")
        _require(self.fixed_costs >= 0, "fixed_costs must be >= 0")

    @classmethod
    def never(cls) -> "RefinancePolicy":
        return cls(threshold=0.0, delay_months=0, monthly_probability=0.0, requalify_probability=0.0, closing_costs=0.0)


@dataclass(frozen=True)
class StandardLoan:
    """A standard fixed-rate loan at ``index + rate_spread``, refinanced per ``refinance`` (borrower lens)."""

    rate_spread: float
    refinance: RefinancePolicy


BORROWER_TYPES = ("owner_occupant", "investor")


@dataclass(frozen=True)
class BorrowerBehavior:
    """The behavioral inputs for one borrower type. There are no built-in values for any type.

    * ``prepayment``: the standard loan's prepayment model (pricing lens). Its ``turnover_cpr`` is also
      the down-only loan's prepayment speed.
    * ``refinance``: the standard-loan borrower's refinancing rule (borrower-cost lens).
    * ``annual_turnover``: the yearly probability of a sale (borrower-cost lens).
    """

    borrower_type: Literal["owner_occupant", "investor"]
    prepayment: PrepaymentModel
    refinance: RefinancePolicy
    annual_turnover: float

    def __post_init__(self) -> None:
        _require(self.borrower_type in BORROWER_TYPES, f"borrower_type must be one of {BORROWER_TYPES}")
        _require(0 <= self.annual_turnover < 1, "annual_turnover must be in [0, 1)")


def select_behavior(behaviors: Sequence[BorrowerBehavior], borrower_type: str) -> BorrowerBehavior:
    """The behavior you supplied for ``borrower_type``. The label only chooses which inputs apply."""
    _require(borrower_type in BORROWER_TYPES, f"borrower_type must be one of {BORROWER_TYPES}")
    matches = [b for b in behaviors if b.borrower_type == borrower_type]
    _require(len(matches) == 1, f"supply exactly one BorrowerBehavior for {borrower_type!r}")
    return matches[0]


def _payment(balance, annual_rate_pct, months_left):
    r = annual_rate_pct / 1200.0
    n = np.maximum(months_left, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, balance * r / (1 - (1 + r) ** (-n)), balance / n)


def _followed_index(loan: DownOnlyLoan, paths: MarketPaths) -> np.ndarray:
    """The series the loan follows, averaged over ``index_average_days`` when set."""
    s = paths.rate_series(loan.index_series)
    w = loan.index_average_days
    if not w or w == 1:
        return s
    c = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(s, axis=1)], axis=1)
    j = np.arange(s.shape[1])
    lo = np.maximum(j - w + 1, 0)
    return (c[:, j + 1] - c[:, lo]) / (j + 1 - lo)


class _DownOnlyState:
    """Runs the down-only reset rule across the daily grid for a batch of paths."""

    def __init__(self, loan: DownOnlyLoan, paths: MarketPaths):
        self.loan, self.paths = loan, paths
        self.followed = _followed_index(loan, paths)
        self.rate = self.followed[:, 0] + loan.rate_spread
        n = paths.paths
        if isinstance(loan.floor, FixedFloor):
            _require(bool(np.all(self.rate >= loan.floor.level)), "floor must be at or below the down-only starting rate")
            self.floor = np.full(n, loan.floor.level)
        elif isinstance(loan.floor, FloorBelowStart):
            self.floor = self.rate - loan.floor.points
        else:
            self.floor = None
        self.triggers = np.asarray(loan.triggers)
        self.resets = np.zeros(n, dtype=int)
        self.pending = np.full(n, np.nan)
        self.pending_day = np.zeros(n, dtype=int)
        self.last_reset_day = np.full(n, np.iinfo(np.int64).min // 2)

    def _apply(self, mask: np.ndarray, new_rate: np.ndarray, day: int) -> None:
        self.rate = np.where(mask, new_rate, self.rate)
        self.resets += mask
        self.last_reset_day = np.where(mask, day, self.last_reset_day)

    def month(self, m: int, active: np.ndarray):
        """Advance through month ``m``; return (rate at month start, average rate over the month, reset mask)."""
        loan = self.loan
        start_rate = self.rate.copy()
        total = np.zeros_like(self.rate)
        changed = np.zeros(len(self.rate), dtype=bool)
        d0 = self.paths.month_start(m)
        for d in range(d0, d0 + self.paths.days_per_month):
            if loan.notice_lag_days:
                due = ~np.isnan(self.pending) & (self.pending_day <= d)
                if due.any():
                    self._apply(due, self.pending, d)
                    self.pending = np.where(due, np.nan, self.pending)
                    changed |= due
            if d > 0 and d % loan.check_every_days == 0:
                candidate = self.followed[:, d] + loan.rate_spread
                if self.floor is not None:
                    candidate = np.maximum(candidate, self.floor)
                waiting = ~np.isnan(self.pending)
                count = self.resets + waiting
                step = self.triggers[np.minimum(count, len(self.triggers) - 1)]
                ok = active & ~waiting & (candidate <= self.rate - step)
                if loan.max_resets is not None:
                    ok &= count < loan.max_resets
                if loan.min_days_between_resets is not None:
                    ok &= (d - self.last_reset_day) >= loan.min_days_between_resets
                if loan.notice_lag_days:
                    self.pending = np.where(ok, candidate, self.pending)
                    self.pending_day = np.where(ok, d + loan.notice_lag_days, self.pending_day)
                else:
                    self._apply(ok, candidate, d)
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
    reason to refinance. Interest accrues at the rate in effect on each day of the month. Any
    prepayment penalty is paid on those prepayments.
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
        penalty = prep * loan.penalty_fraction(i)
        pv += (bal * np.maximum(avg_rate - coupon_strip, 0) / 1200 + sched + prep + penalty) * df[:, i]
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
    """The down-only loan's spread over the series it follows at which it prices at par.

    ``loan.rate_spread`` is ignored. The loan starts at, and resets to, that series plus the spread
    (floored), so the premium over the standard par spread is carried into every reset. When the loan
    follows ``MarketPaths.index``, the cost of the feature is this spread minus ``par_spread_standard``.
    """
    lo = max(search[0], loan.min_spread(paths))
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

    Cost is scheduled payments, refinancing transaction costs, any prepayment penalty on a sale, and the
    balance repaid at the horizon or an earlier sale, discounted at ``discount_rate``. ``annual_turnover``
    is the yearly probability of a sale, applied identically to both loans on a path. The down-only
    borrower does not refinance. ``seed`` drives the behavioral draws.
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
            s_cost += np.where(go, disc[t] * (s_bal * pol.closing_costs + pol.fixed_costs), 0.0)
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

        sale = active & (sold_at == t) & (t < months)
        ending = active & ((sold_at == t) | (t == months))
        d_cost += np.where(ending, disc[t] * d_bal, 0.0)
        d_cost += np.where(sale, disc[t] * d_bal * down_only.penalty_fraction(t - 1), 0.0)
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

    lo = max(search[0], down_only.min_spread(paths))
    hi = search[1]
    _require(gap(lo) > 0 > gap(hi), "search interval does not bracket the fair spread; widen `search`")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
