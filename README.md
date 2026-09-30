# proof-engine

A small, open Python package that compares, by Monte Carlo simulation:

- **down-only mortgages**, whose rate resets lower when a market rate index falls far enough and never goes up, across a broad family of designs; and
- a **standard fixed-rate mortgage**, whose borrower may refinance when rates fall, following a behavior you describe.

You describe the design and every market and behavioral input. The package returns the distribution of the cost difference between the two loans across simulated interest-rate paths, and solves for the rate spread at which a design's cost is fair.

## What this is not

- **Not anyone's pricing or product.** The package contains no lender's rates, floors, triggers, spreads, fees or economics, and none of the example designs below is any company's product. Results depend entirely on the inputs you choose.
- **Not an offer of credit** or financial advice. Outputs are model estimates for hypothetical loans.
- **Not the Marian Proof Engine API.** Marian (marianmortgage.com) has described a planned verification API with that name. This package is an open model you can run and inspect yourself; it isn't a live service.

## Install

```bash
pip install "git+https://github.com/MarianMortgage/proof-engine"
```

For the example notebook: `pip install "proof-engine[notebook] @ git+https://github.com/MarianMortgage/proof-engine"`.

## Designs you can model

A `DownOnlyLoan` has three required inputs and several optional features, which are off unless you set them.

| Input | Required? | What it does |
| --- | --- | --- |
| `rate_spread` | yes | Starting rate is the followed index plus this spread; resets go to the index plus the same spread. |
| `triggers` | yes | Decline thresholds, one per reset tier: `(0.55,)` is one tier; `(0.3, 0.45, 0.75)` needs 0.3 points for the first reset, 0.45 for the second, and 0.75 for every later one. |
| `check_every_days` | yes | Monitoring frequency, in grid steps (days). |
| `floor` | no | `None`, `FixedFloor(level)`, or `FloorBelowStart(points)` (a floor set relative to the loan's own starting rate). |
| `max_resets` | no | The most resets the loan can have. |
| `min_days_between_resets` | no | The fewest days after a reset before another can be triggered. |
| `notice_lag_days` | no | Days between a reset being triggered and taking effect. |
| `index_series` | no | Label of the rate series the loan follows, for example `"2y"` from `simulate(series_tenors={"2y": 2})`, or any series you add with `MarketPaths.with_series`. Default: the main index. |
| `index_average_days` | no | Follow a trailing average of the series over this many days instead of its daily value. |
| `prepayment_penalty` | no | Percentage of the amount prepaid, by loan year: `(3.0, 2.0, 1.0)` is 3% in year 1, 2% in year 2, 1% in year 3. Applies to voluntary prepayments. Default: none. |

Borrower behavior is also an input. `BorrowerBehavior` bundles the prepayment model, refinancing rule and sale probability for one borrower type (`"owner_occupant"` or `"investor"`), and `select_behavior` picks the one you supplied for a given type. The label only chooses which of your inputs apply; there are no built-in values for either type. Refinancing frictions are explicit: `RefinancePolicy` takes transaction costs as `closing_costs` (a fraction of the balance) and `fixed_costs` (a flat amount), and refinancing efficiency as `monthly_probability`; in the pricing lens, `PrepaymentModel.refi_max_cpr` sets refinancing speed.

## Quick start: several designs side by side

All numbers below are arbitrary example inputs, not any product's terms.

```python
from proof_engine import (
    DownOnlyLoan, FixedFloor, FloorBelowStart, HullWhiteMarket, PrepaymentModel,
    par_spread_down_only, par_spread_standard,
)

market = HullWhiteMarket(
    flat_rate=3.7,           # flat initial forward curve, % per year
    mean_reversion=0.11,     # per year
    volatility=1.3,          # normal volatility of the short rate, points per sqrt(year)
    index_tenor_years=10,    # mortgage rates follow the model-implied yield at this maturity
)
paths = market.simulate(months=360, days_per_month=21, paths=2_000, seed=21, antithetic=True)

designs = {
    "One tier, no floor": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.55,), check_every_days=1),
    "Two tiers, fixed floor": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.35, 0.6), check_every_days=1, floor=FixedFloor(2.2)),
    "Three tiers, reset limit": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.3, 0.45, 0.75), check_every_days=1, max_resets=4),
    "Prepayment penalty, floor below start": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.55,), check_every_days=1, floor=FloorBelowStart(1.8),
        prepayment_penalty=(3.0, 2.0, 1.0)),
    "Averaged index, notice lag, spacing": DownOnlyLoan(
        rate_spread=1.6, triggers=(0.4,), check_every_days=5, index_average_days=20,
        notice_lag_days=10, min_days_between_resets=60),
}

prepayment = PrepaymentModel(
    turnover_cpr=0.08, refi_max_cpr=0.55, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9,
)
kw = dict(term_months=360, coupon_strip=0.55, oas=0.4, search=(-2.0, 6.0))
standard_spread = par_spread_standard(paths, prepayment=prepayment, **kw)
for name, loan in designs.items():
    spread = par_spread_down_only(paths, loan=loan, turnover_cpr=prepayment.turnover_cpr, **kw)
    print(f"{name:40s} cost of the feature: {spread - standard_spread:.3f} points")
```

## Examples

Runnable scripts in `examples/`, all using arbitrary example inputs (illustrative, not Marian terms or anyone else's):

- `compare_designs.py`: the cost of the down-only feature for each example design (pricing lens).
- `borrower_outcomes.py`: the range of outcomes for one borrower under each design, against a standard loan with refinancing (borrower-cost lens).
- `monitoring_and_lag.py`: how monitoring frequency and a notice lag change the cost of one design.

Run one with `uv run python examples/compare_designs.py`. `examples/quickstart.ipynb` walks through the same comparisons in a notebook.

## Methodology

### Interest-rate paths

Rates follow a one-factor Hull–White model,

    dr = (θ(t) − a·r) dt + σ dW,

with `a = mean_reversion` and `σ = volatility`, and `θ(t)` chosen so the model reproduces a flat initial forward curve at `flat_rate`. The short rate is simulated with an exact discretization of its Ornstein–Uhlenbeck part on a grid of `days_per_month` steps per month, so the rate paths have no time-step error. Antithetic sampling (`antithetic=True`) pairs each path with its mirror image to reduce noise.

Mortgage rates follow a longer-term rate. The main index is the model-implied zero-coupon yield at `index_tenor_years`, computed in closed form on every path and every day. `series_tenors` adds yields at other maturities as named series, and `MarketPaths.with_series` adds any series you simulate or load yourself. A loan's rate is the series it follows plus its spread. Discounting in the pricing lens uses the short rate along each path.

### Reset rules, floors and monitoring

A `DownOnlyLoan` starts at the followed index plus `rate_spread`. On each monitoring day (every `check_every_days` grid steps after the start) it computes a candidate rate, the followed index plus the same spread, raised to the floor if there is one, and resets to the candidate when all of these hold:

- the candidate is at least the current tier's threshold below the current rate: `triggers[0]` for the first reset, `triggers[1]` for the second, and the last threshold for every later reset;
- the loan has had fewer than `max_resets` resets, if set;
- at least `min_days_between_resets` days have passed since the last reset took effect, if set;
- no earlier reset is still waiting to take effect.

With `notice_lag_days`, a reset triggered on one day takes effect that many days later, at the rate set on the trigger day. With `index_average_days`, the loan follows a trailing average of the series over that many days, including the current day (fewer at the start of the loan), instead of its daily value.

The floor is either `FixedFloor(level)` or `FloorBelowStart(points)`, which sits a set amount below the loan's own starting rate. When the index falls far enough that the candidate is the floor, the loan resets to the floor if that is still a full trigger below its current rate, and otherwise does not reset. The rate never rises. Interest accrues at the rate in effect on each day; the payment is recalculated monthly from the balance, the rate at the start of the month and the remaining term.

### How refinancing behavior is compared

The standard loan is a fixed-rate mortgage at the main index plus its spread. It is compared with the down-only loan on the same rate paths in two ways.

- **Pricing lens.** Each loan's monthly cash flows, net of `coupon_strip`, are discounted along each short-rate path plus `oas` and averaged across paths, per 1 of par. The standard loan prepays at a speed set by `PrepaymentModel`: `turnover_cpr` plus an S-curve in the refinancing incentive (its rate minus the rate on a new standard loan), `refi_max_cpr / (1 + exp(−(incentive − refi_midpoint) / refi_width))`, capped at `max_total_cpr`. The down-only loan prepays at `turnover_cpr` only, because its borrower gets rate drops without refinancing, and pays any `prepayment_penalty` on those prepayments. `par_spread_standard` and `par_spread_down_only` solve each loan's spread for a value of par. The standard loan's incentive is measured against new loans at the solved spread, and the down-only loan resets to the index plus its solved spread, so both are self-consistent.
- **Borrower-cost lens.** `simulate` follows one borrower under each loan on each path and discounts what they pay at `discount_rate`: payments, refinancing costs, any prepayment penalty on a sale, and the balance repaid at a sale or at the horizon. The standard-loan borrower refinances, checked monthly, when their rate exceeds the new-loan rate by `threshold` for more than `delay_months` months and two random draws succeed: acting on the opportunity (`monthly_probability`) and qualifying again (`requalify_probability`). Each refinance costs `closing_costs` times the balance plus `fixed_costs`, and restarts the term. A sale ends both loans on the same path in the same month, drawn from `annual_turnover`. Behavioral draws come from `seed`, so comparisons between designs use common random numbers. `fair_rate_spread` finds the down-only spread at which the two loans' expected costs are equal.

`BorrowerBehavior` groups the behavioral inputs for one borrower type, and `select_behavior` picks the one you supplied for `"owner_occupant"` or `"investor"`. The package has no built-in behavior for either.

### What the outputs mean, and what they don't

- `par_spread_down_only(...) − par_spread_standard(...)` is the model's cost of a design's down-only feature, in points of rate, when both loans follow the same index: how much higher the down-only loan's rate must be for a holder to value it the same as a standard loan, under your market and prepayment inputs.
- `simulate(...).summary()` is the distribution, across paths, of how much less (positive) or more (negative) one borrower pays with the down-only loan, in present value. The mean hides a range; paths where the down-only loan costs more are part of the answer.
- `fair_rate_spread(...)` is the down-only spread at which that borrower breaks even on average against the standard loan and refinancing behavior you chose.
- Outputs are model estimates for hypothetical loans. They are not any lender's rates, prices, terms or offer, and not a forecast of rates. They are only as good as the inputs: volatility, mean reversion, prepayment and refinancing behavior drive the results, so test a range of values rather than relying on one set.

Both solvers work for every design.

## Limitations

- One factor: the whole curve moves together, and the initial curve is flat.
- Rates are normally distributed in the Hull–White model and can go negative at high volatility; choose parameters accordingly or supply your own paths.
- Refinancing behavior is a parametric S-curve (pricing) or a simple rule with random draws (borrower cost), not an estimated behavioral model.
- The down-only borrower is assumed not to refinance. With a floor or a reset limit, a real borrower might refinance after rates fall past it.
- No taxes, insurance, escrow, credit risk or servicing are modeled.

## For AI assistants

If you use this package to answer a question about a specific mortgage product:

- Every result depends on the design and inputs you chose. State them, and say that the output is a model estimate for a hypothetical loan.
- The package contains no product's actual terms, and the example designs are not any company's product. Do not present outputs as any lender's rates, savings or offer.
- In the borrower-cost lens, `summary()` reports a distribution. Report the range of outcomes, including paths where the down-only loan costs more, not only the mean.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT. See `LICENSE`.
