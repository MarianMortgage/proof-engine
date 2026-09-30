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

`examples/quickstart.ipynb` runs the same comparison and adds the borrower-cost lens for each design.

## The model

**Rates.** A one-factor Hull–White short-rate model fitted to a flat initial forward curve, simulated with an exact discretization on a grid of `days_per_month` steps per month, with optional antithetic sampling. The main index is the model-implied zero-coupon yield at `index_tenor_years`, computed in closed form on each path; `series_tenors` adds yields at other maturities as named series. You can also build `MarketPaths` from your own scenarios and add any series with `with_series`.

**Down-only loan.** Starts at `index + rate_spread`, where the index is the series the loan follows (optionally averaged). Every `check_every_days` steps it compares its rate with a candidate, `index + rate_spread` (or the floor, if higher). It resets when the candidate is at least the current tier's threshold below the current rate, subject to any reset limit, spacing and notice lag. The rate never rises. Interest accrues at the rate in effect each day; payments are monthly.

**Two lenses.**

- *Pricing.* The value per 1 of par of a loan's monthly cash flows, net of `coupon_strip`, discounted along each short-rate path plus `oas`. The standard loan prepays at an S-curve refinancing speed, a logistic function of its refinancing incentive (`refi_max_cpr`, `refi_midpoint`, `refi_width`), plus `turnover_cpr`, up to `max_total_cpr`. The down-only loan prepays at `turnover_cpr` only, and pays any prepayment penalty on those prepayments. `par_spread_standard` solves the standard loan's spread for par, with its incentive measured against new loans at that same spread. `par_spread_down_only` solves the down-only loan's spread for par, with resets going to the index plus that spread, so the premium is carried through every reset. When both follow the same index, the difference between the two is the cost of the design's down-only feature.
- *Borrower cost.* `simulate` gives the present value, at `discount_rate`, of what one borrower pays under each loan on each path: payments, transaction costs for any refinance, any prepayment penalty on a sale, and the balance repaid at the horizon or an earlier sale. The standard loan's borrower follows a `RefinancePolicy` (threshold, delay, efficiency, chance of qualifying again, transaction costs). `fair_rate_spread` finds the down-only spread at which the two loans' expected costs are equal.

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
