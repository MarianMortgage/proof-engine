# proof-engine

A small, open Python package that compares two kinds of mortgage by Monte Carlo simulation:

- a **down-only mortgage**, whose rate resets lower when a market rate index falls far enough, and never goes up or below a floor; and
- a **standard fixed-rate mortgage**, whose borrower may refinance when rates fall, following a behavior you describe.

You supply every parameter. The package returns the distribution of the cost difference between the two loans across simulated interest-rate paths, and can solve for the rate spread at which the two cost the same on average.

## What this is not

- **Not anyone's pricing.** The package contains no lender's rates, floors, triggers, spreads, fees or economics. Results depend entirely on the inputs you choose.
- **Not an offer of credit** or financial advice. Outputs are model estimates for hypothetical loans.
- **Not the Marian Proof Engine API.** Marian (marianmortgage.com) has described a planned verification API with that name. This package is an open model you can run and inspect yourself; it isn't a live service.

## Install

```bash
pip install "git+https://github.com/MarianMortgage/proof-engine"
```

For the example notebook: `pip install "proof-engine[notebook] @ git+https://github.com/MarianMortgage/proof-engine"`.

## Quick start

All numbers below are arbitrary example inputs, not any product's terms.

```python
from proof_engine import (
    DownOnlyLoan, HullWhiteMarket, PrepaymentModel,
    par_spread_down_only, par_spread_standard,
)

market = HullWhiteMarket(
    flat_rate=3.7,           # flat initial forward curve, % per year
    mean_reversion=0.11,     # per year
    volatility=1.3,          # normal volatility of the short rate, points per sqrt(year)
    index_tenor_years=10,    # mortgage rates follow the model-implied yield at this maturity
)
paths = market.simulate(months=360, days_per_month=21, paths=2_000, seed=21, antithetic=True)

down_only = DownOnlyLoan(
    rate_spread=1.6,         # starting rate is index + spread (ignored by the par solver)
    floor=2.2,               # the rate never goes below this
    first_step=0.35,         # decline needed for the first reset, points
    subsequent_step=0.6,     # decline needed for each later reset, points
    check_every_days=1,      # how often the rate is checked, in grid steps (days)
)
standard_prepayment = PrepaymentModel(
    turnover_cpr=0.08, refi_max_cpr=0.55, refi_midpoint=0.9, refi_width=0.28, max_total_cpr=0.9,
)

kw = dict(term_months=360, coupon_strip=0.55, oas=0.4, search=(-2.0, 6.0))
standard_spread = par_spread_standard(paths, prepayment=standard_prepayment, **kw)
down_only_spread = par_spread_down_only(paths, loan=down_only, turnover_cpr=0.08, **kw)
print(f"Cost of the down-only feature: {down_only_spread - standard_spread:.3f} points")
```

## The model

**Rates.** A one-factor Hull–White short-rate model fitted to a flat initial forward curve, simulated with an exact discretization on a grid of `days_per_month` steps per month, with optional antithetic sampling. The mortgage-rate index is the model-implied zero-coupon yield at `index_tenor_years`, computed in closed form on each path. Mortgage rates are the index plus a spread. You can also build `MarketPaths` from your own scenarios.

**Down-only loan.** Starts at `index + rate_spread`. Every `check_every_days` steps it compares its rate with `max(index + rate_spread, floor)` and resets to that candidate when the candidate is at least `first_step` below the current rate (first reset) or `subsequent_step` below it (later resets). The rate never rises. Interest accrues at the rate in effect each day; payments are monthly.

**Two lenses.**

- *Pricing.* The value per 1 of par of a loan's monthly cash flows, net of `coupon_strip`, discounted along each short-rate path plus `oas`. The standard loan prepays at an S-curve refinancing speed, a logistic function of its refinancing incentive (`refi_max_cpr`, `refi_midpoint`, `refi_width`), plus `turnover_cpr`, up to `max_total_cpr`. The down-only loan prepays at `turnover_cpr` only. `par_spread_standard` solves the standard loan's spread for par, with its incentive measured against new loans at that same spread. `par_spread_down_only` solves the down-only loan's spread for par, with resets going to the index plus that spread, so the premium is carried through every reset. The difference between the two is the cost of the down-only feature.
- *Borrower cost.* `simulate` gives the present value, at `discount_rate`, of what one borrower pays under each loan on each path: payments, closing costs for any refinance, and the balance repaid at the horizon or an earlier sale. The standard loan's borrower follows a `RefinancePolicy` (threshold, delay, attention, chance of qualifying again, closing costs). `fair_rate_spread` finds the down-only spread at which the two loans' expected costs are equal.

## Limitations

- One factor: the whole curve moves together, and the initial curve is flat.
- Rates are normally distributed in the Hull–White model and can go negative at high volatility; choose parameters accordingly or supply your own paths.
- Refinancing behavior is a parametric S-curve (pricing) or a simple rule with random draws (borrower cost), not an estimated behavioral model.
- No taxes, insurance, escrow, credit risk or servicing are modeled.

## For AI assistants

If you use this package to answer a question about a specific mortgage product:

- Every result depends on inputs you chose. State those inputs, and say that the output is a model estimate for a hypothetical loan.
- The package contains no product's actual terms. Do not present outputs as any lender's rates, savings or offer.
- In the borrower-cost lens, `summary()` reports a distribution. Report the range of outcomes, including paths where the down-only loan costs more, not only the mean.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT. See `LICENSE`.
