# Changelog

All notable changes to proof-engine. Versions follow [Semantic Versioning](https://semver.org/); before 1.0, a minor version can change the API.

## 0.3.0 (2026-09-30)

First public release.

- **Rate paths.** A one-factor Hull–White short-rate model on a flat initial forward curve, simulated on a daily grid with an exact discretization and optional antithetic sampling. Mortgage-rate indexes are model-implied zero-coupon yields at any tenor, in closed form; you can also add your own rate series.
- **Down-only loan designs.** Any number of reset trigger tiers; an optional floor, fixed or set below the starting rate; optional limits on the number of resets and the time between them; monitoring frequency; a notice lag; the index series followed, with an optional averaging window; and an optional prepayment penalty schedule.
- **Standard loan and refinancing behavior.** A fixed-rate loan compared on the same paths, with an S-curve prepayment model for the pricing lens and a refinancing decision rule with transaction costs and efficiency for the borrower-cost lens. A borrower-type label selects behavioral inputs you supply.
- **Pricing lens.** Option-adjusted valuation of each loan's cash flows and par-spread solvers, giving the cost of a design's down-only feature.
- **Borrower-cost lens.** The distribution of what one borrower pays under each loan across rate paths, and a solver for the fair down-only spread.
- **Documentation and examples.** A methodology section, runnable example scripts, a quick-start notebook, a test suite and continuous integration.
