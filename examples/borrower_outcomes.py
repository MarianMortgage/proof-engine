"""The range of outcomes for one borrower under each design (borrower-cost lens).

Illustrative, not Marian terms. All inputs are arbitrary examples; see examples/_designs.py.

Compares each design with a standard fixed-rate loan at the same spread, whose borrower refinances by
the rule below, over a 10-year horizon. Savings are the present value of what the borrower pays under
the standard loan minus under the down-only loan: positive means the down-only loan cost less on that
path. Read the whole range, including paths where the down-only loan costs more.

    uv run python examples/borrower_outcomes.py
"""

from _designs import DESIGNS, LABEL, paths

from proof_engine import RefinancePolicy, StandardLoan, simulate

REFINANCE = RefinancePolicy(threshold=0.75, delay_months=3, monthly_probability=0.3, requalify_probability=0.85,
                            closing_costs=0.02, fixed_costs=1_000)
BORROWER = dict(balance=300_000, term_years=30, horizon_years=10, discount_rate=3.1, annual_turnover=0.08, seed=5)


def main() -> None:
    print(LABEL)
    p = paths()
    standard = StandardLoan(rate_spread=1.6, refinance=REFINANCE)
    print(f"{'Design':40s} {'p5':>9s} {'median':>9s} {'p95':>9s}  cheaper on")
    for name, loan in DESIGNS.items():
        s = simulate(p, down_only=loan, standard=standard, **BORROWER).summary()
        print(f"{name:40s} {s['savings_p5']:>9,.0f} {s['savings_p50']:>9,.0f} {s['savings_p95']:>9,.0f}"
              f"  {s['prob_down_only_cheaper']:.0%} of paths")


if __name__ == "__main__":
    main()
