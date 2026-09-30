"""Compare the cost of the down-only feature across designs (pricing lens).

Illustrative, not Marian terms. All inputs are arbitrary examples; see examples/_designs.py.

For each design, solves the down-only loan's par spread and subtracts the standard loan's par spread
on the same rate paths. The difference, in points of rate, is the model's cost of that design's
feature under these market and prepayment inputs.

    uv run python examples/compare_designs.py
"""

from _designs import DESIGNS, LABEL, PREPAYMENT, PRICING, paths

from proof_engine import par_spread_down_only, par_spread_standard


def main() -> None:
    print(LABEL)
    p = paths()
    standard = par_spread_standard(p, prepayment=PREPAYMENT, **PRICING)
    print(f"Standard loan par spread: {standard:.3f} points over the index\n")
    print(f"{'Design':40s} cost of the feature")
    for name, loan in DESIGNS.items():
        down = par_spread_down_only(p, loan=loan, turnover_cpr=PREPAYMENT.turnover_cpr, **PRICING)
        print(f"{name:40s} {down - standard:+.3f} points")


if __name__ == "__main__":
    main()
