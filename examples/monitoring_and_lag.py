"""How monitoring frequency and a notice lag change the cost of one design (pricing lens).

Illustrative, not Marian terms. All inputs are arbitrary examples; see examples/_designs.py.

Takes the one-tier, no-floor example design and varies how often the rate is checked and how long a
triggered reset takes to take effect. Checking less often, or waiting longer, means some rate drops are
caught later or missed, so the feature is worth less and its cost falls.

    uv run python examples/monitoring_and_lag.py
"""

from dataclasses import replace

from _designs import DESIGNS, LABEL, PREPAYMENT, PRICING, paths

from proof_engine import par_spread_down_only, par_spread_standard


def main() -> None:
    print(LABEL)
    p = paths()
    base = DESIGNS["One tier, no floor"]
    standard = par_spread_standard(p, prepayment=PREPAYMENT, **PRICING)
    print(f"{'Checked every':>14s} {'notice lag':>11s}  cost of the feature")
    for every in (1, 5, 21):
        for lag in (0, 10):
            loan = replace(base, check_every_days=every, notice_lag_days=lag)
            down = par_spread_down_only(p, loan=loan, turnover_cpr=PREPAYMENT.turnover_cpr, **PRICING)
            print(f"{every:>9d} days {lag:>6d} days  {down - standard:+.3f} points")


if __name__ == "__main__":
    main()
