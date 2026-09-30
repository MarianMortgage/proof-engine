"""proof-engine: compare a down-only mortgage with a standard fixed-rate mortgage by Monte Carlo.

All product, market and behavioral parameters are required inputs. See README.md for the model and its limits.
"""

from .model import (
    DownOnlyLoan,
    HullWhiteMarket,
    MarketPaths,
    PrepaymentModel,
    RefinancePolicy,
    SimulationResult,
    StandardLoan,
    fair_rate_spread,
    par_spread_down_only,
    par_spread_standard,
    price_down_only,
    price_standard,
    simulate,
)

__all__ = [
    "DownOnlyLoan",
    "HullWhiteMarket",
    "MarketPaths",
    "PrepaymentModel",
    "RefinancePolicy",
    "SimulationResult",
    "StandardLoan",
    "fair_rate_spread",
    "par_spread_down_only",
    "par_spread_standard",
    "price_down_only",
    "price_standard",
    "simulate",
]
__version__ = "0.2.0"
