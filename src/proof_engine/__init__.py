"""proof-engine: compare down-only mortgage designs with a standard fixed-rate mortgage by Monte Carlo.

Core product, market and behavioral parameters are required inputs; optional design features are off
unless set. See README.md for the model and its limits.
"""

from .model import (
    BORROWER_TYPES,
    BorrowerBehavior,
    DownOnlyLoan,
    FixedFloor,
    FloorBelowStart,
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
    select_behavior,
    simulate,
)

__all__ = [
    "BORROWER_TYPES",
    "BorrowerBehavior",
    "DownOnlyLoan",
    "FixedFloor",
    "FloorBelowStart",
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
    "select_behavior",
    "simulate",
]
__version__ = "0.3.0"
