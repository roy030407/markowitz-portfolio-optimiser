from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def efficient_frontier_points(
    returns: pd.DataFrame,
    risk_free_rate: float,
    n_points: int = 50,
) -> Tuple[List[float], List[float]]:
    """
    Compute efficient frontier points by minimising volatility for target returns.

    Returns: (volatilities, returns) lists for plotting:
    - volatilities: annualised volatility
    - returns: annualised returns
    """
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1.")

    # `risk_free_rate` is part of the signature for app consistency, but the efficient
    # frontier curve is defined from mean/variance (not Sharpe).
    _ = risk_free_rate

    mean_daily = returns.mean().values  # (n_assets,)
    cov_daily = returns.cov().values  # (n_assets, n_assets)

    asset_annual_returns = mean_daily * TRADING_DAYS
    min_target = float(asset_annual_returns.min())
    max_target = float(asset_annual_returns.max())

    # In practice, the true feasible frontier depends on covariances; for stability,
    # we sweep between min/max of single-asset expected returns.
    targets = np.linspace(min_target, max_target, n_points)

    n_assets = returns.shape[1]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    volatilities: List[float] = []
    frontier_returns: List[float] = []

    def annual_return_for_weights(w: np.ndarray) -> float:
        return float((w @ mean_daily) * TRADING_DAYS)

    def annual_vol_for_weights(w: np.ndarray) -> float:
        var_daily = float(w.T @ cov_daily @ w)
        return float(np.sqrt(max(var_daily, 0.0)) * np.sqrt(TRADING_DAYS))

    init = np.full(n_assets, 1.0 / n_assets)

    for target in targets:
        target_daily = target / TRADING_DAYS

        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: float(w @ mean_daily) - target_daily},
        )

        def objective(w: np.ndarray) -> float:
            return annual_vol_for_weights(w)

        res = minimize(
            objective,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not res.success:
            continue

        w = np.array(res.x, dtype=float)
        vol = annual_vol_for_weights(w)
        ret = annual_return_for_weights(w)

        if np.isfinite(vol) and np.isfinite(ret):
            volatilities.append(float(vol))
            frontier_returns.append(float(ret))

    return volatilities, frontier_returns