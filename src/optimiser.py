from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def _annualized_return_and_volatility(
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compute annualized (return, volatility) for a non-short portfolio.
    """
    mean_daily = returns.mean().values  # shape (n_assets,)
    cov_daily = returns.cov().values  # shape (n_assets, n_assets)

    port_return_daily = float(np.dot(weights, mean_daily))
    port_return_annual = port_return_daily * TRADING_DAYS

    port_variance_daily = float(weights.T @ cov_daily @ weights)
    port_volatility_annual = float(np.sqrt(max(port_variance_daily, 0.0)) * np.sqrt(TRADING_DAYS))
    return port_return_annual, port_volatility_annual


def _sharpe_ratio(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    annual_ret, annual_vol = _annualized_return_and_volatility(weights, returns)
    if annual_vol == 0:
        return np.nan
    return (annual_ret - risk_free_rate) / annual_vol


def random_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int,
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Generate random non-negative weight portfolios (weights sum to 1).

    Returns DataFrame columns:
    - Return (annualized)
    - Volatility (annualized)
    - Sharpe (annualized Sharpe vs risk_free_rate)
    - one column per ticker containing portfolio weights
    """
    if n_portfolios <= 0:
        raise ValueError("n_portfolios must be positive.")
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    tickers = list(returns.columns)
    n_assets = len(tickers)

    mean_daily = returns.mean().values  # (n_assets,)
    cov_daily = returns.cov().values  # (n_assets, n_assets)

    rng = np.random.default_rng()
    weights = rng.random((n_portfolios, n_assets))
    weights = weights / weights.sum(axis=1, keepdims=True)  # simplex, no shorting

    # Vectorized portfolio annual return
    port_return_annual = (weights @ mean_daily) * TRADING_DAYS  # (n_portfolios,)

    # Vectorized portfolio variance: w^T C w
    port_variance_daily = np.einsum("ij,jk,ik->i", weights, cov_daily, weights)  # (n_portfolios,)
    port_volatility_annual = np.sqrt(np.maximum(port_variance_daily, 0.0)) * np.sqrt(TRADING_DAYS)  # (n_portfolios,)

    sharpe = (port_return_annual - risk_free_rate) / port_volatility_annual
    # Avoid divide-by-zero artifacts
    sharpe = np.where(np.isfinite(sharpe), sharpe, np.nan)

    df = pd.DataFrame(
        {
            "Return": port_return_annual,
            "Volatility": port_volatility_annual,
            "Sharpe": sharpe,
        }
    )
    for i, t in enumerate(tickers):
        df[t] = weights[:, i]
    return df


def max_sharpe_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float,
) -> Dict[str, np.ndarray | float]:
    """
    Maximise Sharpe ratio subject to:
    - weights sum to 1
    - 0 <= weights <= 1 (no short selling)
    """
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    n_assets = returns.shape[1]
    init = np.full(n_assets, 1.0 / n_assets)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w: np.ndarray) -> float:
        sharpe = _sharpe_ratio(w, returns, risk_free_rate)
        if not np.isfinite(sharpe):
            return 1e6
        return -float(sharpe)  # maximize Sharpe by minimizing negative Sharpe

    res = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    weights = np.array(res.x, dtype=float)

    annual_ret, annual_vol = _annualized_return_and_volatility(weights, returns)
    sharpe = _sharpe_ratio(weights, returns, risk_free_rate)

    return {
        "weights": weights,
        "return": annual_ret,
        "volatility": annual_vol,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
    }


def min_volatility_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float,
) -> Dict[str, np.ndarray | float]:
    """
    Minimise portfolio volatility subject to:
    - weights sum to 1
    - 0 <= weights <= 1 (no short selling)
    """
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    n_assets = returns.shape[1]
    init = np.full(n_assets, 1.0 / n_assets)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w: np.ndarray) -> float:
        _, annual_vol = _annualized_return_and_volatility(w, returns)
        return float(annual_vol)

    res = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    weights = np.array(res.x, dtype=float)

    annual_ret, annual_vol = _annualized_return_and_volatility(weights, returns)
    sharpe = _sharpe_ratio(weights, returns, risk_free_rate)

    return {
        "weights": weights,
        "return": annual_ret,
        "volatility": annual_vol,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
    }


def equal_weight_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float,
) -> Dict[str, np.ndarray | float]:
    """
    Compute the equal-weight portfolio stats.
    """
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    n_assets = returns.shape[1]
    weights = np.full(n_assets, 1.0 / n_assets, dtype=float)

    annual_ret, annual_vol = _annualized_return_and_volatility(weights, returns)
    sharpe = _sharpe_ratio(weights, returns, risk_free_rate)

    return {
        "weights": weights,
        "return": annual_ret,
        "volatility": annual_vol,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
    }