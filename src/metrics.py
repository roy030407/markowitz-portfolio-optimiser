from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def portfolio_cumulative_returns(weights: Sequence[float] | np.ndarray, returns: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative returns series for a portfolio over time.
    """
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != returns.shape[1]:
        raise ValueError("weights must be a 1D vector matching returns columns.")

    portfolio_daily_returns = returns.to_numpy() @ w
    cumulative = (1.0 + pd.Series(portfolio_daily_returns, index=returns.index)).cumprod() - 1.0
    cumulative.name = "Cumulative Returns"
    return cumulative


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Maximum drawdown as a negative fraction (e.g., -0.25 means -25%).
    """
    if cumulative_returns is None or cumulative_returns.empty:
        raise ValueError("cumulative_returns must be a non-empty Series.")

    wealth = 1.0 + cumulative_returns.fillna(0.0)
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min())


def annual_return(returns: pd.DataFrame, weights: Sequence[float] | np.ndarray) -> float:
    """
    Annualised arithmetic return estimate from daily returns.
    """
    w = np.asarray(weights, dtype=float)
    portfolio_daily = returns.to_numpy() @ w
    return float(portfolio_daily.mean() * TRADING_DAYS)


def annual_volatility(returns: pd.DataFrame, weights: Sequence[float] | np.ndarray) -> float:
    """
    Annualised portfolio volatility from daily return covariance.
    """
    w = np.asarray(weights, dtype=float)
    cov_daily = returns.cov().to_numpy()
    var_daily = float(w.T @ cov_daily @ w)
    return float(np.sqrt(max(var_daily, 0.0)) * np.sqrt(TRADING_DAYS))


def sortino_ratio(
    returns: pd.DataFrame,
    weights: Sequence[float] | np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Annualised Sortino ratio using historical downside deviation.
    """
    w = np.asarray(weights, dtype=float)
    portfolio_daily = returns.to_numpy() @ w

    # Convert annual risk-free rate into a daily rate using compounding.
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0

    downside = portfolio_daily[portfolio_daily < rf_daily] - rf_daily
    if len(downside) == 0:
        return np.nan

    downside_deviation_daily = float(np.sqrt(np.mean(downside**2)))
    downside_deviation_annual = downside_deviation_daily * np.sqrt(TRADING_DAYS)
    if downside_deviation_annual == 0:
        return np.nan

    ann_ret = annual_return(returns, w)
    return float((ann_ret - risk_free_rate) / downside_deviation_annual)


def var_95(returns: pd.DataFrame, weights: Sequence[float] | np.ndarray) -> float:
    """
    95% historical Value at Risk (VaR) as a positive loss fraction.
    """
    w = np.asarray(weights, dtype=float)
    portfolio_daily = returns.to_numpy() @ w
    # Historical VaR: 5th percentile of daily portfolio returns.
    # `portfolio_daily` is a numpy array, so use numpy percentile.
    q = float(np.percentile(portfolio_daily, 5))  # worst 5%

    # VaR is typically reported as a positive loss number.
    return float(-q)