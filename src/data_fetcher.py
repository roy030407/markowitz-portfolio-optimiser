from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

def _to_ymd(d: date | datetime | str) -> str:
    """Convert input date-ish value to YYYY-MM-DD string."""
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return pd.to_datetime(d).date().isoformat()


def _extract_adj_close_from_yf(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """
    Normalize yfinance output into a DataFrame of Adj Close prices.

    Handles both:
    - multi-ticker (MultiIndex columns)
    - single ticker (flat columns)
    """
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    # MultiIndex case: e.g. ('Adj Close', 'RELIANCE.NS')
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("yfinance response missing 'Adj Close' field.")
        adj = raw["Adj Close"].copy()
        # Ensure columns are ticker symbols
        if not isinstance(adj, pd.DataFrame):
            adj = adj.to_frame()
        return adj

    # Single ticker case: flat columns include 'Adj Close'
    if "Adj Close" in raw.columns:
        if len(tickers) != 1:
            raise RuntimeError("Expected exactly 1 ticker for single-ticker response.")
        return raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    # If yfinance is configured differently, last-resort: return raw
    # (but most callers expect 'Adj Close')
    return raw


@st.cache_data(show_spinner=False)
def get_stock_data(
    tickers: Sequence[str],
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pd.DataFrame:
    """
    Download adjusted close prices for tickers, then compute daily returns.

    Returns a cleaned DataFrame of daily returns (pct_change, dropna).
    """
    tickers = list(dict.fromkeys([t for t in tickers if t]))
    if not tickers:
        raise ValueError("No tickers provided.")

    start_ymd = _to_ymd(start_date)
    end_ymd = _to_ymd(end_date)

    raw = yf.download(
        tickers,
        start=start_ymd,
        end=end_ymd,
        progress=False,
        group_by="column",
        actions=False,
        auto_adjust=False,
        threads=True,
    )

    prices = _extract_adj_close_from_yf(raw, tickers)
    if prices is None or prices.empty:
        raise RuntimeError("No price data returned from yfinance.")

    kept: List[str] = []
    for t in tickers:
        if t not in prices.columns:
            continue
        s = prices[t]
        # If the entire column is missing, treat as empty and skip.
        if s.dropna().empty:
            continue
        kept.append(t)

    if not kept:
        raise RuntimeError("No valid ticker series returned for the given date range.")

    prices = prices[kept].sort_index()
    returns = prices.pct_change().dropna(how="any")

    # Extra cleanliness: ensure there are no infinities.
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return returns


@st.cache_data(show_spinner=False)
def get_benchmark(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pd.Series:
    """
    Download benchmark data (^NSEI) and return daily returns.
    """
    start_ymd = _to_ymd(start_date)
    end_ymd = _to_ymd(end_date)

    raw = yf.download(
        "^NSEI",
        start=start_ymd,
        end=end_ymd,
        progress=False,
        group_by="column",
        actions=False,
        auto_adjust=False,
        threads=True,
    )

    prices = _extract_adj_close_from_yf(raw, tickers=["^NSEI"])
    if prices is None or prices.empty or "^NSEI" not in prices.columns:
        raise RuntimeError("No benchmark price data returned from yfinance.")

    series = prices["^NSEI"].sort_index()
    returns = series.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Give it a stable name for plotting.
    returns.name = "Nifty 50 (Benchmark)"
    return returns