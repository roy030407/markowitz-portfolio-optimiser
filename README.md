# Markowitz Portfolio Optimiser (Streamlit)

A Python web app that downloads adjusted close data for Indian NSE stocks, then performs mean-variance optimisation (Markowitz) to construct long-only portfolios.

## Features

- Random long-only portfolio simulation (for visual efficient frontier)
- Constrained optimisation with:
  - Max Sharpe
  - Min Volatility
  - Equal Weight
- Portfolio analytics:
  - Annual return / volatility
  - Sharpe and Sortino ratios
  - Max drawdown
  - Historical 95% VaR
- Historical cumulative returns backtest
- Stock analysis:
  - Correlation heatmap
  - Normalised price performance (base=100)
  - Daily return distributions
- What-If simulator: adjust allocations with live metrics

## Setup

1. Create/activate a virtual environment (recommended).
2. Install dependencies:
   - `pip install -r requirements.txt`

## Run the app

From the project root:

- `streamlit run app.py`

## Notes

- The app uses `yfinance` to download adjusted close prices and then converts them into daily return series.
- Optimisation assumes:
  - long-only portfolios (`weights >= 0`)
  - fully invested portfolios (`weights sum to 1`)
- Annualisation uses `252` trading days.
