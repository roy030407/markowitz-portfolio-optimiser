import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datetime import date

from src.data_fetcher import get_benchmark, get_stock_data
from src.efficient_frontier import efficient_frontier_points
from src.metrics import (
    annual_return as metrics_annual_return,
    annual_volatility as metrics_annual_volatility,
    max_drawdown,
    portfolio_cumulative_returns,
    sortino_ratio,
    var_95,
)
from src.optimiser import (
    equal_weight_portfolio,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    random_portfolios,
)


st.set_page_config(
    page_title="Markowitz Portfolio Optimiser",
    layout="wide",
    page_icon="📊",
)

STOCK_OPTIONS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "WIPRO.NS",
    "ICICIBANK.NS",
    "BAJFINANCE.NS",
    "HINDUNILVR.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
]


def _plotly_bg(fig: go.Figure) -> go.Figure:
    """Apply consistent transparent backgrounds to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _weights_bar_chart(tickers: list[str], weights: np.ndarray, title: str) -> go.Figure:
    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    weights_df = weights_df.sort_values("Weight", ascending=False)

    fig = px.bar(
        weights_df,
        x="Weight",
        y="Ticker",
        orientation="h",
        title=title,
        template="plotly_dark",
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_tickformat=".0%",
    )
    return _plotly_bg(fig)


def _portfolio_metrics_table(
    returns: pd.DataFrame,
    benchmark_name: str,
    portfolios: dict[str, dict],
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Build a metrics table for the 3 core portfolios.
    """
    rows = [
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "VaR 95%",
    ]

    columns = list(portfolios.keys())
    out = pd.DataFrame(index=rows, columns=columns, dtype=float)

    for name in columns:
        weights = np.asarray(portfolios[name]["weights"], dtype=float)
        ann_ret = metrics_annual_return(returns, weights)
        ann_vol = metrics_annual_volatility(returns, weights)
        sharpe = float(portfolios[name]["sharpe"])
        sortino = sortino_ratio(returns, weights, risk_free_rate)
        cumulative = portfolio_cumulative_returns(weights, returns)
        mdd = max_drawdown(cumulative)  # negative fraction
        var95 = var_95(returns, weights)  # positive loss fraction

        out.loc["Annual Return", name] = ann_ret
        out.loc["Annual Volatility", name] = ann_vol
        out.loc["Sharpe Ratio", name] = sharpe
        out.loc["Sortino Ratio", name] = sortino
        out.loc["Max Drawdown", name] = mdd
        out.loc["VaR 95%", name] = var95

    return out


if "opt_results" not in st.session_state:
    st.session_state["opt_results"] = None


st.sidebar.header("Portfolio Settings")

selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    options=STOCK_OPTIONS,
    default=STOCK_OPTIONS,
)

start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date(2025, 1, 1))

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (India 10Y)",
    value=0.067,
    step=0.001,
    format="%.3f",
)

n_portfolios = st.sidebar.slider(
    "Number of random portfolios",
    min_value=1000,
    max_value=10000,
    value=3000,
    step=500,
)

optimisation_goal = st.sidebar.radio(
    "Optimisation goal",
    options=["Max Sharpe ⭐", "Min Volatility 🛡️", "Equal Weight ⚖️"],
    index=0,
)


clicked = st.sidebar.button("Optimise Portfolio")

st.title("📊 Markowitz Portfolio Optimiser")
st.subheader("Mean-Variance Optimisation on Indian NSE Stocks")

if clicked:
    if end_date <= start_date:
        st.error("`End Date` must be after `Start Date`.")
    else:
        with st.spinner("Fetching data and optimising..."):
            try:
                returns = get_stock_data(selected_stocks, start_date, end_date)
                benchmark_returns = get_benchmark(start_date, end_date)

                random_df = random_portfolios(returns, n_portfolios, risk_free_rate)

                max_sharpe = max_sharpe_portfolio(returns, risk_free_rate)
                min_vol = min_volatility_portfolio(returns, risk_free_rate)
                equal_w = equal_weight_portfolio(returns, risk_free_rate)

                ef_vols, ef_rets = efficient_frontier_points(
                    returns, risk_free_rate=risk_free_rate, n_points=50
                )

                # Choose which weights to highlight based on radio.
                if optimisation_goal == "Max Sharpe ⭐":
                    chosen = {"name": "Max Sharpe", "weights": max_sharpe["weights"]}
                elif optimisation_goal == "Min Volatility 🛡️":
                    chosen = {"name": "Min Volatility", "weights": min_vol["weights"]}
                else:
                    chosen = {"name": "Equal Weight", "weights": equal_w["weights"]}

                portfolios = {
                    "Max Sharpe": max_sharpe,
                    "Min Volatility": min_vol,
                    "Equal Weight": equal_w,
                }

                metrics_table = _portfolio_metrics_table(
                    returns=returns,
                    benchmark_name="Nifty 50",
                    portfolios=portfolios,
                    risk_free_rate=risk_free_rate,
                )

                cumulative_returns = {
                    "Max Sharpe": portfolio_cumulative_returns(
                        np.asarray(max_sharpe["weights"], dtype=float), returns
                    ),
                    "Min Volatility": portfolio_cumulative_returns(
                        np.asarray(min_vol["weights"], dtype=float), returns
                    ),
                    "Equal Weight": portfolio_cumulative_returns(
                        np.asarray(equal_w["weights"], dtype=float), returns
                    ),
                    "Nifty 50 (Benchmark)": (1.0 + benchmark_returns).cumprod() - 1.0,
                }

                corr_matrix = returns.corr()
                normalized_prices = (1.0 + returns).cumprod() * 100.0

                st.session_state["opt_results"] = {
                    "tickers": list(returns.columns),
                    "returns": returns,
                    "benchmark_returns": benchmark_returns,
                    "random_portfolios": random_df,
                    "max_sharpe": max_sharpe,
                    "min_volatility": min_vol,
                    "equal_weight": equal_w,
                    "efficient_frontier": {"volatility": ef_vols, "return": ef_rets},
                    "chosen": chosen,
                    "risk_free_rate": risk_free_rate,
                    "metrics_table": metrics_table,
                    "cumulative_returns": cumulative_returns,
                    "corr_matrix": corr_matrix,
                    "normalized_prices": normalized_prices,
                }
                st.success("Optimisation complete.")
            except Exception as e:
                st.session_state["opt_results"] = None
                st.error(f"Failed to optimise: {e}")


if not st.session_state["opt_results"]:
    st.info("Select stocks and click `Optimise Portfolio` to see results.")
    st.stop()


results = st.session_state["opt_results"]
tickers = results["tickers"]
returns = results["returns"]
benchmark_returns = results["benchmark_returns"]
random_df = results["random_portfolios"]
max_sharpe = results["max_sharpe"]
min_vol = results["min_volatility"]
equal_w = results["equal_weight"]
ef = results["efficient_frontier"]
chosen = results["chosen"]


tabs = st.tabs(["🎯 Efficient Frontier", "📈 Portfolio Metrics", "🔥 Stock Analysis", "🎮 What-If Simulator"])


with tabs[0]:
    st.subheader("Efficient Frontier — Random Portfolio Simulation")

    weight_cols = [t for t in tickers if t in random_df.columns]
    fig = px.scatter(
        random_df,
        x="Volatility",
        y="Return",
        color="Sharpe",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Efficient Frontier — Random Portfolio Simulation",
        hover_data=weight_cols,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    _plotly_bg(fig)

    # Efficient frontier overlay curve.
    fig.add_trace(
        go.Scatter(
            x=ef["volatility"],
            y=ef["return"],
            mode="lines",
            line=dict(color="white", width=2),
            name="Efficient Frontier",
        )
    )

    # Special portfolios.
    special_points = [
        ("Max Sharpe ⭐", "gold", max_sharpe),
        ("Min Volatility 🛡️", "cyan", min_vol),
        ("Equal Weight ⚖️", "orange", equal_w),
    ]
    for label, color, p in special_points:
        fig.add_trace(
            go.Scatter(
                x=[p["volatility"]],
                y=[p["return"]],
                mode="markers+text",
                text=[label],
                textposition="top center",
                marker=dict(size=20, color=color),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Volatility: %{x:.2%}<br>"
                    "Return: %{y:.2%}<br>"
                    "Sharpe: %{customdata[0]:.3f}<extra></extra>"
                ),
                customdata=[[p["sharpe"]]],
            )
        )

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("What am I looking at?"):
        st.write(
            "Each dot is a randomly sampled long-only portfolio. "
            "The x-axis is annual volatility, and the y-axis is annual return. "
            "Colour indicates the Sharpe ratio (risk-adjusted return). "
            "The white curve is the efficient frontier under a target-return constraint."
        )

    st.plotly_chart(
        _weights_bar_chart(tickers, np.asarray(chosen["weights"], dtype=float), title="Portfolio Weights"),
        use_container_width=True,
    )
    with st.expander("How to interpret the selected weights"):
        st.write(
            "The horizontal bars show the allocation for the selected optimisation goal "
            "(Max Sharpe, Min Volatility, or Equal Weight). "
            "Weights sum to 1 and are constrained to be non-negative (no short selling)."
        )


with tabs[1]:
    st.subheader("Portfolio Metrics")

    def _ensure_weights_vector(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != returns.shape[1]:
            raise ValueError(f"Weight vector length {w.shape[0]} != number of assets {returns.shape[1]}.")
        # Safety: ensure fully invested (should already sum to 1).
        s = float(np.sum(w))
        if s != 0.0:
            w = w / s
        return w

    # Ensure weights are 1D numpy arrays before passing to metrics functions.
    ms_weights = _ensure_weights_vector(max_sharpe["weights"])
    mv_weights = _ensure_weights_vector(min_vol["weights"])
    ew_weights = _ensure_weights_vector(equal_w["weights"])

    metrics_dict = {
        "Max Sharpe": {
            "Annual Return": f"{metrics_annual_return(returns, ms_weights) * 100:.2f}%",
            "Annual Volatility": f"{metrics_annual_volatility(returns, ms_weights) * 100:.2f}%",
            "Sharpe Ratio": f"{max_sharpe['sharpe']:.3f}",
            "Sortino Ratio": f"{sortino_ratio(returns, ms_weights, risk_free_rate):.3f}",
            "Max Drawdown": f"{max_drawdown(portfolio_cumulative_returns(ms_weights, returns)) * 100:.2f}%",
            "VaR 95%": f"{var_95(returns, ms_weights) * 100:.2f}%",
        },
        "Min Volatility": {
            "Annual Return": f"{metrics_annual_return(returns, mv_weights) * 100:.2f}%",
            "Annual Volatility": f"{metrics_annual_volatility(returns, mv_weights) * 100:.2f}%",
            "Sharpe Ratio": f"{min_vol['sharpe']:.3f}",
            "Sortino Ratio": f"{sortino_ratio(returns, mv_weights, risk_free_rate):.3f}",
            "Max Drawdown": f"{max_drawdown(portfolio_cumulative_returns(mv_weights, returns)) * 100:.2f}%",
            "VaR 95%": f"{var_95(returns, mv_weights) * 100:.2f}%",
        },
        "Equal Weight": {
            "Annual Return": f"{metrics_annual_return(returns, ew_weights) * 100:.2f}%",
            "Annual Volatility": f"{metrics_annual_volatility(returns, ew_weights) * 100:.2f}%",
            "Sharpe Ratio": f"{equal_w['sharpe']:.3f}",
            "Sortino Ratio": f"{sortino_ratio(returns, ew_weights, risk_free_rate):.3f}",
            "Max Drawdown": f"{max_drawdown(portfolio_cumulative_returns(ew_weights, returns)) * 100:.2f}%",
            "VaR 95%": f"{var_95(returns, ew_weights) * 100:.2f}%",
        },
    }

    df_metrics = pd.DataFrame(metrics_dict)
    st.dataframe(df_metrics, use_container_width=True)

    st.subheader("Cumulative Returns Backtest")

    # Portfolio cumulative returns for all three portfolios plus Nifty 50.
    fig = go.Figure()

    # Preserve trace order for consistent interpretation.
    for name in ["Max Sharpe", "Min Volatility", "Equal Weight", "Nifty 50 (Benchmark)"]:
        cum = results["cumulative_returns"][name]
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=name))

    fig.update_layout(
        template="plotly_dark",
        title="Cumulative Returns Backtest",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Series",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Financial explanation"):
        st.write(
            "This backtest is a historical simulation using the fitted portfolio weights "
            "over your selected date range. "
            "Cumulative returns reflect how an investor would have performed if weights "
            "were held constant (rebalanced only implicitly in the weight assumption). "
            "Risk/return ratios in the table summarise annualised mean/volatility and downside risk."
        )


with tabs[2]:
    st.subheader("Stock Analysis")

    st.header("Stock Return Correlation Matrix")
    corr = results["corr_matrix"]
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        template="plotly_dark",
        title="Stock Return Correlation Matrix",
    )
    _plotly_bg(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)
    with st.expander("What correlation tells you"):
        st.write(
            "Correlation measures how similarly two stocks move in returns. "
            "Diversification benefits come from combining assets that do not move together."
        )

    st.header("Normalised Stock Price Performance (Base=100)")
    normalized = results["normalized_prices"]
    normalized_reset = normalized.reset_index().rename(columns={"index": "Date"})
    fig_norm = px.line(
        normalized_reset,
        x="Date",
        y=tickers,
        template="plotly_dark",
        title="Normalised Stock Price Performance (Base=100)",
    )
    _plotly_bg(fig_norm)
    st.plotly_chart(fig_norm, use_container_width=True)
    with st.expander("Why normalise prices?"):
        st.write(
            "Normalising to a base of 100 makes it easier to compare relative performance "
            "across stocks that have different price levels."
        )

    st.header("Daily Return Distributions")
    fig_hist = go.Figure()
    for t in tickers:
        fig_hist.add_trace(
            go.Histogram(
                x=returns[t],
                name=t,
                opacity=0.55,
                nbinsx=40,
            )
        )
    fig_hist.update_layout(
        barmode="overlay",
        template="plotly_dark",
        title="Daily Return Distributions",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    with st.expander("How to read the histograms"):
        st.write(
            "The histograms show the distribution of each stock's daily returns over the selected period. "
            "Wider/shifted distributions can indicate higher volatility and/or different average return behaviour."
        )


with tabs[3]:
    st.subheader("What-If Simulator")

    if not tickers:
        st.info("No stocks selected.")
        st.stop()

    st.write("Adjust allocations (long-only). The weights are normalised to sum to 1.")

    # Default equal-weight slider values.
    default_percent = int(round(100 / len(tickers)))
    allocation_percents: list[int] = []

    # Layout sliders with simple vertical flow; Streamlit keys keep them stable.
    for t in tickers:
        key = f"sim_{t}"
        if key not in st.session_state:
            st.session_state[key] = default_percent
        allocation_percents.append(
            st.slider(
                f"Allocation for {t}",
                min_value=0,
                max_value=100,
                value=int(st.session_state[key]),
                step=1,
                key=key,
            )
        )

    perc_arr = np.asarray(allocation_percents, dtype=float)
    if perc_arr.sum() == 0:
        weights_sim = np.full(len(tickers), 1.0 / len(tickers), dtype=float)
    else:
        weights_sim = perc_arr / perc_arr.sum()  # normalise to sum=1

    ann_ret = metrics_annual_return(returns, weights_sim)
    ann_vol = metrics_annual_volatility(returns, weights_sim)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Return", f"{ann_ret:.2%}")
    c2.metric("Volatility", f"{ann_vol:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.3f}")

    pie_values = pd.Series(weights_sim, index=tickers).sort_values(ascending=False)
    pie_fig = px.pie(
        names=pie_values.index,
        values=pie_values.values,
        template="plotly_dark",
        title="Allocation Allocation Breakdown",
    )
    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    with st.expander("How to interpret the allocation pie"):
        st.write(
            "The pie chart shows the simulated weights across the selected stocks. "
            "Even if you choose slider values that don’t add to 100, the app normalises "
            "them so the final portfolio weights sum to 1."
        )

    with st.expander("How the simulator relates to optimisation"):
        max_sharpe_dict = max_sharpe
        max_sharpe_value = float(max_sharpe_dict.get("sharpe", np.nan))

        if not np.isfinite(max_sharpe_value) or not np.isfinite(sharpe):
            st.write("Sharpe comparison is unavailable for these settings.")
        else:
            if sharpe > max_sharpe_value:
                st.info("This simulated portfolio has a better Sharpe ratio than the Max Sharpe portfolio.")
            elif sharpe < max_sharpe_value:
                st.info("This simulated portfolio has a worse Sharpe ratio than the Max Sharpe portfolio.")
            else:
                st.info("This simulated portfolio is essentially tied with the Max Sharpe portfolio on Sharpe ratio.")

