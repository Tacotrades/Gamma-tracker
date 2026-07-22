"""
FlashAlpha Financial Modeling Hub — Streamlit Edition
======================================================
(GEX Bar Chart + NQ Futures Candlestick)

pip install streamlit requests pandas numpy plotly

Run with:
    streamlit run app.py

Auth:
    Set your API key as an environment variable before launching:
        export FLASHALPHA_API_KEY="your-key-here"      (macOS/Linux)
        setx FLASHALPHA_API_KEY "your-key-here"         (Windows)
    Never hardcode the key in source. This app reads it via
    os.environ.get("FLASHALPHA_API_KEY") only.

IMPORTANT NOTE ON THE FLASHALPHA API
-------------------------------------
I don't have verified documentation for FlashAlpha's actual endpoint paths,
auth header format, or JSON field names, so I can't guarantee the request/
response shapes below are 100% correct out of the box. Everything is
isolated inside the `FlashAlphaClient` class (bottom of the "DATA LAYER"
section) so that once you have their real API docs, you only need to edit
that one class — the Streamlit UI, chart rendering, caching, and error
handling all work independently of the exact wire format.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

API_KEY_ENV_VAR = "FLASHALPHA_API_KEY"
BASE_URL = "https://api.flashalpha.com/v1"  # placeholder — confirm against real docs
REQUEST_TIMEOUT = 15
NQ_TICKER = "NQ=F"  # CME E-mini Nasdaq 100 futures — exact string FlashAlpha expects


# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------

class FlashAlphaAPIError(Exception):
    """Raised for any FlashAlpha request failure (auth, rate limit, bad data)."""


@dataclass
class FlashAlphaClient:
    """
    Thin adapter around the FlashAlpha REST API.

    Edit the three `_fetch_*` methods below once you have real API docs —
    everything upstream (Streamlit UI, caching, charts) is decoupled from
    the exact endpoint paths / JSON schema.
    """
    api_key: Optional[str] = None
    base_url: str = BASE_URL
    timeout: int = REQUEST_TIMEOUT

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get(API_KEY_ENV_VAR)
        if not self.api_key:
            raise FlashAlphaAPIError(
                f"No API key found. Set the {API_KEY_ENV_VAR} environment "
                "variable before launching the app."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise FlashAlphaAPIError(f"Network error contacting FlashAlpha: {e}") from e

        if resp.status_code == 401:
            raise FlashAlphaAPIError("Unauthorized (401) — check your API key.")
        if resp.status_code == 429:
            raise FlashAlphaAPIError("Rate limited (429) — slow down requests / retry later.")
        if resp.status_code >= 500:
            raise FlashAlphaAPIError(f"FlashAlpha server error ({resp.status_code}).")
        if not resp.ok:
            raise FlashAlphaAPIError(f"Request failed ({resp.status_code}): {resp.text[:300]}")

        try:
            return resp.json()
        except ValueError as e:
            raise FlashAlphaAPIError(f"Malformed JSON response: {e}") from e

    # --- GEX (single/all expirations) --------------------------------------
    def fetch_gex(self, symbol: str, dte: Optional[int] = None) -> pd.DataFrame:
        """
        Returns columns: strike, call_gex, put_gex, net_gex, dte, expiration
        Adjust the endpoint path/params/field names to match real docs.
        """
        params = {"symbol": symbol}
        if dte is not None:
            params["dte"] = dte
        data = self._get("/options/gex", params=params)
        rows = data.get("data", data if isinstance(data, list) else [])
        if not rows:
            raise FlashAlphaAPIError(f"No GEX data returned for {symbol}.")
        df = pd.DataFrame(rows)
        required = {"strike", "call_gex", "put_gex", "dte", "expiration"}
        missing = required - set(df.columns)
        if missing:
            raise FlashAlphaAPIError(f"GEX response missing fields: {missing}")
        df["net_gex"] = df["call_gex"] - df["put_gex"].abs()
        return df.sort_values("strike").reset_index(drop=True)

    # --- OHLC candles --------------------------------------------------------
    def fetch_candles(self, ticker: str, interval: str = "5m", lookback: str = "5d") -> pd.DataFrame:
        """
        Returns columns: datetime, open, high, low, close, volume
        `ticker` must be passed exactly as FlashAlpha expects it, e.g. "NQ=F".
        requests handles URL-encoding of special characters like '=' automatically
        since we pass it via the `params` dict rather than string-concatenating
        it into the URL path.
        """
        params = {"ticker": ticker, "interval": interval, "range": lookback}
        data = self._get("/futures/candles", params=params)
        rows = data.get("data", data if isinstance(data, list) else [])
        if not rows:
            raise FlashAlphaAPIError(f"No candle data returned for {ticker}.")
        df = pd.DataFrame(rows)
        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise FlashAlphaAPIError(f"Candle response missing fields: {missing}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_client() -> FlashAlphaClient:
    return FlashAlphaClient()


@st.cache_data(ttl=60, show_spinner=False)
def cached_gex(symbol: str, dte: Optional[int]) -> pd.DataFrame:
    return get_client().fetch_gex(symbol, dte)


@st.cache_data(ttl=60, show_spinner=False)
def cached_candles(ticker: str, interval: str, lookback: str) -> pd.DataFrame:
    return get_client().fetch_candles(ticker, interval, lookback)


# ----------------------------------------------------------------------------
# CHART BUILDERS
# ----------------------------------------------------------------------------

def build_gex_bar_chart(df: pd.DataFrame, symbol: str, dte_label: str) -> go.Figure:
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["net_gex"]]
    fig = go.Figure(
        go.Bar(x=df["strike"], y=df["net_gex"], marker_color=colors, name="Net GEX")
    )
    fig.update_layout(
        title=f"{symbol} Gamma Exposure Profile ({dte_label})",
        xaxis_title="Strike",
        yaxis_title="Net Gamma Exposure",
        template="plotly_dark",
        bargap=0.15,
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    return fig


def build_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
        )
    )
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )
    return fig


# ----------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="FlashAlpha Modeling Hub", layout="wide")
st.title("📊 FlashAlpha Financial Modeling Hub")

with st.sidebar:
    st.header("Controls")
    model = st.radio(
        "Select a model",
        ["GEX Bar Chart", "NQ Futures Candlestick"],
    )
    symbol = st.text_input("Underlying symbol (for GEX models)", value="SPX")
    st.caption(f"API key loaded from ${API_KEY_ENV_VAR}: "
               f"{'✅ found' if os.environ.get(API_KEY_ENV_VAR) else '❌ missing'}")

# Guard: fail fast with a clear message rather than an ugly traceback
if not os.environ.get(API_KEY_ENV_VAR):
    st.error(
        f"Environment variable `{API_KEY_ENV_VAR}` is not set. "
        "Set it in your shell before launching Streamlit, e.g.\n\n"
        f"`export {API_KEY_ENV_VAR}=your-key-here`"
    )
    st.stop()

# --- Model 1: GEX Bar Chart --------------------------------------------------
if model == "GEX Bar Chart":
    st.subheader("Gamma Exposure (GEX) Bar Chart")
    dte_filter = st.slider("Filter by DTE (Days to Expiration)", 0, 60, 0)
    use_all_dte = st.checkbox("Show all expirations combined", value=False)

    if st.button("Fetch GEX", type="primary"):
        with st.spinner("Fetching GEX data from FlashAlpha..."):
            try:
                dte_arg = None if use_all_dte else dte_filter
                df = cached_gex(symbol, dte_arg)
            except FlashAlphaAPIError as e:
                st.error(f"FlashAlpha API error: {e}")
            else:
                label = "All expirations" if use_all_dte else f"{dte_filter} DTE"
                st.plotly_chart(build_gex_bar_chart(df, symbol, label), use_container_width=True)
                with st.expander("Raw data"):
                    st.dataframe(df, use_container_width=True)

# --- Model 2: NQ Futures Candlestick -----------------------------------------
else:
    st.subheader("NQ Futures Candlestick Chart")
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
    with col2:
        lookback = st.selectbox("Lookback", ["1d", "5d", "1mo", "3mo"], index=1)

    st.caption(f"Ticker sent to FlashAlpha: `{NQ_TICKER}` (passed via request params, "
               "so `requests` handles URL-encoding of the `=` character automatically).")

    if st.button("Fetch NQ Candles", type="primary"):
        with st.spinner("Fetching NQ=F futures data from FlashAlpha..."):
            try:
                df = cached_candles(NQ_TICKER, interval, lookback)
            except FlashAlphaAPIError as e:
                st.error(f"FlashAlpha API error: {e}")
            else:
                st.plotly_chart(build_candlestick_chart(df, NQ_TICKER), use_container_width=True)
                with st.expander("Raw data"):
                    st.dataframe(df, use_container_width=True)
