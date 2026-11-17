import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from packaging.version import Version

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# APP SETUP
# ============================================================
TICKER = "MRK"

st.set_page_config(
    page_title="Merck & Co., Inc. (MRK) Equity Valuation Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Merck & Co., Inc. (MRK) Equity Valuation Lab")
st.caption("Created by Khulan Ganzorig | MS in Business Analytics, University at Buffalo")
st.markdown(
    "An equity research project covering financial statement analysis, DCF valuation, "
    "comparable company analysis, sensitivity analysis, and scenario modeling."
)
st.divider()

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Streamlit compatibility helper
try:
    if Version(st.__version__) >= Version("1.51.0"):
        STRETCH = {"width": "stretch"}
    else:
        STRETCH = {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}


def show_df(obj):
    st.dataframe(obj, **STRETCH)


def show_chart(fig):
    st.plotly_chart(fig, **STRETCH)


# ============================================================
# SAFE HELPERS
# ============================================================
def number(value, default=np.nan):
    """Convert a scalar-like value to a finite float; otherwise return default."""
    try:
        if value is None:
            return default
        if isinstance(value, (pd.Series, pd.DataFrame, list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = np.asarray(value).reshape(-1)[0]
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def first_valid(*values):
    for value in values:
        value = number(value)
        if np.isfinite(value):
            return value
    return np.nan


def divide(a, b):
    a, b = number(a), number(b)
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def money(value):
    value = number(value)
    return "N/A" if not np.isfinite(value) else f"${value:,.2f}"


def percent(value):
    value = number(value)
    return "N/A" if not np.isfinite(value) else f"{value:.1%}"


def pick(df, *labels, default=None):
