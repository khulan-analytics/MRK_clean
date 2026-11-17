# =====================================================================
# Stock Valuation Dashboard (default ticker: ABBV)
# Part 1  Fundamentals and free cash flow
# Part 2  DCF valuation (with WACC build-up)
# Part 3  Comparable-company valuation
# Part 4  WACC x terminal-growth sensitivity
# Part 5  Extension: Bear/Base/Bull scenarios + football field
# Educational use only. Not investment advice.
# =====================================================================
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from packaging.version import Version

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Valuation Dashboard", layout="wide")

# Streamlit replaced use_container_width with width="stretch" in newer releases.
# This works on both old and new versions.
if Version(st.__version__) >= Version("1.51.0"):
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def to_num(x):
    """Convert to float; return NaN for None / non-numeric / inf."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fmt_money(x):
    x = to_num(x)
    if not np.isfinite(x):
        return "N/A"
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def fmt_b(x):
    """Format a dollar amount in billions."""
    x = to_num(x)
    if not np.isfinite(x):
        return "N/A"
    return f"-${abs(x) / 1e9:,.2f}B" if x < 0 else f"${x / 1e9:,.2f}B"


def fmt_pct(x, digits=2):
    x = to_num(x)
    return "N/A" if not np.isfinite(x) else f"{x:.{digits}%}"


def get_row(df, candidates):
    """First matching row of a yfinance statement as a date-sorted Series, plus the label used."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    for name in candidates:
        if name in df.index:
            s = df.loc[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = pd.to_numeric(s, errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index(), name
    return None, None


def latest(series):
    """Most recent non-null value of a Series (NaN if unavailable)."""
    if series is None:
        return np.nan
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


# ---------------------------------------------------------------------
# Valuation functions
# ---------------------------------------------------------------------
def calculate_dcf(base_fcf, growth, terminal_growth, wacc, years, cash, debt, shares):
    """Two-stage DCF. Returns None if inputs are invalid (WACC must exceed terminal growth)."""
    base_fcf, growth = to_num(base_fcf), to_num(growth)
    terminal_growth, wacc = to_num(terminal_growth), to_num(wacc)
    shares = to_num(shares)
    cash = to_num(cash)
    debt = to_num(debt)
    cash = 0.0 if not np.isfinite(cash) else cash
    debt = 0.0 if not np.isfinite(debt) else debt

    if not all(np.isfinite(v) for v in (base_fcf, growth, terminal_growth, wacc, shares)):
        return None
    if shares <= 0 or wacc <= 0 or wacc <= terminal_growth or years < 1:
        return None

    t = np.arange(1, int(years) + 1)
    fcf = base_fcf * (1 + growth) ** t
    disc = (1 + wacc) ** t
    pv = fcf / disc
    terminal_value = fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / disc[-1]
    ev = pv.sum() + pv_terminal
    equity = ev + cash - debt
    return {
        "projection": pd.DataFrame({
            "Year": t,
            "Projected FCF": fcf,
            "Discount factor": 1 / disc,
            "PV of FCF": pv,
        }),
        "pv_fcf": float(pv.sum()),
        "terminal_value": float(terminal_value),
        "pv_terminal": float(pv_terminal),
        "ev": float(ev),
        "equity": float(equity),
        "intrinsic": float(equity / shares),
    }


def compute_wacc(beta, rf, erp, interest_expense, total_debt, market_cap, tax_rate):
    """CAPM cost of equity, after-tax cost of debt, market-value weights."""
    ke = rf + beta * erp

    kd, kd_source = np.nan, "Interest expense / total debt"
    interest_expense, total_debt = to_num(interest_expense), to_num(total_debt)
    if np.isfinite(interest_expense) and np.isfinite(total_debt) and total_debt > 0:
        kd = abs(interest_expense) / total_debt
    if not np.isfinite(kd):
        kd, kd_source = rf + 0.015, "Fallback: risk-free rate + 1.5%"
    kd = float(np.clip(kd, 0.01, 0.12))

    after_tax_kd = kd * (1 - tax_rate)

    market_cap = to_num(market_cap)
    if np.isfinite(market_cap) and market_cap > 0 and np.isfinite(total_debt) and total_debt >= 0:
        we = market_cap / (market_cap + total_debt)
        weights_source = "Market cap and total debt"
    else:
        we, weights_source = 1.0, "Fallback: 100% equity (market cap or debt unavailable)"
    wd = 1 - we

    return {
        "ke": ke, "kd": kd, "kd_source": kd_source, "after_tax_kd": after_tax_kd,
        "tax_rate": tax_rate, "we": we, "wd": wd, "weights_source": weights_source,
        "wacc": we * ke + wd * after_tax_kd,
    }


def implied_value(kind, multiple, metric, debt, cash, shares):
    """Implied value per share from a peer multiple."""
    if not (np.isfinite(multiple) and np.isfinite(metric)):
        return np.nan
    if kind == "pe":
        return multiple * metric  # metric = EPS
    if not (np.isfinite(shares) and shares > 0):
        return np.nan
    return (multiple * metric - debt + cash) / shares  # metric = EBITDA or revenue


# ---------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading Yahoo Finance data...")
def load_company(ticker):
    t = yf.Ticker(ticker)
    out = {}
    try:
        out["info"] = t.info or {}
    except Exception:
        out["info"] = {}
    for key, attr in [("cashflow", "cashflow"), ("balance", "balance_sheet"), ("income", "income_stmt")]:
        try:
            out[key] = getattr(t, attr)
        except Exception:
            out[key] = pd.DataFrame()
    try:
        h = t.history(period="5d")["Close"].dropna()
        out["last_close"] = float(h.iloc[-1]) if not h.empty else np.nan
    except Exception:
        out["last_close"] = np.nan
    return out


@st.cache_data(ttl=3600, show_spinner="Loading peer multiples...")
def load_peer_multiples(tickers):
    rows = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info or {}
        except Exception:
            info = {}
        rows.append({
            "Ticker": tk,
            "P/E": to_num(info.get("trailingPE")),
            "EV/EBITDA": to_num(info.get("enterpriseToEbitda")),
            "EV/Revenue": to_num(info.get("enterpriseToRevenue")),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_risk_free():
    """10-year Treasury yield from Yahoo (^TNX, quoted in percent). NaN if unavailable."""
    try:
        h = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        v = float(h.iloc[-1]) / 100
        if 0 < v < 0.15:
            return v
    except Exception:
        pass
    return np.nan


# ---------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker", "ABBV").upper().strip()
peer_text = st.sidebar.text_input("Peer tickers (comma-separated)", "LLY, MRK, PFE, BMY, AMGN, JNJ")
peers = tuple(p.strip().upper() for p in peer_text.split(",") if p.strip() and p.strip().upper() != ticker)

st.sidebar.subheader("DCF assumptions")
years = st.sidebar.slider("Projection years", 3, 10, 5)
fcf_growth = st.sidebar.slider("Annual FCF growth (%)", -5.0, 15.0, 5.0, 0.5) / 100
terminal_growth = st.sidebar.slider("Terminal growth (%)", 0.0, 4.0, 2.5, 0.25) / 100

st.sidebar.subheader("WACC inputs")
rf_fetched = load_risk_free()
rf_default = round(rf_fetched * 100, 2) if np.isfinite(rf_fetched) else 4.25
rf = st.sidebar.number_input("Risk-free rate (%)", 0.0, 15.0, rf_default, 0.05) / 100
erp = st.sidebar.number_input("Equity risk premium (%)", 0.0, 12.0, 5.0, 0.25) / 100
override_wacc = st.sidebar.checkbox("Override WACC manually")
wacc_manual = st.sidebar.number_input("Manual WACC (%)", 1.0, 25.0, 8.0, 0.25) / 100

st.sidebar.subheader("Balance-sheet items")
override_bs = st.sidebar.checkbox("Override cash and debt ($ billions)")
cash_manual = st.sidebar.number_input("Cash ($B)", 0.0, 500.0, 5.0, 0.5) * 1e9
debt_manual = st.sidebar.number_input("Debt ($B)", 0.0, 1000.0, 60.0, 0.5) * 1e9

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
st.title(f"{ticker} Valuation Dashboard")
st.caption("Educational use only. Data from Yahoo Finance (yfinance) and may be incomplete or delayed. Not investment advice.")

data = load_company(ticker)
info, cashflow, balance, income = data["info"], data["cashflow"], data["balance"], data["income"]

cfo_s, cfo_label = get_row(cashflow, [
    "Operating Cash Flow", "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities"])
capex_s, capex_label = get_row(cashflow, [
    "Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"])

if cfo_s is None or capex_s is None:
    st.error("Could not find operating cash flow and capital expenditure in Yahoo Finance data for this ticker.")
    st.stop()

# Assignment-defined free cash flow: operating cash flow minus capital expenditures
fcf_s = (cfo_s - capex_s.abs()).dropna()
if fcf_s.empty:
    st.error("Free cash flow could not be computed from the available statements.")
    st.stop()

last3 = fcf_s.tail(3)
base_fcf = float(last3.mean())

# Price, shares, market cap
price = to_num(info.get("currentPrice"))
if not np.isfinite(price):
    price = to_num(info.get("regularMarketPrice"))
if not np.isfinite(price):
    price = to_num(data.get("last_close"))

shares = to_num(info.get("sharesOutstanding"))
if not np.isfinite(shares):
    sh_s, _ = get_row(balance, ["Ordinary Shares Number", "Share Issued"])
    shares = latest(sh_s)
if not np.isfinite(shares):
    sh_s, _ = get_row(income, ["Diluted Average Shares", "Basic Average Shares"])
    shares = latest(sh_s)

market_cap = to_num(info.get("marketCap"))
if not np.isfinite(market_cap) and np.isfinite(price) and np.isfinite(shares):
    market_cap = price * shares

beta = to_num(info.get("beta"))
beta_note = ""
if not np.isfinite(beta):
    beta, beta_note = 1.0, "Beta unavailable from Yahoo; 1.0 used."

# Balance sheet items
cash_s, cash_label = get_row(balance, [
    "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
debt_s, debt_label = get_row(balance, ["Total Debt"])
cash_bs, debt_bs = latest(cash_s), latest(debt_s)

if override_bs:
    cash_assumption, debt_assumption = cash_manual, debt_manual
else:
    cash_assumption = 0.0 if not np.isfinite(cash_bs) else cash_bs
    debt_assumption = 0.0 if not np.isfinite(debt_bs) else debt_bs

# Tax rate and interest expense
int_s, int_label = get_row(income, ["Interest Expense", "Interest Expense Non Operating"])
tax_s, _ = get_row(income, ["Tax Provision"])
pretax_s, _ = get_row(income, ["Pretax Income"])
tax_rate = 0.21
if tax_s is not None and pretax_s is not None:
    t_val, p_val = latest(tax_s), latest(pretax_s)
    if np.isfinite(t_val) and np.isfinite(p_val) and p_val > 0:
        tax_rate = float(np.clip(t_val / p_val, 0.0, 0.35))

wacc_calc = compute_wacc(beta, rf, erp, latest(int_s), debt_assumption, market_cap, tax_rate)
selected_wacc = wacc_manual if override_wacc else wacc_calc["wacc"]

# =====================================================================
# PART 1: FUNDAMENTALS AND FREE CASH FLOW
# =====================================================================
st.header("Part 1: Fundamentals and Free Cash Flow")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", fmt_money(price))
c2.metric("Market cap", fmt_b(market_cap))
c3.metric("Beta", f"{beta:.2f}")
c4.metric("Shares outstanding", f"{shares / 1e9:,.3f}B" if np.isfinite(shares) else "N/A")

fcf_table = pd.DataFrame({
    "Operating cash flow": cfo_s,
    "CapEx (absolute)": capex_s.abs(),
    "Free cash flow": cfo_s - capex_s.abs(),
}).dropna(how="all").sort_index(ascending=False)
fcf_table.index = fcf_table.index.year
fcf_table.index.name = "Fiscal year"
st.dataframe(fcf_table.style.format(lambda v: fmt_b(v), na_rep="N/A"), **STRETCH)

fig_fcf = go.Figure(go.Bar(
    x=[str(d.year) for d in fcf_s.index],
    y=(fcf_s / 1e9).tolist(),
    text=[fmt_b(v) for v in fcf_s],
    textposition="outside",
))
fig_fcf.update_layout(title="Annual free cash flow (CFO minus CapEx)",
                      xaxis_title="Fiscal year", yaxis_title="$ billions", yaxis_tickprefix="$")
st.plotly_chart(fig_fcf, **STRETCH)

st.success(f"Base FCF (average of the latest {len(last3)} annual observations): {fmt_b(base_fcf)}")
if len(last3) < 3:
    st.warning("Fewer than three annual FCF observations are available; the average uses what exists.")
if base_fcf <= 0:
    st.warning("Base FCF is not positive, so a DCF on this cash flow is not meaningful.")

with st.expander("Data diagnostics"):
    st.write({
        "Operating cash flow row": cfo_label,
        "CapEx row": capex_label,
        "Cash row": cash_label,
        "Debt row": debt_label,
        "Interest expense row": int_label,
        "Latest cash used": fmt_b(cash_assumption),
        "Latest debt used": fmt_b(debt_assumption),
    })

# =====================================================================
# PART 2: DCF VALUATION
# =====================================================================
st.header("Part 2: DCF Valuation")

st.subheader("WACC build-up")
wacc_table = pd.DataFrame([
    ("Risk-free rate", fmt_pct(rf)),
    ("Equity risk premium", fmt_pct(erp)),
    ("Beta", f"{beta:.2f}"),
    ("Cost of equity (CAPM)", fmt_pct(wacc_calc["ke"])),
    ("Pre-tax cost of debt", f"{fmt_pct(wacc_calc['kd'])} ({wacc_calc['kd_source']})"),
    ("Tax rate", fmt_pct(wacc_calc["tax_rate"])),
    ("After-tax cost of debt", fmt_pct(wacc_calc["after_tax_kd"])),
    ("Equity weight", f"{fmt_pct(wacc_calc['we'])} ({wacc_calc['weights_source']})"),
    ("Debt weight", fmt_pct(wacc_calc["wd"])),
    ("Computed WACC", fmt_pct(wacc_calc["wacc"])),
    ("WACC used in the model", fmt_pct(selected_wacc) + (" (manual override)" if override_wacc else "")),
], columns=["Component", "Value"])
st.dataframe(wacc_table, hide_index=True, **STRETCH)
if beta_note:
    st.warning(beta_note)

dcf = calculate_dcf(base_fcf, fcf_growth, terminal_growth, selected_wacc, years,
                    cash_assumption, debt_assumption, shares)

if dcf is None:
    st.error("DCF could not be computed. WACC must exceed terminal growth, and base FCF, WACC and shares must be valid.")
else:
    proj = dcf["projection"].copy()
    st.dataframe(
        proj.style.format({
            "Projected FCF": lambda v: fmt_b(v),
            "Discount factor": "{:.4f}",
            "PV of FCF": lambda v: fmt_b(v),
        }),
        hide_index=True, **STRETCH)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PV of projected FCF", fmt_b(dcf["pv_fcf"]))
    m2.metric("PV of terminal value", fmt_b(dcf["pv_terminal"]))
    m3.metric("Enterprise value", fmt_b(dcf["ev"]))
    m4.metric("Equity value", fmt_b(dcf["equity"]))

    upside = (dcf["intrinsic"] / price - 1) if np.isfinite(price) and price > 0 else np.nan
    st.metric("Intrinsic value per share", fmt_money(dcf["intrinsic"]),
              delta=f"{upside:+.1%} vs market price" if np.isfinite(upside) else None)
    st.write(
        f"Equity value = enterprise value + cash ({fmt_b(cash_assumption)}) - debt ({fmt_b(debt_assumption)}). "
        f"Terminal value is {dcf['pv_terminal'] / dcf['ev']:.0%} of enterprise value."
    )

    fig_dcf = go.Figure()
    fig_dcf.add_trace(go.Bar(x=proj["Year"], y=proj["Projected FCF"] / 1e9, name="Projected FCF"))
    fig_dcf.add_trace(go.Bar(x=proj["Year"], y=proj["PV of FCF"] / 1e9, name="PV of FCF"))
    fig_dcf.update_layout(barmode="group", title="Projected vs discounted free cash flow",
                          xaxis_title="Projection year", yaxis_title="$ billions", yaxis_tickprefix="$")
    st.plotly_chart(fig_dcf, **STRETCH)

# =====================================================================
# PART 3: COMPARABLE COMPANIES
# =====================================================================
st.header("Part 3: Comparable-Company Valuation")

comp_detail = pd.DataFrame()
if not peers:
    st.info("Enter at least one peer ticker in the sidebar.")
else:
    peer_df = load_peer_multiples(peers)
    st.dataframe(peer_df.style.format({c: "{:.1f}x" for c in ["P/E", "EV/EBITDA", "EV/Revenue"]}, na_rep="N/A"),
                 hide_index=True, **STRETCH)

    eps = to_num(info.get("trailingEps"))
    ebitda = to_num(info.get("ebitda"))
    revenue = to_num(info.get("totalRevenue"))
    methods = [
        ("P/E", "P/E", eps, "pe"),
        ("EV/EBITDA", "EV/EBITDA", ebitda, "ev"),
        ("EV/Revenue", "EV/Revenue", revenue, "ev"),
    ]
    rows = []
    for label, col, metric, kind in methods:
        mult = peer_df[col].astype(float)
        mult = mult[np.isfinite(mult) & (mult > 0)]  # exclude missing, zero, negative
        if mult.empty or not np.isfinite(metric) or metric <= 0:
            rows.append({"Method": label, "Peers used": len(mult), "Median multiple": np.nan,
                         "Implied value/share": np.nan, "Low value/share": np.nan, "High value/share": np.nan})
            continue
        med, p25, p75 = float(mult.median()), float(mult.quantile(0.25)), float(mult.quantile(0.75))
        rows.append({
            "Method": label,
            "Peers used": len(mult),
            "Median multiple": med,
            "Implied value/share": implied_value(kind, med, metric, debt_assumption, cash_assumption, shares),
            "Low value/share": implied_value(kind, p25, metric, debt_assumption, cash_assumption, shares),
            "High value/share": implied_value(kind, p75, metric, debt_assumption, cash_assumption, shares),
        })
    comp_detail = pd.DataFrame(rows)
    st.dataframe(
        comp_detail.style.format({
            "Median multiple": "{:.1f}x",
            "Implied value/share": lambda v: fmt_money(v),
            "Low value/share": lambda v: fmt_money(v),
            "High value/share": lambda v: fmt_money(v),
        }, na_rep="N/A"),
        hide_index=True, **STRETCH)
    st.caption("Low and high are the 25th and 75th percentile peer multiples. Medians exclude missing, zero and negative multiples.")

# =====================================================================
# PART 4: SENSITIVITY (WACC x TERMINAL GROWTH)
# =====================================================================
st.header("Part 4: Sensitivity Analysis")

price_val = to_num(price)

if np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0 and np.isfinite(selected_wacc):
    wacc_grid = np.round(selected_wacc + np.arange(-0.02, 0.0201, 0.005), 6)
    tg_grid = np.round(terminal_growth + np.arange(-0.01, 0.0101, 0.005), 6)
    tg_grid = tg_grid[tg_grid >= 0]

    values = []
    for w in wacc_grid:
        row = []
        for g in tg_grid:
            res = calculate_dcf(base_fcf, fcf_growth, g, w, years, cash_assumption, debt_assumption, shares)
            row.append(res["intrinsic"] if res else np.nan)
        values.append(row)

    sensitivity = pd.DataFrame(
        values,
        index=[f"{x:.2%}" for x in wacc_grid],
        columns=[f"{x:.2%}" for x in tg_grid],
    )
    sensitivity.index.name = "WACC"
    sensitivity.columns.name = "Terminal growth"

    st.dataframe(
        sensitivity.style.format("${:,.2f}", na_rep="N/A").background_gradient(cmap="RdYlGn", axis=None),
        **STRETCH)

    heat_kwargs = {"zmid": price_val} if np.isfinite(price_val) and price_val > 0 else {}
    heat = go.Figure(data=go.Heatmap(
        z=sensitivity.to_numpy(dtype=float),
        x=sensitivity.columns.tolist(),
        y=sensitivity.index.tolist(),
        colorscale="RdYlGn",
        colorbar={"title": "$ / share"},
        hovertemplate="WACC: %{y}<br>Terminal growth: %{x}<br>Intrinsic value: $%{z:,.2f}<extra></extra>",
        **heat_kwargs,
    ))
    heat.update_layout(
        title="Intrinsic value per share: WACC vs terminal growth"
              + (" (green = above market price)" if heat_kwargs else ""),
        xaxis_title="Terminal growth",
        yaxis_title="WACC",
    )
    st.plotly_chart(heat, **STRETCH)

    base_row, base_col = f"{selected_wacc:.2%}", f"{terminal_growth:.2%}"
    if base_row in sensitivity.index and base_col in sensitivity.columns:
        st.success(f"Base-case grid value: {fmt_money(sensitivity.loc[base_row, base_col])} per share.")
    valid = sensitivity.to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size:
        st.write(f"Across valid grid cells, estimated value ranges from {fmt_money(valid.min())} to {fmt_money(valid.max())} per share.")
    st.write("**Interpretation:** Lower WACC increases present value; higher terminal growth increases terminal value. "
             "The model becomes especially sensitive when WACC approaches terminal growth.")
else:
    st.info("Sensitivity analysis requires valid base FCF, shares outstanding and WACC.")

# =====================================================================
# PART 5: EXTENSION - SCENARIOS + FOOTBALL FIELD
# =====================================================================
st.header("Part 5: Extension: Scenario Analysis and Valuation Football Field")
st.write("Scenario assumptions are illustrative stress tests around the selected base case, not forecasts or investment recommendations.")


def run_scenario(g, w, tg):
    """DCF for one scenario; None if inputs are invalid (WACC must exceed terminal growth)."""
    if not (np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0 and w > tg):
        return None
    return calculate_dcf(base_fcf, g, tg, w, years, cash_assumption, debt_assumption, shares)


# Bear/Bull keep WACC at least 1 pt above terminal growth so a scenario never
# silently becomes N/A, and long-run growth is capped at 3%.
bear_tg = max(0.0, terminal_growth - 0.005)
bull_tg = max(terminal_growth, min(terminal_growth + 0.005, 0.03))
scenario_specs = [
    ("Bear", fcf_growth - 0.03, selected_wacc + 0.015, bear_tg),
    ("Base", fcf_growth, selected_wacc, terminal_growth),
    ("Bull", fcf_growth + 0.03, max(selected_wacc - 0.01, bull_tg + 0.01), bull_tg),
]

scenario_rows = []
for name, g, w, tg in scenario_specs:
    result = run_scenario(g, w, tg)
    value = result["intrinsic"] if result else np.nan
    has_price = np.isfinite(value) and np.isfinite(price_val) and price_val > 0
    scenario_rows.append({
        "Scenario": name,
        "FCF growth": g,
        "WACC": w,
        "Terminal growth": tg,
        "Intrinsic value/share": value,
        "vs. market price": (value / price_val - 1) if has_price else np.nan,
    })

scenario_df = pd.DataFrame(scenario_rows)
st.dataframe(
    scenario_df.style.format({
        "FCF growth": "{:.1%}",
        "WACC": "{:.1%}",
        "Terminal growth": "{:.1%}",
        "Intrinsic value/share": lambda v: fmt_money(v),
        "vs. market price": "{:+.1%}",
    }, na_rep="N/A"),
    hide_index=True, **STRETCH)

scen_value = scenario_df.set_index("Scenario")["Intrinsic value/share"]
base_value = scen_value.get("Base", np.nan)
if dcf and np.isfinite(base_value) and abs(base_value - dcf["intrinsic"]) > 0.01:
    st.warning("The Base scenario does not match the Part 2 DCF value. Check that both use the same inputs.")

# Football field: horizontal ranges per method, market price as a vertical line
field = []
bear_v, bull_v = scen_value.get("Bear", np.nan), scen_value.get("Bull", np.nan)
if np.isfinite(bear_v) and np.isfinite(bull_v):
    field.append({"Method": "DCF (bear to bull)", "Low": min(bear_v, bull_v),
                  "High": max(bear_v, bull_v), "Mid": base_value})

if isinstance(comp_detail, pd.DataFrame) and not comp_detail.empty:
    for _, row in comp_detail.iterrows():
        mid = to_num(row.get("Implied value/share"))
        if not (np.isfinite(mid) and mid > 0):
            continue
        low, high = to_num(row.get("Low value/share")), to_num(row.get("High value/share"))
        if not (np.isfinite(low) and np.isfinite(high) and low > 0):
            low = high = mid
        field.append({"Method": f"Comps: {row['Method']}", "Low": min(low, high),
                      "High": max(low, high), "Mid": mid})

if field:
    st.subheader("Valuation football field")
    method_names = [r["Method"] for r in field]
    fig = go.Figure()
    for r in field:
        if r["High"] > r["Low"]:
            fig.add_trace(go.Bar(
                y=[r["Method"]], x=[r["High"] - r["Low"]], base=[r["Low"]],
                orientation="h",
                text=[f"{fmt_money(r['Low'])} to {fmt_money(r['High'])}"],
                textposition="inside",
                hovertemplate=(f"{r['Method']}<br>Low: {fmt_money(r['Low'])}"
                               f"<br>High: {fmt_money(r['High'])}<extra></extra>"),
                showlegend=False,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[r["Mid"]], y=[r["Method"]], mode="markers+text",
                marker={"symbol": "diamond", "size": 14},
                text=[fmt_money(r["Mid"])], textposition="middle right",
                hovertemplate=f"{r['Method']}: {fmt_money(r['Mid'])}<extra></extra>",
                showlegend=False,
            ))
    if np.isfinite(price_val) and price_val > 0:
        fig.add_vline(x=price_val, line_dash="dash", line_color="black",
                      annotation_text=f"Market price {fmt_money(price_val)}", annotation_position="top")
    fig.update_yaxes(categoryorder="array", categoryarray=method_names[::-1])
    fig.update_layout(
        title="Valuation football field: DCF range vs comparable valuation vs market price",
        xaxis_title="Value per share (USD)", xaxis_tickprefix="$", yaxis_title="", bargap=0.4,
    )
    st.plotly_chart(fig, **STRETCH)
    st.caption("Bars show low-to-high ranges (DCF: bear to bull; comps: 25th to 75th percentile peer multiple). "
               "Diamonds are single-point estimates. The dashed line is the current market price.")
else:
    st.info("The football field will appear when at least one valuation method returns a valid value.")

st.markdown("---")
st.subheader("Key assumptions and limitations")
st.markdown(
    """
    - Free cash flow is defined as operating cash flow minus capital expenditures; the base is the average of the latest three **available annual observations**.
    - WACC uses CAPM cost of equity and market-value weights when market capitalization and debt are available.
    - The DCF requires WACC to exceed perpetual growth.
    - Bear and Bull cases shift FCF growth by 3 points, WACC by +1.5 / -1.0 points and terminal growth by -0.5 / +0.5 points (long-run growth capped at 3%).
    - Comparable-company medians exclude missing, zero and negative trading multiples.
    - Peer multiples and market prices are point-in-time market data and may change.
    - Yahoo Finance data can contain missing or differently named line items; review the diagnostics and source statements.
    - The assignment-defined CFO-minus-CapEx cash flow may not be equivalent to unlevered FCFF. This affects interpretation when discounted using WACC.
    """
)
