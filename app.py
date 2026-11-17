# ---------------------------------------------------------------------
# PART 5 — EXTENSION: SCENARIOS + FOOTBALL FIELD
# Drop-in replacement: paste over everything from the Part 5 header to the
# end of the file. Uses names already defined earlier in your script:
# base_fcf, fcf_growth, selected_wacc, terminal_growth, years, shares, price,
# cash_assumption, debt_assumption, dcf, calculate_dcf, to_num, fmt_money,
# and (optionally) comp_detail.
# ---------------------------------------------------------------------
st.header("Part 5 — Extension: Scenario Analysis and Valuation Football Field")
st.write("Scenario assumptions are illustrative stress tests around the selected base case, not forecasts or investment recommendations.")

price_val = to_num(price)


def run_scenario(g, w, tg):
    """DCF for one scenario; None if inputs are invalid (WACC must exceed terminal growth)."""
    if not (np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0 and w > tg):
        return None
    return calculate_dcf(base_fcf, g, tg, w, years, cash_assumption, debt_assumption, shares)


# Bear/Bull keep WACC at least 1 pt above terminal growth so a scenario never
# silently turns into N/A, and cap long-run growth at 3%.
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
        "Intrinsic value/share": "${:,.2f}",
        "vs. market price": "{:+.1%}",
    }, na_rep="N/A"),
    use_container_width=True,
    hide_index=True,
)

scen_value = scenario_df.set_index("Scenario")["Intrinsic value/share"]
base_value = scen_value.get("Base", np.nan)
if dcf and np.isfinite(base_value) and abs(base_value - dcf["intrinsic"]) > 0.01:
    st.warning("The Base scenario does not match the Part 2 DCF value. Check that both use the same inputs.")

# ---------------------------------------------------------------------
# Football field: horizontal RANGES per method, with the market price as a line
# ---------------------------------------------------------------------
field = []

bear_v, bull_v = scen_value.get("Bear", np.nan), scen_value.get("Bull", np.nan)
if np.isfinite(bear_v) and np.isfinite(bull_v):
    field.append({
        "Method": "DCF (bear to bull)",
        "Low": min(bear_v, bull_v),
        "High": max(bear_v, bull_v),
        "Mid": base_value,
    })

comp_df = globals().get("comp_detail")
if isinstance(comp_df, pd.DataFrame) and not comp_df.empty:
    for _, row in comp_df.iterrows():
        mid = to_num(row.get("Implied value/share"))
        if not (np.isfinite(mid) and mid > 0):
            continue
        # Optional columns; if you do not compute peer low/high, a single point is drawn.
        low = to_num(row.get("Low value/share"))
        high = to_num(row.get("High value/share"))
        if not (np.isfinite(low) and np.isfinite(high)):
            low = high = mid
        field.append({"Method": str(row["Method"]), "Low": low, "High": high, "Mid": mid})

if field:
    st.subheader("Valuation football field")
    methods = [r["Method"] for r in field]
    fig = go.Figure()
    for r in field:
        if r["High"] > r["Low"]:
            fig.add_trace(go.Bar(
                y=[r["Method"]],
                x=[r["High"] - r["Low"]],
                base=[r["Low"]],
                orientation="h",
                text=[f"{fmt_money(r['Low'])} – {fmt_money(r['High'])}"],
                textposition="inside",
                hovertemplate=(f"{r['Method']}<br>Low: {fmt_money(r['Low'])}"
                               f"<br>High: {fmt_money(r['High'])}<extra></extra>"),
                showlegend=False,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[r["Mid"]],
                y=[r["Method"]],
                mode="markers+text",
                marker={"symbol": "diamond", "size": 14},
                text=[fmt_money(r["Mid"])],
                textposition="middle right",
                hovertemplate=f"{r['Method']}: {fmt_money(r['Mid'])}<extra></extra>",
                showlegend=False,
            ))
    if np.isfinite(price_val) and price_val > 0:
        fig.add_vline(
            x=price_val, line_dash="dash", line_color="black",
            annotation_text=f"Market price {fmt_money(price_val)}",
            annotation_position="top",
        )
    fig.update_yaxes(categoryorder="array", categoryarray=methods[::-1])
    fig.update_layout(
        title="Valuation Football Field: DCF Range vs Comparable Valuation vs Market Price",
        xaxis_title="Value per share (USD)",
        xaxis_tickprefix="$",
        yaxis_title="",
        bargap=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars show low-to-high ranges; diamonds are single-point estimates (for example peer-median multiples); the dashed line is the current market price.")
else:
    st.info("The football field will appear when at least one valuation method returns a valid value.")

st.markdown("---")
st.subheader("Key assumptions and limitations")
st.markdown(
    """
    - The three-year average uses the latest three **available annual FCF observations**.
    - WACC uses CAPM cost of equity and market-value weights when market capitalization and debt are available.
    - The DCF requires WACC to exceed perpetual growth.
    - Bear and Bull cases shift FCF growth by 3 points, WACC by +1.5 / -1.0 points and terminal growth by -0.5 / +0.5 points (long-run growth capped at 3%).
    - Comparable-company medians exclude missing, zero and negative trading multiples.
    - Peer multiples and market prices are point-in-time market data and may change.
    - Yahoo Finance data can contain missing or differently named line items; review the diagnostics and source statements.
    - The assignment-defined CFO-minus-CapEx cash flow may not be equivalent to unlevered FCFF. This affects interpretation when discounted using WACC.
    """
)
