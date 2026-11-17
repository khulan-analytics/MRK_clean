                )
                row.append(result["intrinsic"] if result else np.nan)
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
        use_container_width=True,
    )

    heat = go.Figure(data=go.Heatmap(
        z=sensitivity.to_numpy(dtype=float),
        x=sensitivity.columns.tolist(),
        y=sensitivity.index.tolist(),
        colorscale="RdYlGn",
        colorbar={"title": "$ / share"},
        hovertemplate="WACC: %{y}<br>Terminal growth: %{x}<br>Intrinsic value: $%{z:,.2f}<extra></extra>",
    ))
    heat.update_layout(
        title="Intrinsic Value per Share — WACC vs Terminal Growth",
        xaxis_title="Terminal growth",
        yaxis_title="WACC",
    )
    st.plotly_chart(heat, use_container_width=True)

    base_row = f"{selected_wacc:.2%}"
    base_col = f"{terminal_growth:.2%}"
    if base_row in sensitivity.index and base_col in sensitivity.columns:
        st.success(f"Base-case grid value: {fmt_money(sensitivity.loc[base_row, base_col])} per share.")
    valid = sensitivity.to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size:
        st.write(f"Across valid grid cells, estimated value ranges from {fmt_money(valid.min())} to {fmt_money(valid.max())} per share.")
    st.write("**Interpretation:** Lower WACC increases present value; higher terminal growth increases terminal value. The model becomes especially sensitive when WACC approaches terminal growth.")
else:
    st.info("Sensitivity analysis requires valid base FCF, shares outstanding and WACC.")


# ---------------------------------------------------------------------
# PART 5 — EXTENSION: SCENARIOS + FOOTBALL FIELD
# ---------------------------------------------------------------------
st.header("Part 5 — Extension: Scenario Analysis and Valuation Football Field")
st.write("Scenario assumptions are illustrative stress tests around the selected base case, not forecasts or investment recommendations.")

scenario_rows = []
scenario_specs = [
    ("Bear", fcf_growth - 0.03, selected_wacc + 0.015, max(0.0, terminal_growth - 0.005)),
    ("Base", fcf_growth, selected_wacc, terminal_growth),
    ("Bull", fcf_growth + 0.03, max(0.01, selected_wacc - 0.01), terminal_growth + 0.005),
]
for name, g, w, tg in scenario_specs:
    result = calculate_dcf(
        base_fcf, g, tg, w, years,
        cash_assumption, debt_assumption, shares
    ) if np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0 and tg < w else None
    scenario_rows.append({
        "Scenario": name,
        "FCF growth": g,
        "WACC": w,
        "Terminal growth": tg,
        "Intrinsic value/share": result["intrinsic"] if result else np.nan,
    })

scenario_df = pd.DataFrame(scenario_rows)
st.dataframe(
    scenario_df.style.format({
        "FCF growth": "{:.1%}",
        "WACC": "{:.1%}",
        "Terminal growth": "{:.1%}",
        "Intrinsic value/share": "${:,.2f}",
    }, na_rep="N/A"),
    use_container_width=True,
)

# Build football field from DCF base case and peer methods.
football = []
if dcf:
    football.append({"Method": "DCF", "Value": dcf["intrinsic"]})
if "comp_detail" in globals() and not comp_detail.empty:
    for _, row in comp_detail.iterrows():
        val = to_num(row.get("Implied value/share"))
        if np.isfinite(val) and val > 0:
            football.append({"Method": str(row["Method"]), "Value": val})
if np.isfinite(price) and price > 0:
    football.append({"Method": "Market price", "Value": price})

football_df = pd.DataFrame(football)
if not football_df.empty:
    st.subheader("Valuation football field")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=football_df["Method"],
        y=football_df["Value"],
        text=[fmt_money(v) for v in football_df["Value"]],
        textposition="outside",
        name="Value per share",
    ))
    fig.update_layout(
        title="DCF vs Comparable Valuation vs Market Price",
        xaxis_title="Valuation method",
        yaxis_title="Value per share (USD)",
        yaxis_tickprefix="$",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("The football field will appear when at least one valuation method returns a valid value.")

st.markdown("---")
st.subheader("Key assumptions and limitations")
st.markdown(
    """
    - The three-year average uses the latest three **available annual FCF observations**.
    - WACC uses CAPM cost of equity and market-value weights when market capitalization and debt are available.
    - The DCF requires WACC to exceed perpetual growth.
    - Comparable-company medians exclude missing, zero and negative trading multiples.
    - Peer multiples and market prices are point-in-time market data and may change.
    - Yahoo Finance data can contain missing or differently named line items; review the diagnostics and source statements.
    - The assignment-defined CFO-minus-CapEx cash flow may not be equivalent to unlevered FCFF. This affects interpretation when discounted using WACC.
    """
)
