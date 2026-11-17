    x=scenario_table["Scenario"],
    y=scenario_table["Intrinsic Value / Share"],
    text=scenario_table["Intrinsic Value / Share"].map(money),
    textposition="outside",
    name="Scenario value",
))
if np.isfinite(price):
    scenario_chart.add_hline(y=price, line_dash="dash", annotation_text=f"Market price: {money(price)}")
scenario_chart.update_layout(
    title="Bull / Base / Bear valuation",
    yaxis_title="Intrinsic value per share ($)",
    showlegend=False,
)
show_chart(scenario_chart)
st.caption(
    "The scenario presets are illustrative assumptions, not forecasts. The Base preset is fixed "
    "(6% growth, 2.5% terminal growth) and may differ from your custom Part 2 assumptions; "
    "the Part 2 (custom) row shows them side by side."
)

# ============================================================
# VALUATION FOOTBALL FIELD
# ============================================================
st.subheader("Valuation football field")

def scenario_value_of(name):
    row = scenario_table.loc[scenario_table["Scenario"] == name, "Intrinsic Value / Share"]
    return number(row.iloc[0]) if not row.empty else np.nan


field = []
bear_value = scenario_value_of("Bear")
bull_value = scenario_value_of("Bull")
base_scenario_value = scenario_value_of("Base")
if np.isfinite(bear_value) and np.isfinite(bull_value):
    field.append({
        "Method": "DCF (Bear to Bull)",
        "Low": min(bear_value, bull_value),
        "High": max(bear_value, bull_value),
        "Mid": base_scenario_value,
    })

for row in comp_ranges:
    mid = number(row["Mid"])
    low = number(row["Low"])
    high = number(row["High"])
    if not np.isfinite(mid) or mid <= 0:
        continue
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= 0:
        low = high = mid
    field.append({
        "Method": f"Comps: {row['Method']}",
        "Low": min(low, high),
        "High": max(low, high),
        "Mid": mid,
    })

low_52 = number(info.get("fiftyTwoWeekLow"))
high_52 = number(info.get("fiftyTwoWeekHigh"))
if np.isfinite(low_52) and np.isfinite(high_52) and high_52 > low_52:
    field.append({"Method": "52-week trading range", "Low": low_52, "High": high_52, "Mid": price})

if field:
    method_names = [row["Method"] for row in field]
    football = go.Figure()
    for row in field:
        if row["High"] > row["Low"]:
            football.add_trace(go.Bar(
                y=[row["Method"]],
                x=[row["High"] - row["Low"]],
                base=[row["Low"]],
                orientation="h",
                text=[f"{money(row['Low'])} to {money(row['High'])}"],
                textposition="inside",
                hovertemplate=f"{row['Method']}<br>Low: {money(row['Low'])}<br>High: {money(row['High'])}<extra></extra>",
                showlegend=False,
            ))
        else:
            football.add_trace(go.Scatter(
                x=[row["Mid"]],
                y=[row["Method"]],
                mode="markers+text",
                marker={"symbol": "diamond", "size": 14},
                text=[money(row["Mid"])],
                textposition="middle right",
                hovertemplate=f"{row['Method']}: {money(row['Mid'])}<extra></extra>",
                showlegend=False,
            ))

    if np.isfinite(price) and price > 0:
        football.add_vline(
            x=price,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Market price {money(price)}",
            annotation_position="top",
        )
    football.update_yaxes(categoryorder="array", categoryarray=method_names[::-1])
    football.update_layout(
        title=f"{ticker} valuation football field",
        xaxis_title="Value per share (USD)",
        xaxis_tickprefix="$",
        yaxis_title="",
        bargap=0.4,
    )
    show_chart(football)
    st.caption(
        "Bars show low-to-high ranges: DCF is Bear to Bull; comps use the 25th to 75th "
        "percentile peer multiple; the 52-week range is the trading range. Diamonds are "
        "single-point estimates. The dashed line is the current market price."
    )
else:
    st.info("The football field appears once at least one valuation method returns a valid value.")

# ============================================================
# DISCLAIMER
# ============================================================
st.divider()
st.caption(
    "Educational use only, not investment advice. Yahoo Finance data may be incomplete or delayed. "
    "Check company filings and verify assumptions before relying on any valuation."
)
