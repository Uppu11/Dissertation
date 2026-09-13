"""
Minimal Streamlit front-end for the calibrated, cost-aware rain tool.
This is the whole point of Week 2: a person can open a page, enter a day,
and get a calibrated probability, a cost-based decision, and the reasons.

    pip install streamlit
    python3 rainfall_tool.py train      # produces model/rain_tool.joblib
    streamlit run app_streamlit.py

The app is a thin shell — ALL the logic lives in RainTool.predict(), the same
function the CLI and any API would call. That separation (one inference contract,
many front-ends) is what makes it a tool rather than a notebook.
"""
import streamlit as st
from rainfall_tool import RainTool, FEATURES, EXAMPLE_DAY

st.set_page_config(page_title="Rain tool — calibrated & cost-aware", page_icon="🌧️")
st.title("🌧️ Will it rain tomorrow?")
st.caption("Calibrated probability · cost-based decision · per-prediction reasons")

tool = RainTool()  # loads model/rain_tool.joblib once

st.sidebar.header("Today's weather")
vals = {}
NUMERIC = [f for f in FEATURES if f != "RainToday"]
for f in NUMERIC:
    vals[f] = st.sidebar.number_input(f, value=float(EXAMPLE_DAY[f]))
vals["RainToday"] = 1 if st.sidebar.checkbox("Rain today?", value=bool(EXAMPLE_DAY["RainToday"])) else 0

if st.button("Assess", type="primary"):
    res = tool.predict(vals)
    p = res["rain_probability"]
    st.metric("Calibrated rain probability", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    if res["decision"].startswith("ISSUE"):
        st.error(f"**{res['decision']}** — probability {p:.0%} ≥ threshold {res['threshold']:.0%}")
    else:
        st.success(f"**{res['decision']}** — probability {p:.0%} < threshold {res['threshold']:.0%}")
    st.caption(res["cost_assumption"])
    st.subheader("Why")
    for r in res["top_reasons"]:
        st.write(f"- **{r['factor']}** ({r['value']}) {r['effect']}  ·  SHAP {r['shap']:+.2f}")
    st.caption(res["note"])
