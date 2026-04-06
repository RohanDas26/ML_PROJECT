import streamlit as st
import pandas as pd
import numpy as np
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import run_sector_pipeline, load_config
from src.data.loader import load_raw_data, ALL_SECTORS
import logging
logging.getLogger().setLevel(logging.ERROR)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Energy Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* App font */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

  /* Remove default Streamlit padding */
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

  /* Metric cards */
  [data-testid="stMetric"] {
      background: #1e2635;
      border: 1px solid #2d3748;
      border-radius: 8px;
      padding: 16px 20px;
  }
  [data-testid="stMetricValue"]  { color: #93c5fd !important; font-size: 1.6rem !important; }
  [data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.85rem !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-size: 0.95rem;
    padding-bottom: 8px;
  }
  .stTabs [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
  }

  /* Table */
  [data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #111827; }
  [data-testid="stSidebar"] label { color: #94a3b8 !important; }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #e2e8f0 !important; }

  /* Divider */
  hr { border-color: #1e2635; }

  /* Result card */
  .result-card {
      background: #1e2635;
      border-left: 5px solid var(--c);
      border-radius: 8px;
      padding: 24px 28px;
      margin-top: 20px;
  }
  .result-card h1 { color: var(--c); margin: 0 0 4px 0; font-size: 3.2rem; }
  .result-card h3 { color: var(--c); margin: 0 0 8px 0; font-weight: 600; }
  .result-card p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Data (cached) ────────────────────────────────────────────────────────────
@st.cache_data
def get_base_data():
    cfg = load_config("config/config.yaml")
    dc  = cfg["data"]
    df  = load_raw_data(dc["raw_path"], sheet_name=dc.get("sheet_name","Sheet1"), skiprows=dc.get("skiprows",0))
    return df, cfg

df_raw, config = get_base_data()

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("US Energy Consumption Forecasting")
st.caption("Cross-validated ML pipeline — results in original Trillion BTU units")
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    selected_sector = st.selectbox("Sector", ["All Sectors"] + ALL_SECTORS)
    select_features = st.checkbox("Lasso Feature Selection", value=False)
    forecast_horizon = st.slider("Forecast Horizon (months)", 0, 24, 12)
    quick_mode = st.checkbox("Quick Mode (demo)", value=True,
                             help="Skips LSTM, Optuna, and Stacking. Runs in ~60-90s instead of 10+ minutes.")

    st.divider()
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

    if "run_cfg" in st.session_state:
        st.caption(f"Last run: {st.session_state['run_cfg']}")

# ─── Cache-key helper ─────────────────────────────────────────────────────────
def make_key(sector, feat, horizon):
    return hashlib.md5(f"{sector}|{feat}|{horizon}".encode()).hexdigest()[:8]

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Model Evaluation & Forecast", "Stress Test", "Custom Prediction"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if run_btn:
        run_key = make_key(selected_sector, select_features, forecast_horizon)
        cached  = st.session_state.get("results", {})

        if cached.get("key") == run_key:
            # Results already present — no re-training needed
            st.toast("Results loaded from cache (no changes detected)", icon="")
        else:
            sectors_to_run = ALL_SECTORS if selected_sector == "All Sectors" else [selected_sector]

            class _Log:
                def info(self, *a, **k): pass
                def warning(self, *a, **k): pass
                def debug(self, *a, **k): pass
                def error(self, *a, **k): pass

            all_res = {}
            with st.status("Running pipeline...", expanded=True) as status:
                for i, sec in enumerate(sectors_to_run):
                    st.write(f"Training models — {sec} ({i+1}/{len(sectors_to_run)})")
                    all_res[sec] = run_sector_pipeline(
                        df=df_raw.copy(), target_sector=sec, cfg=config,
                        log=_Log(), exog_df=None,
                        select_features=select_features,
                        forecast_horizon=forecast_horizon,
                        quick_mode=quick_mode,
                    )
                    st.write(f"  Done — best: {all_res[sec]['results'].iloc[0]['Model']} "
                             f"RMSE={all_res[sec]['results'].iloc[0]['RMSE']:.2f}")
                status.update(label="Pipeline complete", state="complete")

            st.session_state["results"] = {"key": run_key, "data": all_res, "sectors": sectors_to_run}
            st.session_state["run_cfg"] = f"{selected_sector} | lasso={select_features} | horizon={forecast_horizon}"

    if "results" in st.session_state:
        # Base paper data (from Results/tables/base_paper_replica_comparison.json)
        BASE_PAPER = {
            "Residential": {
                "zscale":  {"Ridge": 0.0815, "GradientBoosting": 0.1416, "RandomForest": 0.1573},
                "honest":  {"Ridge": 26.10,  "GradientBoosting": 45.33,  "RandomForest": 50.66},
            },
            "Commercial": {
                "zscale":  {"Ridge": 0.0669, "GradientBoosting": 0.1490, "RandomForest": 0.1656},
                "honest":  {"Ridge": 9.77,   "GradientBoosting": 21.78,  "RandomForest": 24.20},
            },
            "Industrial": {
                "zscale":  {"Ridge": 0.5720, "GradientBoosting": 0.4582, "RandomForest": 0.5358},
                "honest":  {"Ridge": 81.34,  "GradientBoosting": 65.14,  "RandomForest": 75.90},
            },
            "Transportation": {
                "zscale":  {"Ridge": 0.2646, "GradientBoosting": 0.2704, "RandomForest": 0.2898},
                "honest":  {"Ridge": 77.33,  "GradientBoosting": 78.54,  "RandomForest": 85.27},
            },
        }

        res_store = st.session_state["results"]
        for sec in res_store["sectors"]:
            sec_data   = res_store["data"][sec]
            results_df = sec_data["results"]
            best       = results_df.iloc[0]

            st.subheader(sec)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Model",   best["Model"])
            c2.metric("RMSE (T-BTU)", f"{best['RMSE']:.2f}")
            c3.metric("MAE",          f"{best['MAE']:.2f}")
            c4.metric("R²",           f"{best['R2']:.4f}")

            disp_cols = ["Model","RMSE","MAE","R2","Train_MSE","Overfit_Ratio","Time"]
            existing  = [c for c in disp_cols if c in results_df.columns]
            st.dataframe(results_df[existing].style.highlight_min(subset=["RMSE"], color="#064e3b"),
                         use_container_width=True)

            # ── Comparison Tables ─────────────────────────────────────────────
            if sec in BASE_PAPER:
                bp = BASE_PAPER[sec]

                # Compute our Z-scaled RMSE and MAE for top 3 models
                sec_mean = df_raw[sec].mean()
                sec_std  = df_raw[sec].std()

                # Table 1: Honest Trillion BTU comparison
                our_top3 = results_df.head(3)[["Model","RMSE","MAE","R2"]].copy()
                our_top3.columns = ["Our Model", "Our RMSE (T-BTU)", "Our MAE (T-BTU)", "Our R²"]

                bp_honest_rows = [
                    {"Base Paper Model": m, "BP RMSE (T-BTU) [Decoded]": v}
                    for m, v in bp["honest"].items()
                ]
                bp_honest_df = pd.DataFrame(bp_honest_rows)

                st.markdown("**Comparison 1 — Honest Trillion BTU Units (fair, apples-to-apples)**")
                col_l, col_r = st.columns(2)
                with col_l:
                    st.caption("Our Models")
                    st.dataframe(our_top3.style.highlight_min(subset=["Our RMSE (T-BTU)"], color="#064e3b"),
                                 use_container_width=True, hide_index=True)
                with col_r:
                    st.caption("Base Paper (decoded from Z-scale to Trillion BTU via population σ)")
                    st.dataframe(bp_honest_df.style.highlight_min(subset=["BP RMSE (T-BTU) [Decoded]"], color="#7c2d12"),
                                 use_container_width=True, hide_index=True)
                st.error(
                    "**Why is the base paper lower here?** Their model STILL uses leaky features "
                    "(`total_energy[t]`, `res_com_ratio[t]`, `sector_std[t]`) — current month data that "
                    "includes the answer. This table only changes the *measurement unit* from Z-score to "
                    "Trillion BTU. It does NOT fix the leakage. In real-world deployment, their model would "
                    "be completely unusable because you cannot compute `total_energy[t]` until the entire "
                    "month is over — which is exactly when you would already know the answer you were "
                    "trying to predict. **Our model uses only past data and is the only one that can "
                    "actually work in production.**"
                )

                # Table 2: Z-scaled comparison
                our_zscale_rows = []
                for _, row in results_df.head(3).iterrows():
                    our_zscale_rows.append({
                        "Our Model":           row["Model"],
                        "Our RMSE (Z-scaled)": round(row["RMSE"] / sec_std, 4),
                        "Our MAE (Z-scaled)":  round(row["MAE"]  / sec_std, 4),
                    })
                our_zscale_df = pd.DataFrame(our_zscale_rows)

                bp_zscale_rows = [
                    {"Base Paper Model": m, "BP RMSE (Z-scaled, as reported)": v}
                    for m, v in bp["zscale"].items()
                ]
                bp_zscale_df = pd.DataFrame(bp_zscale_rows)

                st.markdown("**Comparison 2 — Z-Scaled Space (our honest RMSE ÷ σ vs. base paper's published Z-RMSE)**")
                col_l2, col_r2 = st.columns(2)
                with col_l2:
                    st.caption("Our Models (converted to Z-scale for comparison only)")
                    st.dataframe(our_zscale_df.style.highlight_min(subset=["Our RMSE (Z-scaled)"], color="#064e3b"),
                                 use_container_width=True, hide_index=True)
                with col_r2:
                    st.caption("Base Paper (published Z-scaled values — evaluated on LEAKY features)")
                    st.dataframe(bp_zscale_df.style.highlight_min(subset=["BP RMSE (Z-scaled, as reported)"], color="#7c2d12"),
                                 use_container_width=True, hide_index=True)

            if forecast_horizon > 0 and sec_data.get("forecast"):
                hist = df_raw[["Month", sec]].tail(48)
                fore = pd.DataFrame(sec_data["forecast"])

                fig = go.Figure()
                fig.add_scatter(x=hist["Month"], y=hist[sec],
                                mode="lines", name="Historical", line=dict(color="#60a5fa", width=2))
                fig.add_scatter(x=[hist.iloc[-1]["Month"], fore.iloc[0]["Month"]],
                                y=[hist.iloc[-1][sec],     fore.iloc[0]["Forecast"]],
                                mode="lines", showlegend=False, line=dict(color="#f87171", width=2, dash="dash"))
                fig.add_scatter(x=fore["Month"], y=fore["Forecast"],
                                mode="lines+markers", name="Forecast",
                                line=dict(color="#f87171", width=2, dash="dash"), marker=dict(size=5))
                fig.update_layout(
                    xaxis_title="Date", yaxis_title="Trillion BTU",
                    hovermode="x unified", template="plotly_dark",
                    margin=dict(t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
    else:
        st.info("Configure the pipeline in the sidebar and click Run Pipeline.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stress Test
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Historical Shock Simulation")
    st.caption("Inject a percentage variance into the last 3 months of history. The model re-learns from the shocked state and generates a new forecast.")

    sim_sec   = st.selectbox("Sector", ALL_SECTORS, index=1, key="sim_sec")
    shock_pct = st.slider("Shock (%)", -50.0, 50.0, 20.0, 1.0)

    if st.button("Run Simulation", type="primary"):
        class _Log:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def debug(self, *a, **k): pass
            def error(self, *a, **k): pass

        df_sh = df_raw.copy()
        df_sh.loc[df_sh.index[-3:], sim_sec] *= (1.0 + shock_pct / 100.0)

        with st.status("Running baseline and shocked pipeline...", expanded=True) as s:
            st.write("Baseline pipeline...")
            r_base = run_sector_pipeline(df=df_raw.copy(), target_sector=sim_sec, cfg=config, log=_Log(), forecast_horizon=12, quick_mode=True)
            st.write("Shocked pipeline...")
            r_shock = run_sector_pipeline(df=df_sh, target_sector=sim_sec, cfg=config, log=_Log(), forecast_horizon=12, quick_mode=True)
            s.update(label="Simulation complete", state="complete")

        hist   = df_raw[["Month", sim_sec]].tail(24)
        hist_s = df_sh[["Month", sim_sec]].tail(24)
        f_base = pd.DataFrame(r_base["forecast"])
        f_shock = pd.DataFrame(r_shock["forecast"])

        fig2 = go.Figure()
        fig2.add_scatter(x=hist["Month"],           y=hist[sim_sec],           name="History",          mode="lines",        line=dict(color="#64748b", width=2))
        fig2.add_scatter(x=hist_s["Month"].tail(4), y=hist_s[sim_sec].tail(4), name=f"Shock ({shock_pct:+.0f}%)", mode="lines+markers", line=dict(color="#fbbf24", width=3))
        fig2.add_scatter(x=f_base["Month"],         y=f_base["Forecast"],      name="Baseline Forecast",mode="lines",        line=dict(color="#60a5fa", width=2, dash="dot"))
        fig2.add_scatter(x=f_shock["Month"],        y=f_shock["Forecast"],     name="Shocked Forecast", mode="lines+markers",line=dict(color="#f87171", width=2))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Trillion BTU",
                           hovermode="x unified", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Custom Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Custom Date Prediction")
    st.caption("Specify a target date and the most recent historical volumetrics. The system reconstructs the full feature matrix and returns a single prediction.")

    ca, cb = st.columns(2)
    with ca:
        cp_sector = st.selectbox("Sector", ALL_SECTORS, key="cp_sec")
        cp_year   = st.number_input("Year",  min_value=2000, max_value=2030, value=2025)
        cp_month  = st.selectbox("Month", list(range(1,13)), format_func=lambda x: datetime(2000,x,1).strftime("%B"))
    with cb:
        avg_val   = float(df_raw[cp_sector].mean())
        cp_t1     = st.number_input("Previous month (t-1)",         value=round(avg_val, 2))
        cp_t2     = st.number_input("Two months ago (t-2)",         value=round(avg_val, 2))
        cp_t12    = st.number_input("Same month last year (t-12)",  value=round(avg_val, 2))

    if st.button("Predict", type="primary"):
        with st.status("Building feature matrix and predicting...", expanded=True) as s:
            try:
                # Get pipeline objects — run a short pipeline if not already cached
                cache_key = f"cp_{cp_sector}"
                if cache_key not in st.session_state:
                    class _Log:
                        def info(self, *a, **k): pass
                        def warning(self, *a, **k): pass
                        def debug(self, *a, **k): pass
                        def error(self, *a, **k): pass
                    st.write(f"Training models for {cp_sector}...")
                    res = run_sector_pipeline(df=df_raw.copy(), target_sector=cp_sector,
                                             cfg=config, log=_Log(), forecast_horizon=0,
                                             quick_mode=True)
                    st.session_state[cache_key] = res
                else:
                    res = st.session_state[cache_key]

                st.write("Computing feature vector...")
                preproc      = res["preprocessor"]
                feat_fn      = res["feature_fn"]
                feat_cfg     = res["feat_cfg"]
                feature_cols = res["feature_cols"]
                best_name    = res["results"].iloc[0]["Model"]
                best_model   = res["best_models"][best_name]

                target_date = pd.Timestamp(cp_year, cp_month, 1)
                t1  = target_date - pd.DateOffset(months=1)
                t2  = target_date - pd.DateOffset(months=2)
                t12 = target_date - pd.DateOffset(months=12)

                date_range = pd.date_range(start=df_raw["Month"].min(), end=target_date, freq="MS")
                test_df = pd.DataFrame({"Month": date_range})
                test_df = pd.merge(test_df, df_raw, on="Month", how="left").ffill()

                test_df.loc[test_df["Month"] == t1,  cp_sector] = cp_t1
                test_df.loc[test_df["Month"] == t2,  cp_sector] = cp_t2
                test_df.loc[test_df["Month"] == t12, cp_sector] = cp_t12
                test_df.loc[test_df["Month"] == target_date, cp_sector] = np.nan

                feat_df  = feat_fn(test_df, cp_sector, **feat_cfg)
                trow     = feat_df[feat_df["Month"] == target_date]
                X_vals   = np.nan_to_num(trow[feature_cols].values, nan=0.0)
                X_sc     = preproc.scaler_X.transform(X_vals)
                y_sc     = best_model.predict(X_sc)
                y_btu    = max(0.0, preproc.inverse_transform_y(y_sc)[0])

                # Bracket classification
                h_avg, h_std = df_raw[cp_sector].mean(), df_raw[cp_sector].std()
                if y_btu < h_avg - 0.5 * h_std:
                    label, accent = "LOW CONSUMPTION",     "#4ade80"
                elif y_btu > h_avg + 0.5 * h_std:
                    label, accent = "HIGH CONSUMPTION",    "#f87171"
                else:
                    label, accent = "AVERAGE CONSUMPTION", "#60a5fa"

                season = ("Winter" if cp_month in [12,1,2] else
                          "Summer" if cp_month in [6,7,8]  else
                          "Spring" if cp_month in [3,4,5]  else "Fall")

                s.update(label="Prediction complete", state="complete")

                st.markdown(f"""
                <div style="background:#1e2635;border-left:6px solid {accent};
                            border-radius:8px;padding:24px 28px;margin-top:16px;">
                  <p style="color:#94a3b8;margin:0 0 4px 0;font-size:0.85rem;">
                    {cp_sector} — {datetime(cp_year,cp_month,1).strftime('%B %Y')} — {season}
                  </p>
                  <h1 style="color:{accent};margin:0;font-size:3rem;">{y_btu:,.2f}
                    <span style="font-size:1rem;color:#94a3b8;">Trillion BTU</span>
                  </h1>
                  <h3 style="color:{accent};margin:4px 0 0 0;font-weight:600;">{label}</h3>
                  <p style="color:#64748b;margin:10px 0 0 0;font-size:0.8rem;">
                    Algorithm: {best_name} &nbsp;|&nbsp; Historical avg: {h_avg:,.1f} ± {h_std:,.1f} T-BTU
                  </p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                s.update(label="Error", state="error")
                st.error(str(e))
