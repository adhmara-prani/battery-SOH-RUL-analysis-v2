# """
# Battery Health Prediction — Streamlit Interface
# =================================================
# Loads the pre-trained GradientBoostingRegressor models (SOH & RUL) that were
# persisted by the notebook pipeline, lets a user upload a new CSV dataset with
# the expected feature columns, then displays:
#   • R² score for the uploaded data
#   • Top-5 most important features according to SHAP (TreeExplainer)
#   • SHAP summary plot

# The feature list and models come from  ../models/ (relative to notebooks/).
# """

# import os
# import sys

# import joblib
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import shap
# import streamlit as st
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # ---------------------------------------------------------------------------
# # Paths – resolve relative to *this* file so it works regardless of cwd
# # ---------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# SRC_DIR = os.path.join(BASE_DIR, "src")

# SOH_MODEL_PATH = os.path.join(MODELS_DIR, "soh_model.pkl")
# RUL_MODEL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")
# FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.pkl")
# SAMPLE_DATA_PATH = os.path.join(SRC_DIR, "final_data.csv")

# # ---------------------------------------------------------------------------
# # Page config
# # ---------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Battery Health Predictor",
#     page_icon="🔋",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------------------------------------------------------------------------
# # Custom CSS for a polished, premium look
# # ---------------------------------------------------------------------------
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

#     /* ---------- global ---------- */
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#     }
#     .stApp {
#         background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
#     }

#     /* ---------- header bar ---------- */
#     header[data-testid="stHeader"] {
#         background: rgba(15, 12, 41, 0.85);
#         backdrop-filter: blur(12px);
#     }

#     /* ---------- sidebar ---------- */
#     section[data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
#         border-right: 1px solid rgba(255,255,255,0.06);
#     }
#     section[data-testid="stSidebar"] .stMarkdown h1,
#     section[data-testid="stSidebar"] .stMarkdown h2,
#     section[data-testid="stSidebar"] .stMarkdown h3 {
#         color: #e0e0e0;
#     }

#     /* ---------- glass-card ---------- */
#     .glass-card {
#         background: rgba(255, 255, 255, 0.04);
#         border: 1px solid rgba(255, 255, 255, 0.08);
#         border-radius: 16px;
#         padding: 28px 32px;
#         margin-bottom: 24px;
#         backdrop-filter: blur(16px);
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
#     }

#     /* ---------- metric cards ---------- */
#     .metric-row {
#         display: flex;
#         gap: 20px;
#         flex-wrap: wrap;
#         margin-bottom: 20px;
#     }
#     .metric-card {
#         flex: 1;
#         min-width: 200px;
#         background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
#         border: 1px solid rgba(139, 92, 246, 0.25);
#         border-radius: 14px;
#         padding: 22px 26px;
#         text-align: center;
#         transition: transform 0.2s ease, box-shadow 0.2s ease;
#     }
#     .metric-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 12px 28px rgba(139, 92, 246, 0.20);
#     }
#     .metric-card .label {
#         font-size: 0.82rem;
#         color: #a0a0b8;
#         text-transform: uppercase;
#         letter-spacing: 1.5px;
#         margin-bottom: 6px;
#         font-weight: 600;
#     }
#     .metric-card .value {
#         font-size: 2rem;
#         font-weight: 800;
#         background: linear-gradient(90deg, #818cf8, #c084fc);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#     }

#     /* ---------- feature bar chart ---------- */
#     .feature-bar {
#         display: flex;
#         align-items: center;
#         margin-bottom: 10px;
#         gap: 12px;
#     }
#     .feature-bar .fname {
#         width: 200px;
#         text-align: right;
#         font-size: 0.88rem;
#         color: #c8c8dd;
#         font-weight: 500;
#         white-space: nowrap;
#         overflow: hidden;
#         text-overflow: ellipsis;
#     }
#     .feature-bar .bar-bg {
#         flex: 1;
#         height: 26px;
#         background: rgba(255,255,255,0.06);
#         border-radius: 13px;
#         overflow: hidden;
#         position: relative;
#     }
#     .feature-bar .bar-fill {
#         height: 100%;
#         border-radius: 13px;
#         background: linear-gradient(90deg, #6366f1, #a855f7);
#         transition: width 0.6s ease;
#     }
#     .feature-bar .bar-val {
#         width: 90px;
#         text-align: left;
#         font-size: 0.85rem;
#         color: #a0a0b8;
#         font-weight: 600;
#     }

#     /* ---------- title ---------- */
#     .hero-title {
#         font-size: 2.6rem;
#         font-weight: 800;
#         background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 4px;
#         letter-spacing: -0.5px;
#     }
#     .hero-sub {
#         font-size: 1.05rem;
#         color: #94a3b8;
#         margin-bottom: 28px;
#         line-height: 1.6;
#     }

#     /* ---------- misc ---------- */
#     h2, h3 {
#         color: #e2e8f0 !important;
#     }
#     .stDataFrame {
#         border-radius: 12px;
#         overflow: hidden;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # ---------------------------------------------------------------------------
# # Helper: load artefacts
# # ---------------------------------------------------------------------------
# @st.cache_resource(show_spinner="Loading models …")
# def load_artefacts():
#     """Load the persisted models and feature list from disk."""
#     soh_model = joblib.load(SOH_MODEL_PATH)
#     rul_model = joblib.load(RUL_MODEL_PATH)
#     features = joblib.load(FEATURE_LIST_PATH)
#     return soh_model, rul_model, features


# # ---------------------------------------------------------------------------
# # Helper: compute SHAP values and produce a summary bar-chart (top-5)
# # ---------------------------------------------------------------------------
# def compute_shap(model, X: pd.DataFrame, features: list):
#     """Return mean-abs SHAP importance Series and a matplotlib figure."""
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X[features])

#     mean_abs_shap = np.abs(shap_values).mean(axis=0)
#     importance = pd.Series(mean_abs_shap, index=features).sort_values(ascending=False)

#     return importance, shap_values


# def render_top5_bars(importance: pd.Series):
#     """Render custom HTML bar chart for the top-5 features."""
#     top5 = importance.head(5)
#     max_val = top5.iloc[0] if top5.iloc[0] > 0 else 1.0

#     html_parts = []
#     for feat, val in top5.items():
#         pct = val / max_val * 100
#         html_parts.append(
#             f"""
#             <div class="feature-bar">
#                 <div class="fname" title="{feat}">{feat}</div>
#                 <div class="bar-bg">
#                     <div class="bar-fill" style="width:{pct:.1f}%"></div>
#                 </div>
#                 <div class="bar-val">{val:.4f}</div>
#             </div>
#             """
#         )
#     st.markdown("".join(html_parts), unsafe_allow_html=True)


# def render_shap_summary(model, X, features, model_name="Model"):
#     """Render SHAP beeswarm / summary plot as a matplotlib figure."""
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X[features])

#     fig, ax = plt.subplots(figsize=(10, 6))
#     fig.patch.set_facecolor("#1a1a2e")
#     ax.set_facecolor("#1a1a2e")

#     shap.summary_plot(shap_values, X[features], show=False, plot_size=None)

#     # Style the current axes produced by shap
#     for a in fig.get_axes():
#         a.set_facecolor("#1a1a2e")
#         a.tick_params(colors="#c8c8dd", labelsize=9)
#         a.xaxis.label.set_color("#c8c8dd")
#         a.yaxis.label.set_color("#c8c8dd")
#         a.title.set_color("#e2e8f0")
#         for spine in a.spines.values():
#             spine.set_color("#334155")

#     fig.tight_layout()
#     return fig


# # ---------------------------------------------------------------------------
# # Validate uploaded data
# # ---------------------------------------------------------------------------
# def validate_data(df: pd.DataFrame, features: list):
#     """Check that all required feature columns exist; return missing list."""
#     missing = [f for f in features if f not in df.columns]
#     return missing


# # ---------------------------------------------------------------------------
# # MAIN UI
# # ---------------------------------------------------------------------------
# def main():
#     # -- load models ---------------------------------------------------------
#     try:
#         soh_model, rul_model, features = load_artefacts()
#     except FileNotFoundError as exc:
#         st.error(
#             f"**Model files not found.**  \n"
#             f"Please run the *Model_training_Battery_Claude* notebook first to "
#             f"generate the `.pkl` files under `models/`.  \n\n`{exc}`"
#         )
#         return

#     # -- sidebar -------------------------------------------------------------
#     with st.sidebar:
#         st.markdown("## 🔋 Battery Health Predictor")
#         st.markdown(
#             "Upload a CSV dataset with **battery cycle features** to evaluate "
#             "model performance and see which features matter most."
#         )
#         st.markdown("---")

#         st.markdown("### 📂 Upload Dataset")
#         uploaded_file = st.file_uploader(
#             "Choose a CSV file",
#             type=["csv"],
#             help="The file must contain the feature columns used during training.",
#         )

#         st.markdown("---")
#         st.markdown("### ⚙️ Prediction Target")
#         target_choice = st.radio(
#             "Select the target variable:",
#             options=["SOH (State of Health)", "RUL (Remaining Useful Life)"],
#             index=0,
#         )

#         st.markdown("---")
#         st.markdown(
#             "<div style='color:#64748b; font-size:0.78rem; line-height:1.5'>"
#             "Models: GradientBoostingRegressor<br>"
#             "Explainability: SHAP TreeExplainer<br>"
#             "Built with Streamlit"
#             "</div>",
#             unsafe_allow_html=True,
#         )

#     # -- hero ----------------------------------------------------------------
#     st.markdown('<div class="hero-title">Battery Health Prediction Dashboard</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="hero-sub">'
#         "Upload a new battery dataset to evaluate the pre-trained Gradient Boosting model. "
#         "The dashboard computes <strong>R² score</strong>, <strong>MSE</strong>, <strong>MAE</strong> "
#         "and highlights the <strong>top-5 most important features</strong> according to SHAP."
#         "</div>",
#         unsafe_allow_html=True,
#     )

#     # -- resolve dataset -----------------------------------------------------
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.success(f"✅  Dataset loaded — **{len(df)} rows × {len(df.columns)} columns**")
#         except Exception as exc:
#             st.error(f"Failed to read the uploaded CSV: {exc}")
#             return
#     else:
#         # Fall back to the built-in sample dataset
#         if os.path.exists(SAMPLE_DATA_PATH):
#             df = pd.read_csv(SAMPLE_DATA_PATH)
#             st.info(
#                 "ℹ️  No file uploaded — using the built-in sample dataset "
#                 f"(`final_data.csv`, {len(df)} rows)."
#             )
#         else:
#             st.warning("Please upload a dataset to proceed.")
#             return

#     # -- validate features ---------------------------------------------------
#     missing = validate_data(df, features)
#     if missing:
#         st.error(
#             f"**Missing feature columns** in your dataset:  \n"
#             f"`{'`, `'.join(missing)}`  \n\n"
#             f"Expected columns: `{'`, `'.join(features)}`"
#         )
#         return

#     # -- resolve model & target column ---------------------------------------
#     if "SOH" in target_choice:
#         model = soh_model
#         target_col = "SOH"
#         model_name = "SOH Model"
#     else:
#         model = rul_model
#         target_col = "RUL"
#         model_name = "RUL Model"

#     has_target = target_col in df.columns and df[target_col].notna().any()

#     # -- data preview --------------------------------------------------------
#     with st.expander("📋 Data Preview", expanded=False):
#         st.dataframe(df.head(20), use_container_width=True)

#     # -- predictions ---------------------------------------------------------
#     X = df[features]
#     predictions = model.predict(X)
#     df["Predicted_" + target_col] = predictions

#     # -- metrics (only if ground-truth exists) --------------------------------
#     st.markdown("---")

#     if has_target:
#         y_true = df[target_col].dropna()
#         y_pred = pd.Series(predictions, index=df.index).loc[y_true.index]

#         r2 = r2_score(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)

#         st.markdown("### 📊 Model Performance")
#         st.markdown(
#             f"""
#             <div class="metric-row">
#                 <div class="metric-card">
#                     <div class="label">R² Score</div>
#                     <div class="value">{r2:.4f}</div>
#                 </div>
#                 <div class="metric-card">
#                     <div class="label">MSE</div>
#                     <div class="value">{mse:.4f}</div>
#                 </div>
#                 <div class="metric-card">
#                     <div class="label">MAE</div>
#                     <div class="value">{mae:.4f}</div>
#                 </div>
#                 <div class="metric-card">
#                     <div class="label">Samples</div>
#                     <div class="value">{len(y_true)}</div>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         # -- Actual vs Predicted scatter --------------------------------------
#         st.markdown("### 🔍 Actual vs Predicted")
#         col_chart, col_table = st.columns([2, 1])

#         with col_chart:
#             fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
#             fig_scatter.patch.set_facecolor("#1a1a2e")
#             ax_scatter.set_facecolor("#1a1a2e")

#             ax_scatter.scatter(
#                 y_true, y_pred,
#                 alpha=0.5, s=18, color="#818cf8", edgecolors="#4f46e5", linewidths=0.3
#             )

#             lims = [
#                 min(y_true.min(), y_pred.min()) - 1,
#                 max(y_true.max(), y_pred.max()) + 1,
#             ]
#             ax_scatter.plot(lims, lims, "--", color="#f472b6", linewidth=1, alpha=0.7)
#             ax_scatter.set_xlim(lims)
#             ax_scatter.set_ylim(lims)
#             ax_scatter.set_xlabel(f"Actual {target_col}", color="#c8c8dd", fontsize=11)
#             ax_scatter.set_ylabel(f"Predicted {target_col}", color="#c8c8dd", fontsize=11)
#             ax_scatter.set_title(f"Actual vs Predicted — {model_name}", color="#e2e8f0", fontsize=13)
#             ax_scatter.tick_params(colors="#c8c8dd")
#             for spine in ax_scatter.spines.values():
#                 spine.set_color("#334155")
#             fig_scatter.tight_layout()
#             st.pyplot(fig_scatter)

#         with col_table:
#             st.markdown("#### Prediction Sample")
#             sample_df = df[[target_col, "Predicted_" + target_col]].head(15)
#             st.dataframe(sample_df, use_container_width=True)
#     else:
#         st.info(
#             f"ℹ️  The uploaded dataset does not contain a `{target_col}` column, "
#             f"so R² / MSE / MAE cannot be computed.  \n"
#             f"The predictions are still available below."
#         )

#         st.markdown("### 📈 Predictions")
#         st.dataframe(
#             df[features + ["Predicted_" + target_col]].head(20),
#             use_container_width=True,
#         )

#     # -- SHAP ----------------------------------------------------------------
#     st.markdown("---")
#     st.markdown("### 🧠 Top-5 Important Features (SHAP)")
#     st.markdown(
#         '<div class="glass-card">',
#         unsafe_allow_html=True,
#     )

#     with st.spinner("Computing SHAP values …"):
#         importance, shap_values = compute_shap(model, X, features)

#     render_top5_bars(importance)

#     st.markdown("</div>", unsafe_allow_html=True)

#     # -- Full SHAP summary plot ----------------------------------------------
#     with st.expander("📊 Full SHAP Summary Plot", expanded=False):
#         fig_shap = render_shap_summary(model, X, features, model_name)
#         st.pyplot(fig_shap)

#     # -- download predictions ------------------------------------------------
#     st.markdown("---")
#     st.markdown("### 💾 Download Predictions")
#     csv_out = df.to_csv(index=False)
#     st.download_button(
#         label="⬇️  Download CSV with predictions",
#         data=csv_out,
#         file_name=f"predictions_{target_col.lower()}.csv",
#         mime="text/csv",
#     )


# if __name__ == "__main__":
#     main()

# """
# Battery Health Prediction — Streamlit Interface
# =================================================
# Loads the pre-trained GradientBoostingRegressor models (SOH & RUL) that were
# persisted by the notebook pipeline, lets a user upload a new CSV dataset with
# the expected feature columns, then displays:
#   • R² / MSE / MAE for the uploaded data (if the target column is present)
#   • Actual vs Predicted plot
#   • Top-5 most important features according to SHAP (TreeExplainer)
#   • Full SHAP summary plot
#   • Downloadable CSV with predictions

# Expected project layout (next to this file):
#     app.py
#     models/soh_model.pkl
#     models/rul_model.pkl
#     models/feature_list.pkl
#     src/final_data.csv        (optional – used as sample data if nothing is uploaded)
# """

# import os

# import joblib
# import matplotlib

# matplotlib.use("Agg")  # non-interactive backend, required for Streamlit

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import shap
# import streamlit as st
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # ---------------------------------------------------------------------------
# # Paths – resolve relative to *this* file so it works regardless of cwd
# # ---------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# SRC_DIR = os.path.join(BASE_DIR, "src")

# SOH_MODEL_PATH = os.path.join(MODELS_DIR, "soh_model.pkl")
# RUL_MODEL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")
# FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.pkl")
# SAMPLE_DATA_PATH = os.path.join(SRC_DIR, "final_data.csv")

# MAX_SHAP_ROWS = 2000  # cap rows used for SHAP so large uploads stay fast

# # ---------------------------------------------------------------------------
# # Page config
# # ---------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Battery Health Predictor",
#     page_icon="🔋",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------------------------------------------------------------------------
# # Custom CSS
# # ---------------------------------------------------------------------------
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

#     html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#     .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }

#     header[data-testid="stHeader"] {
#         background: rgba(15, 12, 41, 0.85);
#         backdrop-filter: blur(12px);
#     }

#     section[data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
#         border-right: 1px solid rgba(255,255,255,0.06);
#     }
#     section[data-testid="stSidebar"] .stMarkdown h1,
#     section[data-testid="stSidebar"] .stMarkdown h2,
#     section[data-testid="stSidebar"] .stMarkdown h3 { color: #e0e0e0; }

#     .glass-card {
#         background: rgba(255, 255, 255, 0.04);
#         border: 1px solid rgba(255, 255, 255, 0.08);
#         border-radius: 16px;
#         padding: 28px 32px;
#         margin-bottom: 24px;
#         backdrop-filter: blur(16px);
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
#     }

#     .metric-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
#     .metric-card {
#         flex: 1; min-width: 200px;
#         background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
#         border: 1px solid rgba(139, 92, 246, 0.25);
#         border-radius: 14px;
#         padding: 22px 26px;
#         text-align: center;
#         transition: transform 0.2s ease, box-shadow 0.2s ease;
#     }
#     .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(139, 92, 246, 0.20); }
#     .metric-card .label {
#         font-size: 0.82rem; color: #a0a0b8; text-transform: uppercase;
#         letter-spacing: 1.5px; margin-bottom: 6px; font-weight: 600;
#     }
#     .metric-card .value {
#         font-size: 2rem; font-weight: 800;
#         background: linear-gradient(90deg, #818cf8, #c084fc);
#         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#     }

#     .feature-bar { display: flex; align-items: center; margin-bottom: 10px; gap: 12px; }
#     .feature-bar .fname {
#         width: 200px; text-align: right; font-size: 0.88rem; color: #c8c8dd;
#         font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
#     }
#     .feature-bar .bar-bg {
#         flex: 1; height: 26px; background: rgba(255,255,255,0.06);
#         border-radius: 13px; overflow: hidden; position: relative;
#     }
#     .feature-bar .bar-fill {
#         height: 100%; border-radius: 13px;
#         background: linear-gradient(90deg, #6366f1, #a855f7);
#         transition: width 0.6s ease;
#     }
#     .feature-bar .bar-val { width: 90px; text-align: left; font-size: 0.85rem; color: #a0a0b8; font-weight: 600; }

#     .hero-title {
#         font-size: 2.6rem; font-weight: 800;
#         background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
#         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#         margin-bottom: 4px; letter-spacing: -0.5px;
#     }
#     .hero-sub { font-size: 1.05rem; color: #94a3b8; margin-bottom: 28px; line-height: 1.6; }

#     h2, h3 { color: #e2e8f0 !important; }
#     .stDataFrame { border-radius: 12px; overflow: hidden; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# def show_df(data: pd.DataFrame):
#     """Render a dataframe full-width (works across Streamlit versions)."""
#     try:
#         st.dataframe(data, width="stretch")
#     except TypeError:
#         st.dataframe(data, use_container_width=True)


# @st.cache_resource(show_spinner="Loading models …")
# def load_artefacts():
#     """Load the persisted models and feature list from disk."""
#     soh_model = joblib.load(SOH_MODEL_PATH)
#     rul_model = joblib.load(RUL_MODEL_PATH)
#     features = list(joblib.load(FEATURE_LIST_PATH))
#     return soh_model, rul_model, features


# @st.cache_data(show_spinner=False)
# def compute_shap(_model, X: pd.DataFrame, model_key: str):
#     """
#     Return (mean-abs SHAP importance Series, shap_values array, X used).
#     `_model` is excluded from Streamlit's hashing; `model_key` identifies it.
#     """
#     if len(X) > MAX_SHAP_ROWS:
#         X = X.sample(MAX_SHAP_ROWS, random_state=42)

#     explainer = shap.TreeExplainer(_model)
#     shap_values = explainer.shap_values(X)

#     importance = (
#         pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
#         .sort_values(ascending=False)
#     )
#     return importance, shap_values, X


# def render_top5_card(importance: pd.Series):
#     """Render the top-5 features as a single HTML 'glass card' with bars."""
#     top5 = importance.head(5)
#     max_val = top5.iloc[0] if top5.iloc[0] > 0 else 1.0

#     bars = ""
#     for feat, val in top5.items():
#         pct = val / max_val * 100
#         bars += (
#             '<div class="feature-bar">'
#             f'<div class="fname" title="{feat}">{feat}</div>'
#             f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
#             f'<div class="bar-val">{val:.4f}</div>'
#             "</div>"
#         )
#     st.markdown(f'<div class="glass-card">{bars}</div>', unsafe_allow_html=True)


# def render_shap_summary(shap_values, X_used: pd.DataFrame):
#     """Build the SHAP beeswarm summary plot, themed for the dark UI."""
#     plt.close("all")
#     height = max(4.5, 0.45 * X_used.shape[1] + 2)
#     shap.summary_plot(shap_values, X_used, show=False, plot_size=(10, height))
#     fig = plt.gcf()

#     fig.patch.set_facecolor("#1a1a2e")
#     for ax in fig.get_axes():
#         ax.set_facecolor("#1a1a2e")
#         for spine in ax.spines.values():
#             spine.set_color("#334155")
#     for txt in fig.findobj(matplotlib.text.Text):
#         txt.set_color("#c8c8dd")
#     for ax in fig.get_axes():
#         ax.tick_params(colors="#c8c8dd", labelsize=9)

#     fig.tight_layout()
#     return fig


# def render_metric_cards(r2, mse, mae, n):
#     cards = [("R² Score", f"{r2:.4f}"), ("MSE", f"{mse:.4f}"), ("MAE", f"{mae:.4f}"), ("Samples", f"{n}")]
#     inner = "".join(
#         f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>'
#         for lbl, val in cards
#     )
#     st.markdown(f'<div class="metric-row">{inner}</div>', unsafe_allow_html=True)


# def validate_data(df: pd.DataFrame, features: list):
#     """Return the list of required feature columns missing from df."""
#     return [f for f in features if f not in df.columns]


# # ---------------------------------------------------------------------------
# # MAIN UI
# # ---------------------------------------------------------------------------
# def main():
#     # -- load models ---------------------------------------------------------
#     try:
#         soh_model, rul_model, features = load_artefacts()
#     except FileNotFoundError as exc:
#         st.error(
#             "**Model files not found.**  \n"
#             "Please run the *Model_training_Battery_Claude* notebook first to "
#             "generate the `.pkl` files under `models/`.  \n\n"
#             f"`{exc}`"
#         )
#         return

#     # -- sidebar -------------------------------------------------------------
#     with st.sidebar:
#         st.markdown("## 🔋 Battery Health Predictor")
#         st.markdown(
#             "Upload a CSV dataset with **battery cycle features** to evaluate "
#             "model performance and see which features matter most."
#         )
#         st.markdown("---")

#         st.markdown("### 📂 Upload Dataset")
#         uploaded_file = st.file_uploader(
#             "Choose a CSV file",
#             type=["csv"],
#             help="The file must contain the feature columns used during training.",
#         )

#         st.markdown("---")
#         st.markdown("### ⚙️ Prediction Target")
#         target_choice = st.radio(
#             "Select the target variable:",
#             options=["SOH (State of Health)", "RUL (Remaining Useful Life)"],
#             index=0,
#         )

#         st.markdown("---")
#         st.markdown(
#             "<div style='color:#64748b; font-size:0.78rem; line-height:1.5'>"
#             "Models: GradientBoostingRegressor<br>"
#             "Explainability: SHAP TreeExplainer<br>"
#             "Built with Streamlit"
#             "</div>",
#             unsafe_allow_html=True,
#         )

#     # -- hero ----------------------------------------------------------------
#     st.markdown('<div class="hero-title">Battery Health Prediction Dashboard</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="hero-sub">'
#         "Upload a new battery dataset to evaluate the pre-trained Gradient Boosting model. "
#         "The dashboard computes <strong>R² score</strong>, <strong>MSE</strong>, <strong>MAE</strong> "
#         "and highlights the <strong>top-5 most important features</strong> according to SHAP."
#         "</div>",
#         unsafe_allow_html=True,
#     )

#     # -- resolve dataset -----------------------------------------------------
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.success(f"✅  Dataset loaded — **{len(df)} rows × {len(df.columns)} columns**")
#         except Exception as exc:
#             st.error(f"Failed to read the uploaded CSV: {exc}")
#             return
#     else:
#         if os.path.exists(SAMPLE_DATA_PATH):
#             df = pd.read_csv(SAMPLE_DATA_PATH)
#             st.info(
#                 "ℹ️  No file uploaded — using the built-in sample dataset "
#                 f"(`final_data.csv`, {len(df)} rows)."
#             )
#         else:
#             st.warning("Please upload a dataset to proceed.")
#             return

#     # -- validate features ---------------------------------------------------
#     missing = validate_data(df, features)
#     if missing:
#         st.error(
#             "**Missing feature columns** in your dataset:  \n"
#             f"`{'`, `'.join(missing)}`  \n\n"
#             f"Expected columns: `{'`, `'.join(features)}`"
#         )
#         return

#     # -- resolve model & target column --------------------------------------
#     if "SOH" in target_choice:
#         model, target_col, model_name = soh_model, "SOH", "SOH Model"
#     else:
#         model, target_col, model_name = rul_model, "RUL", "RUL Model"
#     pred_col = "Predicted_" + target_col

#     # -- clean features (numeric only, no NaNs) -------------------------------
#     X_all = df[features].apply(pd.to_numeric, errors="coerce")
#     X_all = X_all.replace([np.inf, -np.inf], np.nan)
#     valid_mask = X_all.notna().all(axis=1)
#     n_dropped = int((~valid_mask).sum())

#     if n_dropped:
#         st.warning(
#             f"⚠️  {n_dropped} row(s) contain missing or non-numeric feature values "
#             "and were skipped for prediction."
#         )
#     if valid_mask.sum() == 0:
#         st.error("No valid rows left after cleaning the feature columns.")
#         return

#     X = X_all.loc[valid_mask]

#     # -- data preview ---------------------------------------------------------
#     with st.expander("📋 Data Preview", expanded=False):
#         show_df(df.head(20))

#     # -- predictions ----------------------------------------------------------
#     df[pred_col] = np.nan
#     df.loc[valid_mask, pred_col] = model.predict(X)

#     # -- metrics (only if ground truth exists) -------------------------------
#     st.markdown("---")

#     y_all = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else None
#     eval_mask = valid_mask & y_all.notna() if y_all is not None else None
#     has_target = eval_mask is not None and int(eval_mask.sum()) >= 2

#     if has_target:
#         y_true = y_all.loc[eval_mask]
#         y_pred = df.loc[eval_mask, pred_col]

#         r2 = r2_score(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)

#         st.markdown("### 📊 Model Performance")
#         render_metric_cards(r2, mse, mae, len(y_true))

#         # -- Actual vs Predicted ----------------------------------------------
#         st.markdown("### 🔍 Actual vs Predicted")
#         col_chart, col_table = st.columns([2, 1])

#         with col_chart:
#             fig_sc, ax_sc = plt.subplots(figsize=(7, 5))
#             fig_sc.patch.set_facecolor("#1a1a2e")
#             ax_sc.set_facecolor("#1a1a2e")

#             ax_sc.scatter(
#                 y_true, y_pred, alpha=0.5, s=18,
#                 color="#818cf8", edgecolors="#4f46e5", linewidths=0.3,
#             )

#             lo = min(y_true.min(), y_pred.min())
#             hi = max(y_true.max(), y_pred.max())
#             pad = (hi - lo) * 0.03 if hi > lo else 1.0
#             lims = [lo - pad, hi + pad]
#             ax_sc.plot(lims, lims, "--", color="#f472b6", linewidth=1, alpha=0.7)
#             ax_sc.set_xlim(lims)
#             ax_sc.set_ylim(lims)
#             ax_sc.set_xlabel(f"Actual {target_col}", color="#c8c8dd", fontsize=11)
#             ax_sc.set_ylabel(f"Predicted {target_col}", color="#c8c8dd", fontsize=11)
#             ax_sc.set_title(f"Actual vs Predicted — {model_name}", color="#e2e8f0", fontsize=13)
#             ax_sc.tick_params(colors="#c8c8dd")
#             for spine in ax_sc.spines.values():
#                 spine.set_color("#334155")
#             fig_sc.tight_layout()
#             st.pyplot(fig_sc)
#             plt.close(fig_sc)

#         with col_table:
#             st.markdown("#### Prediction Sample")
#             show_df(df.loc[eval_mask, [target_col, pred_col]].head(15))
#     else:
#         st.info(
#             f"ℹ️  The dataset has no usable `{target_col}` values, so R² / MSE / MAE "
#             "cannot be computed.  \nThe predictions are still available below."
#         )
#         st.markdown("### 📈 Predictions")
#         show_df(df.loc[valid_mask, features + [pred_col]].head(20))

#     # -- SHAP -----------------------------------------------------------------
#     st.markdown("---")
#     st.markdown("### 🧠 Top-5 Important Features (SHAP)")

#     with st.spinner("Computing SHAP values …"):
#         importance, shap_values, X_shap = compute_shap(model, X, target_col)

#     if len(X) > MAX_SHAP_ROWS:
#         st.caption(f"SHAP computed on a random sample of {MAX_SHAP_ROWS} rows for speed.")

#     render_top5_card(importance)

#     with st.expander("📊 Full SHAP Summary Plot", expanded=False):
#         fig_shap = render_shap_summary(shap_values, X_shap)
#         st.pyplot(fig_shap)
#         plt.close(fig_shap)

#     # -- download predictions -------------------------------------------------
#     st.markdown("---")
#     st.markdown("### 💾 Download Predictions")
#     st.download_button(
#         label="⬇️  Download CSV with predictions",
#         data=df.to_csv(index=False).encode("utf-8"),
#         file_name=f"predictions_{target_col.lower()}.csv",
#         mime="text/csv",
#     )


# if __name__ == "__main__":
#     main()

"""
Battery Health Prediction — Streamlit Interface
=================================================
Loads the pre-trained GradientBoostingRegressor models (SOH & RUL) that were
persisted by the notebook pipeline, lets a user upload a new CSV dataset with
the expected feature columns, then displays:
  • R² / MSE / MAE for the uploaded data (if the target column is present)
  • Actual vs Predicted plot
  • Top-5 most important features according to SHAP (TreeExplainer)
  • Full SHAP summary plot
  • Downloadable CSV with predictions

Expected project layout (next to this file):
    app.py
    models/soh_model.pkl
    models/rul_model.pkl
    models/feature_list.pkl
    src/final_data.csv        (optional – used as sample data if nothing is uploaded)
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend, required for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths – resolve relative to *this* file so it works regardless of cwd
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

SOH_MODEL_PATH = os.path.join(MODELS_DIR, "soh_model.pkl")
RUL_MODEL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.pkl")
SAMPLE_DATA_PATH = os.path.join(SRC_DIR, "final_data.csv")

MAX_SHAP_ROWS = 2000  # cap rows used for SHAP so large uploads stay fast

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery Health Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }

    header[data-testid="stHeader"] {
        background: rgba(15, 12, 41, 0.85);
        backdrop-filter: blur(12px);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e0e0e0; }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .metric-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
    .metric-card {
        flex: 1; min-width: 200px;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 14px;
        padding: 22px 26px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(139, 92, 246, 0.20); }
    .metric-card .label {
        font-size: 0.82rem; color: #a0a0b8; text-transform: uppercase;
        letter-spacing: 1.5px; margin-bottom: 6px; font-weight: 600;
    }
    .metric-card .value {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .feature-bar { display: flex; align-items: center; margin-bottom: 10px; gap: 12px; }
    .feature-bar .fname {
        width: 200px; text-align: right; font-size: 0.88rem; color: #c8c8dd;
        font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .feature-bar .bar-bg {
        flex: 1; height: 26px; background: rgba(255,255,255,0.06);
        border-radius: 13px; overflow: hidden; position: relative;
    }
    .feature-bar .bar-fill {
        height: 100%; border-radius: 13px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        transition: width 0.6s ease;
    }
    .feature-bar .bar-val { width: 90px; text-align: left; font-size: 0.85rem; color: #a0a0b8; font-weight: 600; }

    .hero-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px; letter-spacing: -0.5px;
    }
    .hero-sub { font-size: 1.05rem; color: #94a3b8; margin-bottom: 28px; line-height: 1.6; }

    h2, h3 { color: #e2e8f0 !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_df(data: pd.DataFrame):
    """Render a dataframe full-width (works across Streamlit versions)."""
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


@st.cache_resource(show_spinner="Loading models …")
def load_artefacts():
    """Load the persisted models and feature list from disk."""
    soh_model = joblib.load(SOH_MODEL_PATH)
    rul_model = joblib.load(RUL_MODEL_PATH)
    features = list(joblib.load(FEATURE_LIST_PATH))
    return soh_model, rul_model, features


@st.cache_data(show_spinner=False)
def compute_shap(_model, X: pd.DataFrame, model_key: str):
    """
    Return (mean-abs SHAP importance Series, shap_values array, X used).
    `_model` is excluded from Streamlit's hashing; `model_key` identifies it.
    """
    if len(X) > MAX_SHAP_ROWS:
        X = X.sample(MAX_SHAP_ROWS, random_state=42)

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)

    importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
        .sort_values(ascending=False)
    )
    return importance, shap_values, X


def render_top5_card(importance: pd.Series):
    """Render the top-5 features as a single HTML 'glass card' with bars."""
    top5 = importance.head(5)
    max_val = top5.iloc[0] if top5.iloc[0] > 0 else 1.0

    bars = ""
    for feat, val in top5.items():
        pct = val / max_val * 100
        bars += (
            '<div class="feature-bar">'
            f'<div class="fname" title="{feat}">{feat}</div>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="bar-val">{val:.4f}</div>'
            "</div>"
        )
    st.markdown(f'<div class="glass-card">{bars}</div>', unsafe_allow_html=True)


def render_shap_summary(shap_values, X_used: pd.DataFrame):
    """Build the SHAP beeswarm summary plot, themed for the dark UI."""
    plt.close("all")
    height = max(4.5, 0.45 * X_used.shape[1] + 2)
    shap.summary_plot(shap_values, X_used, show=False, plot_size=(10, height))
    fig = plt.gcf()

    fig.patch.set_facecolor("#1a1a2e")
    for ax in fig.get_axes():
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_color("#334155")
    for txt in fig.findobj(matplotlib.text.Text):
        txt.set_color("#c8c8dd")
    for ax in fig.get_axes():
        ax.tick_params(colors="#c8c8dd", labelsize=9)

    fig.tight_layout()
    return fig


def render_metric_cards(r2, mse, mae, n):
    cards = [("R² Score", f"{r2:.4f}"), ("MSE", f"{mse:.4f}"), ("MAE", f"{mae:.4f}"), ("Samples", f"{n}")]
    inner = "".join(
        f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>'
        for lbl, val in cards
    )
    st.markdown(f'<div class="metric-row">{inner}</div>', unsafe_allow_html=True)


TRAINING_BATTERIES = {"B0005", "B0006", "B0007", "B0018"}
EOL_SOH = 70.0  # end-of-life threshold used in feature_engg notebook


def add_engineered_features(df: pd.DataFrame):
    """
    Recreate the columns produced by feature_engg_Battery_Claude.ipynb when the
    uploaded file is a RAW cycle file (like battery_cycle_data.csv).
    Existing columns are never overwritten. Returns (df, list_of_added_columns).
    """
    df = df.copy()
    added = []

    key = df["battery_id"] if "battery_id" in df.columns else pd.Series("ALL", index=df.index)
    cycle = pd.to_numeric(df["cycle"], errors="coerce") if "cycle" in df.columns else None

    if "cycle_normalized" not in df.columns and cycle is not None:
        df["cycle_normalized"] = cycle / cycle.groupby(key).transform("max")
        added.append("cycle_normalized")

    if "voltage_range" not in df.columns and {"voltage_max", "voltage_min"} <= set(df.columns):
        df["voltage_range"] = (
            pd.to_numeric(df["voltage_max"], errors="coerce")
            - pd.to_numeric(df["voltage_min"], errors="coerce")
        )
        added.append("voltage_range")

    if "temperature_range" not in df.columns and {"temperature_max", "temperature_min"} <= set(df.columns):
        df["temperature_range"] = (
            pd.to_numeric(df["temperature_max"], errors="coerce")
            - pd.to_numeric(df["temperature_min"], errors="coerce")
        )
        added.append("temperature_range")

    # Ground-truth targets, derived from capacity exactly as in the notebook
    if cycle is not None and "capacity" in df.columns:
        cap = pd.to_numeric(df["capacity"], errors="coerce")

        if "SOH" not in df.columns:
            first_cap = cap.loc[cycle.sort_values().index].groupby(key).transform("first")
            df["SOH"] = 100 * cap / first_cap
            added.append("SOH")

        if "RUL" not in df.columns:
            soh = pd.to_numeric(df["SOH"], errors="coerce")
            eol_cycle = cycle[soh <= EOL_SOH].groupby(key[soh <= EOL_SOH]).min()
            df["RUL"] = key.map(eol_cycle) - cycle  # NaN if battery never reaches 70% SOH
            added.append("RUL")

    return df, added


def validate_data(df: pd.DataFrame, features: list):
    """Return the list of required feature columns missing from df."""
    return [f for f in features if f not in df.columns]


# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------
def main():
    # -- load models ---------------------------------------------------------
    try:
        soh_model, rul_model, features = load_artefacts()
    except FileNotFoundError as exc:
        st.error(
            "**Model files not found.**  \n"
            "Please run the *Model_training_Battery_Claude* notebook first to "
            "generate the `.pkl` files under `models/`.  \n\n"
            f"`{exc}`"
        )
        return

    # -- sidebar -------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🔋 Battery Health Predictor")
        st.markdown(
            "Upload a CSV dataset with **battery cycle features** to evaluate "
            "model performance and see which features matter most."
        )
        st.markdown("---")

        st.markdown("### 📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="The file must contain the feature columns used during training.",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Prediction Target")
        target_choice = st.radio(
            "Select the target variable:",
            options=["SOH (State of Health)", "RUL (Remaining Useful Life)"],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            "<div style='color:#64748b; font-size:0.78rem; line-height:1.5'>"
            "Models: GradientBoostingRegressor<br>"
            "Explainability: SHAP TreeExplainer<br>"
            "Built with Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )

    # -- hero ----------------------------------------------------------------
    st.markdown('<div class="hero-title">Battery Health Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">'
        "Upload a new battery dataset to evaluate the pre-trained Gradient Boosting model. "
        "The dashboard computes <strong>R² score</strong>, <strong>MSE</strong>, <strong>MAE</strong> "
        "and highlights the <strong>top-5 most important features</strong> according to SHAP."
        "</div>",
        unsafe_allow_html=True,
    )

    # -- resolve dataset -----------------------------------------------------
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅  Dataset loaded — **{len(df)} rows × {len(df.columns)} columns**")
        except Exception as exc:
            st.error(f"Failed to read the uploaded CSV: {exc}")
            return
    else:
        if os.path.exists(SAMPLE_DATA_PATH):
            df = pd.read_csv(SAMPLE_DATA_PATH)
            st.info(
                "ℹ️  No file uploaded — using the built-in sample dataset "
                f"(`final_data.csv`, {len(df)} rows)."
            )
        else:
            st.warning("Please upload a dataset to proceed.")
            return

    # -- build engineered columns if the file is a raw cycle file --------------
    df, added_cols = add_engineered_features(df)
    if added_cols:
        derived_targets = [c for c in added_cols if c in ("SOH", "RUL")]
        engineered = [c for c in added_cols if c not in ("SOH", "RUL")]
        msg = "ℹ️  Computed automatically from your raw columns: " + ", ".join(f"`{c}`" for c in added_cols) + "."
        if derived_targets:
            msg += (
                "  \n`SOH` / `RUL` are derived from `capacity` (first cycle = 100% SOH; "
                f"RUL = cycles until SOH ≤ {EOL_SOH:.0f}%), so they are only valid if each battery's "
                "file starts at its first cycle. RUL is blank for batteries that never reach the threshold."
            )
        st.info(msg)

    # -- warn about in-sample evaluation --------------------------------------
    if "battery_id" in df.columns and set(df["battery_id"].astype(str)) <= TRAINING_BATTERIES:
        st.warning(
            "⚠️  All batteries in this data (" + ", ".join(sorted(set(df["battery_id"].astype(str)))) + ") "
            "were used to train the saved models, so these scores are **in-sample** and look better "
            "than real-world performance. For honest numbers use unseen batteries, or see the "
            "leave-one-battery-out results in the training notebook."
        )

    # -- validate features ---------------------------------------------------
    missing = validate_data(df, features)
    if missing:
        st.error(
            "**Missing feature columns** in your dataset:  \n"
            f"`{'`, `'.join(missing)}`  \n\n"
            f"Expected columns: `{'`, `'.join(features)}`"
        )
        return

    # -- resolve model & target column --------------------------------------
    if "SOH" in target_choice:
        model, target_col, model_name = soh_model, "SOH", "SOH Model"
    else:
        model, target_col, model_name = rul_model, "RUL", "RUL Model"
    pred_col = "Predicted_" + target_col

    # -- clean features (numeric only, no NaNs) -------------------------------
    X_all = df[features].apply(pd.to_numeric, errors="coerce")
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    valid_mask = X_all.notna().all(axis=1)
    n_dropped = int((~valid_mask).sum())

    if n_dropped:
        st.warning(
            f"⚠️  {n_dropped} row(s) contain missing or non-numeric feature values "
            "and were skipped for prediction."
        )
    if valid_mask.sum() == 0:
        st.error("No valid rows left after cleaning the feature columns.")
        return

    X = X_all.loc[valid_mask]

    # -- data preview ---------------------------------------------------------
    with st.expander("📋 Data Preview", expanded=False):
        show_df(df.head(20))

    # -- predictions ----------------------------------------------------------
    df[pred_col] = np.nan
    df.loc[valid_mask, pred_col] = model.predict(X)

    # -- metrics (only if ground truth exists) -------------------------------
    st.markdown("---")

    y_all = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else None
    eval_mask = valid_mask & y_all.notna() if y_all is not None else None
    has_target = eval_mask is not None and int(eval_mask.sum()) >= 2

    if has_target:
        y_true = y_all.loc[eval_mask]
        y_pred = df.loc[eval_mask, pred_col]

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        st.markdown("### 📊 Model Performance")
        render_metric_cards(r2, mse, mae, len(y_true))

        # -- Actual vs Predicted ----------------------------------------------
        st.markdown("### 🔍 Actual vs Predicted")
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            fig_sc, ax_sc = plt.subplots(figsize=(7, 5))
            fig_sc.patch.set_facecolor("#1a1a2e")
            ax_sc.set_facecolor("#1a1a2e")

            ax_sc.scatter(
                y_true, y_pred, alpha=0.5, s=18,
                color="#818cf8", edgecolors="#4f46e5", linewidths=0.3,
            )

            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            pad = (hi - lo) * 0.03 if hi > lo else 1.0
            lims = [lo - pad, hi + pad]
            ax_sc.plot(lims, lims, "--", color="#f472b6", linewidth=1, alpha=0.7)
            ax_sc.set_xlim(lims)
            ax_sc.set_ylim(lims)
            ax_sc.set_xlabel(f"Actual {target_col}", color="#c8c8dd", fontsize=11)
            ax_sc.set_ylabel(f"Predicted {target_col}", color="#c8c8dd", fontsize=11)
            ax_sc.set_title(f"Actual vs Predicted — {model_name}", color="#e2e8f0", fontsize=13)
            ax_sc.tick_params(colors="#c8c8dd")
            for spine in ax_sc.spines.values():
                spine.set_color("#334155")
            fig_sc.tight_layout()
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        with col_table:
            st.markdown("#### Prediction Sample")
            show_df(df.loc[eval_mask, [target_col, pred_col]].head(15))
    else:
        st.info(
            f"ℹ️  The dataset has no usable `{target_col}` values, so R² / MSE / MAE "
            "cannot be computed.  \nThe predictions are still available below."
        )
        st.markdown("### 📈 Predictions")
        show_df(df.loc[valid_mask, features + [pred_col]].head(20))

    # -- SHAP -----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧠 Top-5 Important Features (SHAP)")

    with st.spinner("Computing SHAP values …"):
        importance, shap_values, X_shap = compute_shap(model, X, target_col)

    if len(X) > MAX_SHAP_ROWS:
        st.caption(f"SHAP computed on a random sample of {MAX_SHAP_ROWS} rows for speed.")

    render_top5_card(importance)

    with st.expander("📊 Full SHAP Summary Plot", expanded=False):
        fig_shap = render_shap_summary(shap_values, X_shap)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    # -- download predictions -------------------------------------------------
    st.markdown("---")
    st.markdown("### 💾 Download Predictions")
    st.download_button(
        label="⬇️  Download CSV with predictions",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{target_col.lower()}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()