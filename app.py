import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="IPL Player Performance Dashboard",
    page_icon="🏏",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main { background-color: #0E1117; color: #F4EBD0; }
        h1, h2, h3, h4 { color: #FFD700 !important; }
        .stMetricValue { color: #FFD700 !important; }
        .stDataFrame, .stTable { background-color: #1A1D23 !important; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

DATA_PATH = "processed_ipl_full.csv"
MODELS_PATH = "models"

required_files = [
    "runs_linear.pkl", "runs_rf.pkl", "runs_gb.pkl", "runs_xgb.pkl",
    "wkts_linear.pkl", "wkts_rf.pkl", "wkts_gb.pkl", "wkts_xgb.pkl",
    "runs_scaler.pkl", "wkts_scaler.pkl",
    "player_role_clf.pkl", "player_role_scaler.pkl",
    "player_roles_reference.csv", "all_models_metrics.csv"
]

missing = [f for f in required_files if not os.path.exists(os.path.join(MODELS_PATH, f))]
if not os.path.exists(DATA_PATH) or missing:
    st.error("🚨 Required files missing. Please run `train_all_models.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
metrics = pd.read_csv(os.path.join(MODELS_PATH, "all_models_metrics.csv"))
roles = pd.read_csv(os.path.join(MODELS_PATH, "player_roles_reference.csv"))

models = {
    "runs": {
        "linear": joblib.load(os.path.join(MODELS_PATH, "runs_linear.pkl")),
        "rf": joblib.load(os.path.join(MODELS_PATH, "runs_rf.pkl")),
        "gb": joblib.load(os.path.join(MODELS_PATH, "runs_gb.pkl")),
        "xgb": joblib.load(os.path.join(MODELS_PATH, "runs_xgb.pkl")),
        "scaler": joblib.load(os.path.join(MODELS_PATH, "runs_scaler.pkl")),
    },
    "wkts": {
        "linear": joblib.load(os.path.join(MODELS_PATH, "wkts_linear.pkl")),
        "rf": joblib.load(os.path.join(MODELS_PATH, "wkts_rf.pkl")),
        "gb": joblib.load(os.path.join(MODELS_PATH, "wkts_gb.pkl")),
        "xgb": joblib.load(os.path.join(MODELS_PATH, "wkts_xgb.pkl")),
        "scaler": joblib.load(os.path.join(MODELS_PATH, "wkts_scaler.pkl")),
    },
    "role": {
        "clf": joblib.load(os.path.join(MODELS_PATH, "player_role_clf.pkl")),
        "scaler": joblib.load(os.path.join(MODELS_PATH, "player_role_scaler.pkl")),
    },
}

st.sidebar.title("Dashboard Controls")

task = st.sidebar.selectbox("Select Prediction Type", ["Batting (Runs)", "Bowling (Wickets)", "Player Classification"])
model_choice = st.sidebar.selectbox("Select Model", ["linear", "rf", "gb", "xgb"])
selected_player = st.sidebar.selectbox("Select Player", sorted(df["Player_Name"].unique()))

st.title("IPL Player Performance Predictor")
st.caption("Next-season Predictions • Powered by ML")

if task in ["Batting (Runs)", "Bowling (Wickets)"]:
    prefix = "runs" if "Batting" in task else "wkts"
    features = {
        "runs": ['Matches_Batted','Runs_Scored','Batting_Average','Balls_Faced','Batting_Strike_Rate',
                 'Centuries','Half_Centuries','Fours','Sixes','Form_3yr','Cons_3yr','Experience','Career_Runs'],
        "wkts": ['Matches_Bowled','Balls_Bowled','Runs_Conceded','Wickets_Taken','Wkts_Form_3yr','Wkts_Cons_3yr',
                 'Bowling_Average','Economy_Rate','Career_Wkts','Experience']
    }

    player_data = df[df["Player_Name"] == selected_player].sort_values("Year")

    latest_record = player_data[player_data["Year"] == 2023].copy()
    if latest_record.empty:
        latest_record = player_data.iloc[-1:].copy() 

    if latest_record.empty:
        st.warning("No valid data found for this player.")
        st.stop()

    model = models[prefix][model_choice]
    scaler = models[prefix]["scaler"]
    X = latest_record[features[prefix]].fillna(0)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    actual_col = "Runs_Scored" if prefix == "runs" else "Wickets_Taken"

    st.subheader(f"Predicted {task.split()[1]} for 2024 (Based on 2023 Stats)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual (2023)", int(latest_record[actual_col].values[0]))
    c2.metric("Predicted (2024)", int(pred))
    c3.metric("Model Used", model_choice.upper())

    # Trend Plot
    st.markdown("Player Performance Over Years")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=player_data["Year"], y=player_data[actual_col],
        mode="lines+markers", name=f"Actual {actual_col}",
        line=dict(color="#FFD700", width=3)
    ))
    if prefix == "runs":
        fig.add_trace(go.Scatter(
            x=player_data["Year"], y=player_data["Form_3yr"],
            mode="lines+markers", name="Form (3yr Avg)",
            line=dict(color="orange", dash="dot")
        ))
    else:
        fig.add_trace(go.Scatter(
            x=player_data["Year"], y=player_data["Wkts_Form_3yr"],
            mode="lines+markers", name="Wicket Form (3yr Avg)",
            line=dict(color="orange", dash="dot")
        ))
    fig.update_layout(template="plotly_dark", title=f"{selected_player} - Yearly Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Model Metrics Comparison
    st.markdown("Model Performance Comparison")
    perf = metrics[metrics["Task"] == ("Runs" if prefix == "runs" else "Wickets")]
    fig_perf = px.bar(perf, x="Model", y=["MAE", "RMSE", "R2"], barmode="group",
                      title="Model Metrics (Lower = Better for MAE/RMSE)", template="plotly_dark")
    st.plotly_chart(fig_perf, use_container_width=True)

    # Feature Importance
    st.markdown("Feature Importance")
    fi_path = os.path.join(MODELS_PATH, f"{prefix}_{model_choice}_feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        fi.columns = ["Feature", "Importance"]
        fig_fi = px.bar(fi, x="Feature", y="Importance", color="Importance",
                        color_continuous_scale="solar", template="plotly_dark",
                        title=f"Feature Importance — {model_choice.upper()} ({prefix.upper()})")
        fig_fi.update_traces(marker=dict(line=dict(color="#FFD700", width=1.2)))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info(f"No feature importance available for {model_choice.upper()}.")

    st.markdown(f"Top 5 Predicted {task.split()[1]} Scorers for 2024")
    latest_season = df[df["Year"] == 2023].copy()
    X_all = latest_season[features[prefix]].fillna(0)
    X_all_scaled = scaler.transform(X_all)
    latest_season["Predicted"] = model.predict(X_all_scaled)
    top5 = latest_season.sort_values("Predicted", ascending=False).head(5)
    st.table(top5[["Player_Name", actual_col, "Predicted"]])

elif task == "Player Classification":
    st.header("Player Role Classification")
    clf = models["role"]["clf"]
    scaler = models["role"]["scaler"]
    latest_data = df.groupby("Player_Name").tail(1).reset_index(drop=True)

    clf_features = [
        'Matches_Batted','Runs_Scored','Batting_Average','Balls_Faced','Batting_Strike_Rate',
        'Centuries','Fours','Sixes','Matches_Bowled','Wickets_Taken','Balls_Bowled',
        'Economy_Rate','Career_Runs','Career_Wkts'
    ]
    clf_features = [f for f in clf_features if f in latest_data.columns]

    X = latest_data[clf_features].fillna(0)
    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)
    latest_data["Predicted_Role"] = preds

    player_role = latest_data.loc[latest_data["Player_Name"] == selected_player, "Predicted_Role"].values[0]
    st.metric("Predicted Player Role", player_role)

    st.markdown("### All Players — Predicted Roles")
    st.dataframe(latest_data[["Player_Name", "Predicted_Role"]].sort_values("Predicted_Role"))

st.markdown("---")
st.markdown("<h6 style='text-align:center;color:gray;'>© 2025 IPL Performance Analyzer — Built with Streamlit, Scikit-learn & XGBoost</h6>", unsafe_allow_html=True)
