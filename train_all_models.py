import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
import time

DATA_FILE = "ipl_stats.csv"
OUT_DIR = "models"
PROCESSED_FILE = "processed_ipl_full.csv"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

def safe_to_numeric(series):
    """Remove non-numeric characters and convert to numeric safely."""
    return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

def save_metrics(df_metrics, path):
    df_metrics.to_csv(path, index=False)
    print(f"Saved metrics -> {path}")

print("⏳ Loading dataset:", DATA_FILE)
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"'{DATA_FILE}' not found. Place your IPL CSV in the project folder and name it '{DATA_FILE}'.")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

df['Year'] = pd.to_numeric(df.get('Year', pd.Series()), errors='coerce')
df['Player_Name'] = df.get('Player_Name', pd.Series()).astype(str).str.strip()
df = df.dropna(subset=['Year', 'Player_Name']).copy()
df['Year'] = df['Year'].astype(int)

df.replace(['No stats', 'No data', '-', '–', 'NA', 'N/A', '', ' '], np.nan, inplace=True)

expected_numeric = [
    'Matches_Batted','Not_Outs','Runs_Scored','Highest_Score','Batting_Average','Balls_Faced','Batting_Strike_Rate',
    'Centuries','Half_Centuries','Fours','Sixes','Catches_Taken','Stumpings',
    'Matches_Bowled','Balls_Bowled','Runs_Conceded','Wickets_Taken','Best_Bowling_Match','Bowling_Average',
    'Economy_Rate','Bowling_Strike_Rate','Four_Wicket_Hauls','Five_Wicket_Hauls'
]

for col in expected_numeric:
    if col in df.columns:
        df[col] = safe_to_numeric(df[col]).fillna(0)
    else:
        df[col] = 0

print("Basic cleaning complete. Years:", int(df['Year'].min()), "to", int(df['Year'].max()))
print("Total rows:", len(df), "unique players:", df['Player_Name'].nunique())
print("\nGenerating features (rolling 3-year form/consistency, career aggregates, experience)...")
df = df.sort_values(['Player_Name', 'Year'])
frames = []
for player, g in df.groupby('Player_Name'):
    g = g.sort_values('Year').copy()
    g['Form_3yr'] = g['Runs_Scored'].rolling(window=3, min_periods=1).mean().shift(1)
    g['Cons_3yr'] = g['Runs_Scored'].rolling(window=3, min_periods=1).std().shift(1).fillna(0)
    g['Wkts_Form_3yr'] = g['Wickets_Taken'].rolling(window=3, min_periods=1).mean().shift(1)
    g['Wkts_Cons_3yr'] = g['Wickets_Taken'].rolling(window=3, min_periods=1).std().shift(1).fillna(0)
    g['Career_Runs'] = g['Runs_Scored'].expanding(min_periods=1).sum().shift(1).fillna(0)
    g['Career_Wkts'] = g['Wickets_Taken'].expanding(min_periods=1).sum().shift(1).fillna(0)
    g['Experience'] = np.arange(1, len(g) + 1)
    frames.append(g)

df_feat = pd.concat(frames).reset_index(drop=True)
print("✅ Feature generation complete. Total rows after features:", len(df_feat))

df_feat.to_csv(PROCESSED_FILE, index=False)
print(f"📁 Saved processed dataset -> {PROCESSED_FILE}")

df_feat['Runs_next'] = df_feat.groupby('Player_Name')['Runs_Scored'].shift(-1)
df_feat['Wkts_next'] = df_feat.groupby('Player_Name')['Wickets_Taken'].shift(-1)

df_runs = df_feat.dropna(subset=['Runs_next']).copy()
df_wkts = df_feat.dropna(subset=['Wkts_next']).copy()
print(f"Rows available for runs model: {len(df_runs)}; for wickets model: {len(df_wkts)}")

runs_features = [
    'Matches_Batted','Runs_Scored','Batting_Average','Balls_Faced','Batting_Strike_Rate',
    'Centuries','Half_Centuries','Fours','Sixes','Form_3yr','Cons_3yr','Experience','Career_Runs'
]
wkts_features = [
    'Matches_Bowled','Balls_Bowled','Runs_Conceded','Wickets_Taken','Wkts_Form_3yr','Wkts_Cons_3yr',
    'Bowling_Average','Economy_Rate','Career_Wkts','Experience'
]

runs_features = [c for c in runs_features if c in df_feat.columns]
wkts_features = [c for c in wkts_features if c in df_feat.columns]

print("Runs features:", runs_features)
print("Wickets features:", wkts_features)


def train_regressors(df_model, features, target, prefix):
    """
    Train: LinearRegression, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor
    Use basic GridSearch for tree models (small grids).
    Save models, scaler, metrics, feature importances.
    """
    results = []
    fi_map = {} 

    X = df_model[features].fillna(0).astype(float)
    y = df_model[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(random_state=RANDOM_STATE),
        'gb': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'xgb': xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, verbosity=0)
    }

    grid_params = {
        'rf': {'n_estimators': [100, 200], 'max_depth': [4, 6]},
        'gb': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
        'xgb': {'n_estimators': [150, 250], 'learning_rate': [0.03, 0.07], 'max_depth': [3, 5]}
    }

    for name, model in models.items():
        start = time.time()
        print(f"\n➡️ Training {prefix.upper()} model: {name} ...")
        if name in grid_params:
            try:
                gs = GridSearchCV(model, grid_params[name], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
                gs.fit(X_train_s, y_train)
                best = gs.best_estimator_
                print(f"   • Best params for {prefix}_{name}: {gs.best_params_}")
                model_final = best
            except Exception as e:
                print(f"   ⚠ GridSearch failed for {prefix}_{name}, falling back to default. Error: {e}")
                model_final = model
                model_final.fit(X_train_s, y_train)
        else:
            model_final = model
            model_final.fit(X_train_s, y_train)

        # Predict & metrics
        y_pred = model_final.predict(X_test_s)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({'Model': f"{prefix}_{name}", 'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 4)})
        elapsed = time.time() - start
        print(f"   ✓ Finished {prefix}_{name} (time: {elapsed:.1f}s) — MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.4f}")

        # Save model
        model_path = os.path.join(OUT_DIR, f"{prefix}_{name}.pkl")
        joblib.dump(model_final, model_path)
        print(f"   • Saved model -> {model_path}")

        # Feature importances if available
        if hasattr(model_final, "feature_importances_"):
            fi = pd.Series(model_final.feature_importances_, index=features).sort_values(ascending=False)
            fi_file = os.path.join(OUT_DIR, f"{prefix}_{name}_feature_importance.csv")
            fi.to_csv(fi_file, header=True)
            fi_map[name] = fi_file
            print(f"   • Saved feature importances -> {fi_file}")

    scaler_path = os.path.join(OUT_DIR, f"{prefix}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   • Saved scaler -> {scaler_path}")

    metrics_df = pd.DataFrame(results)
    metrics_file = os.path.join(OUT_DIR, f"{prefix}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"   • Saved metrics -> {metrics_file}")

    return metrics_df, fi_map

print("\n=== TRAINING RUNS MODELS ===")
runs_metrics_df, runs_fi = train_regressors(df_runs, runs_features, 'Runs_next', 'runs')

print("\n=== TRAINING WICKETS MODELS ===")
wkts_metrics_df, wkts_fi = train_regressors(df_wkts, wkts_features, 'Wkts_next', 'wkts')


print("\n=== TRAINING PLAYER ROLE CLASSIFIER ===")

agg = df_feat.groupby('Player_Name').agg({
    'Runs_Scored': 'sum',
    'Wickets_Taken': 'sum',
    'Stumpings': 'sum',
    'Catches_Taken': 'sum'
}).rename(columns={'Runs_Scored': 'career_runs', 'Wickets_Taken': 'career_wkts'})

agg['is_keeper'] = agg['Stumpings'] > 0
runs_thresh = agg['career_runs'].quantile(0.4)
wkts_thresh = agg['career_wkts'].quantile(0.4)

def classify_role(row):
    if row['is_keeper']:
        return 'Wicketkeeper'
    if row['career_runs'] >= runs_thresh and row['career_wkts'] < wkts_thresh:
        return 'Batsman'
    if row['career_wkts'] >= wkts_thresh and row['career_runs'] < runs_thresh:
        return 'Bowler'
    if row['career_runs'] >= runs_thresh and row['career_wkts'] >= wkts_thresh:
        return 'All-rounder'
    return 'Batsman'

agg['role'] = agg.apply(classify_role, axis=1)
agg.reset_index()[['Player_Name', 'role']].to_csv(os.path.join(OUT_DIR, "player_roles_reference.csv"), index=False)

latest_rows = df_feat.sort_values(['Player_Name', 'Year']).groupby('Player_Name').tail(1).reset_index(drop=True)

clf_features = [
    'Matches_Batted','Runs_Scored','Batting_Average','Balls_Faced','Batting_Strike_Rate',
    'Centuries','Fours','Sixes','Matches_Bowled','Wickets_Taken','Balls_Bowled','Economy_Rate',
    'Career_Runs','Career_Wkts'
]
clf_features = [f for f in clf_features if f in latest_rows.columns]

clf_X = latest_rows[clf_features].fillna(0)
clf_y = agg.loc[latest_rows['Player_Name'], 'role'].values

if len(clf_X) >= 10:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(clf_X, clf_y, test_size=0.2, random_state=RANDOM_STATE)
    clf_scaler = StandardScaler().fit(Xc_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(clf_scaler.transform(Xc_train), yc_train)

    preds = clf.predict(clf_scaler.transform(Xc_test))

    acc = accuracy_score(yc_test, preds)
    report = classification_report(yc_test, preds, digits=3)
    cm = confusion_matrix(yc_test, preds, labels=sorted(set(yc_test)))

    print(f"\n🎯 Player Role Classifier Accuracy: {acc:.3f}")
    print("\n📋 Classification Report:\n", report)
    print("\n🧩 Confusion Matrix:\n", cm)

    joblib.dump(clf, os.path.join(OUT_DIR, "player_role_clf.pkl"))
    joblib.dump(clf_scaler, os.path.join(OUT_DIR, "player_role_scaler.pkl"))

    with open(os.path.join(OUT_DIR, "player_role_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(set(yc_test)),
                yticklabels=sorted(set(yc_test)))
    plt.title(f"Player Role Classifier Confusion Matrix (Accuracy: {acc:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "player_role_confusion_matrix.png"))
    plt.close()

latest_rows = df_feat.sort_values(['Player_Name','Year']).groupby('Player_Name').tail(1).reset_index(drop=True)
clf_features = [
    'Matches_Batted','Runs_Scored','Batting_Average','Balls_Faced','Batting_Strike_Rate',
    'Centuries','Fours','Sixes','Matches_Bowled','Wickets_Taken','Balls_Bowled','Economy_Rate','Career_Runs','Career_Wkts'
]
clf_features = [c for c in clf_features if c in latest_rows.columns]
clf_X = latest_rows[clf_features].fillna(0)
clf_y = agg.loc[latest_rows['Player_Name'], 'role'].values

if len(clf_X) >= 10: 
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(clf_X, clf_y, test_size=0.2, random_state=RANDOM_STATE)
    clf_scaler = StandardScaler().fit(Xc_train)
    Xc_train_s = clf_scaler.transform(Xc_train)
    Xc_test_s = clf_scaler.transform(Xc_test)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(Xc_train_s, yc_train)
    preds = clf.predict(Xc_test_s)
    acc = accuracy_score(yc_test, preds)
    print(f"   • Classifier test accuracy: {acc:.3f}")
    print("   • Classification report:\n", classification_report(yc_test, preds))
    joblib.dump(clf, os.path.join(OUT_DIR, "player_role_clf.pkl"))
    joblib.dump(clf_scaler, os.path.join(OUT_DIR, "player_role_scaler.pkl"))
    print("   • Saved classifier and scaler -> models/player_role_clf.pkl, models/player_role_scaler.pkl")
else:
    print("   ⚠ Not enough distinct players to train a reliable classifier. Skipping classifier training.")

all_metrics = pd.concat([
    runs_metrics_df.assign(Task='Runs'),
    wkts_metrics_df.assign(Task='Wickets')
], ignore_index=True)
all_metrics_file = os.path.join(OUT_DIR, "all_models_metrics.csv")
all_metrics.to_csv(all_metrics_file, index=False)
print(f"\n📊 Saved combined metrics -> {all_metrics_file}")

runs_metrics_df.to_csv(os.path.join(OUT_DIR, "runs_metrics.csv"), index=False)
wkts_metrics_df.to_csv(os.path.join(OUT_DIR, "wkts_metrics.csv"), index=False)
print("✅ All metrics saved.")

print("\n🎉 Training pipeline finished. All models, scalers, metrics, and processed data are saved in the 'models/' folder and as processed_ipl_full.csv.")
