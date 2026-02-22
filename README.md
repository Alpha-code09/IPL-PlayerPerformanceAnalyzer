

# 🏏 IPL Player Performance Analyzer

A machine learning-powered analytics system that predicts next-season IPL player performance and classifies player roles using historical data from 2016–2024.

This project combines sports analytics, feature engineering, and advanced ML models to deliver actionable insights through an interactive dashboard.

----------

## 🚀 Overview

The IPL Player Performance Analyzer helps answer questions like:

-   🔮 How many runs will a player score next season?
    
-   🎯 How many wickets can a bowler take?
    
-   🏷️ What is a player's primary role?
    
-   📊 Who are the predicted top performers?
    

The system uses historical IPL data and applies multiple machine learning models to generate accurate predictions and insights.

----------

## ✨ Features

-   📈 Runs Prediction (Regression)
    
-   🎯 Wickets Prediction (Regression)
    
-   🏷️ Player Role Classification (Multi-class)
    
-   🔍 Rolling 3-Year Form Analysis
    
-   📊 Model Comparison Dashboard
    
-   🏆 Top 5 Predicted Run Scorers
    
-   🏏 Top 5 Predicted Wicket Takers
    
-   📤 Exportable Results
    
-   🌐 Interactive Streamlit Dashboard
    

----------

## 🧠 Machine Learning Models

### Regression

-   Linear Regression
    
-   Random Forest Regressor
    
-   Gradient Boosting Regressor
    
-   XGBoost Regressor (Best Performing)
    

### Classification

-   Random Forest Classifier
    

Tree-based models significantly outperform linear baselines due to non-linear relationships in cricket performance data.

----------

## 📊 Model Performance

### Runs Prediction (2024 Test Season)

Model

R² Score

Linear Regression

0.58

Random Forest

0.71

Gradient Boosting

0.74

XGBoost

**0.76**

### Wickets Prediction

Model

R² Score

Linear Regression

0.52

Random Forest

0.67

Gradient Boosting

0.70

XGBoost

**0.73**

### Role Classification

-   Accuracy: **88.4%**
    

----------

## 🛠 Tech Stack

-   Python 3.8+
    
-   Pandas
    
-   NumPy
    
-   Scikit-learn
    
-   XGBoost
    
-   Matplotlib
    
-   Plotly
    
-   Streamlit
    
-   Joblib
    

----------

## 📂 Project Structure

```
.
├── train_all_models.py        # Model training pipeline
├── app.py                    # Streamlit dashboard
├── models/                    # Serialized trained models
├── processed_ipl_full.csv     # Feature-engineered dataset
├── requirements.txt
└── README.md

```

----------

## 🔄 Workflow

### Training Phase

1.  Load raw IPL dataset
    
2.  Perform preprocessing & feature engineering
    
3.  Apply temporal split (2016–2023 train, 2024 test)
    
4.  Train multiple models
    
5.  Evaluate and save best-performing models
    

### Inference Phase

1.  Launch Streamlit dashboard
    
2.  Load trained models
    
3.  Select player or analysis type
    
4.  Generate real-time predictions
    
5.  Visualize insights interactively
    

----------

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ipl-player-performance-analyzer.git
cd ipl-player-performance-analyzer

```

Install dependencies:

```bash
pip install -r requirements.txt

```

----------

## ▶️ Run the Project

### Train Models

```bash
python train_all_models.py

```

### Launch Dashboard

```bash
streamlit run app.py

```

The dashboard will open in your browser at:

```
http://localhost:8501

```

----------

## 📌 Key Insights

-   Recent form (rolling averages) is the strongest predictor of future performance.
    
-   Tree-based ensemble models outperform linear models significantly.
    
-   Experience metrics (career aggregates) improve prediction stability.
    
-   Hybrid roles (All-rounders) are hardest to classify accurately.
    

----------

## 🔮 Future Improvements

-   Add venue-level features
    
-   Incorporate injury history
    
-   Player age and fitness metrics
    
-   Match-level granular analysis
    
-   LSTM-based time series modeling
    
-   Auction price prediction module
    

----------

## 📜 License

This project is open-source and available under the MIT License.

----------

