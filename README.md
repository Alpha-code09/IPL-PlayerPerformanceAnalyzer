# 🏏 IPL Player Performance Prediction System

**Team Project | Machine Learning | Python | Streamlit**

A machine learning–based system to predict **next-season IPL player performance** (runs and wickets) using historical IPL data from **2008 onwards**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Notes](#notes)
- [Team & Contributors](#team--contributors)
- [License](#license)

---

## 📖 Overview

This project predicts IPL player performance for the upcoming season using historical match and player statistics.  
Multiple machine learning models are trained and evaluated, and predictions are presented through an interactive Streamlit dashboard.

---

## ✨ Features

- **Machine Learning Models**
  - Linear Regression  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost  

- **Feature Engineering**
  - Rolling averages  
  - Career aggregates  
  - Consistency metrics  

- **Interactive Dashboard**
  - Player-wise performance analysis  
  - Role classification  
  - Model comparison  

- **Data-Driven Approach**
  - Historical IPL data from 2008 onwards  

---

## 📁 File Structure
IPL-FINALPROJECT/
|
├── app.py # Streamlit dashboard
├── train_all_models.py # Model training script
├── requirements.txt # Python dependencies
│
├── models/ # Saved ML models
├── ipl_stats.csv # Raw IPL historical data
├── processed_ipl_full.csv # Processed dataset
│
├── venv/ # Virtual environment (ignored in Git)
└── README.md # Project documentation


---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/USERNAME/REPO.git
cd IPL-FINALPROJECT

# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

