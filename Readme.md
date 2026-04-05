# 🏏 ChasePulse — Feel the Pulse of the Chase

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

> 🎯 A real-time ML-powered win probability engine for cricket chases with advanced match intelligence.

🔗 **Live App:** [chasepulse-nnf7rra8fnncspelxaxdjg.streamlit.app](https://chasepulse-nnf7rra8fnncspelxaxdjg.streamlit.app)

---

## 📸 Preview

| 🔮 Prediction Engine | 📊 Match Intelligence |
|--------------------|---------------------|
| Real-time win % with team comparison | Momentum, difficulty & insights |

---

## 🎯 Overview

**ChasePulse** is an end-to-end machine learning application that predicts win probability during cricket run chases in real time.

It uses match context such as runs left, balls remaining, wickets in hand, and run rates to dynamically estimate how the game evolves.

---

## 🚀 Features

### 🔮 Core Prediction
- Real-time win probability prediction
- Logistic Regression model
- Instant results based on match inputs

---

### 📊 Match Context

| Metric | Description |
|--------|------------|
| Runs Left | Remaining runs |
| Balls Left | Remaining deliveries |
| Wickets Left | Remaining wickets |
| CRR | Current Run Rate |
| RRR | Required Run Rate |

---

### 🧠 Match Intelligence

#### 🎯 Chase Difficulty
| Level | Meaning |
|------|--------|
| Easy | Comfortable chase |
| Moderate | Competitive |
| Tough | High pressure |
| Extreme | Nearly impossible |

---

#### ⚡ Momentum Indicator
- Tracks which team is ahead
- Based on CRR vs RRR
- Visual momentum bar

---

#### 📈 Match Phase
| Phase | Insight |
|------|--------|
| Powerplay | Aggressive scoring |
| Middle Overs | Control phase |
| Death Overs | High-risk finish |

---

#### 🚩 Key Milestones
- Next 50-run mark
- Halfway target tracking
- Win projection at current rate

---

### 🔬 Scenario Simulator

Simulates different next-over outcomes:

| Scenario | Effect |
|----------|--------|
| Wicket Maiden | Decreases probability |
| Average Over | Stable |
| Boundary Over | Increases probability |
| Collapse | Sharp drop |

---

### 📈 Win Probability Trend
- Future probability projection
- Based on required run rate
- Visual trend chart

---

### 🎨 UI Features
- Custom dark theme
- Gradient cards
- Team logos
- Clean dashboard layout

---

## 🧠 ML Pipeline

### Input Features
- Batting Team  
- Bowling Team  
- City  
- Runs Left  
- Balls Left  
- Wickets Left  
- Target  
- CRR  
- RRR  

### Output
- Win probability (classification)

---

## 🗂️ Project Structure

```
ChasePulse/
│
├── app.py                               # Main Streamlit application
├── deliveries.csv                       # Deliveries dataset
├── matches.csv                          # Matches dataset
├── IPL_Win_Probability_Predictor.ipynb  # Full ML notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | ML models & preprocessing |
| Plotly | Interactive charts |
| Streamlit | Web app framework & deployment |
| GitHub | Version control |
| Streamlit Cloud | Free hosting & deployment |

---

## 👤 Author

**Manan Jain**
