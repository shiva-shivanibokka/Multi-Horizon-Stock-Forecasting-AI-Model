# 📈 Multi-Horizon Stock Price Forecasting

This repository contains my **CSC-865: Artificial Intelligence** final project — **Multi-Horizon Stock Price Forecasting for Long-Term Financial Prediction**.  
I designed and compared **Transformer**, **Long Short-Term Memory (LSTM)**, **Recurrent Neural Network (RNN)**, and **Random Forest** models to predict S&P 500 stock prices over **five different time horizons**.

---

## 📂 Repository Structure
```plaintext
📁 AI_PROJECT_SB/
│
├── 📁 frontend/ # React frontend for user interaction
│ ├── 📁 public/ # Public assets
│ ├── 📁 src/ # Frontend source code
│ └── 📄 package.json # Frontend dependencies
│
├── 📁 lstm_final_project/ # LSTM model scripts
│ ├── 📄 train_lstm.py
│ ├── 📄 infer_lstm.py
│ └── 📄 *.pkl # Model weights (stored externally)
│
├── 📁 rf_final/ # Random Forest model scripts
│ ├── 📄 train_rf.py
│ ├── 📄 infer_rf.py
│ └── 📄 *.pkl
│
├── 📁 rnn_final/ # RNN model scripts
│ ├── 📄 train_rnn.py
│ ├── 📄 infer_rnn.py
│ └── 📄 *.pkl / *.pth
│
├── 📁 transformer_final/ # Transformer model scripts + Flask backend
│ ├── 📄 train_transformer.py
│ ├── 📄 infer_transformer.py
│ ├── 📄 app.py # Flask API
│ └── 📄 *.pkl
│
├── 📄 requirements.txt # Python dependencies
├── 📄 start_app.py # Backend entry point
└── 📄 README.md # This file
```
---

## 🎯 Project Goals
- Build four different forecasting models using the **same dataset** and preprocessing pipeline.
- Predict stock prices for:
  - **1 Day**
  - **1 Week**
  - **1 Month**
  - **6 Months**
  - **1 Year**
- Evaluate each model on:
  - **MSE, RMSE, MAE, R², MAPE**
  - **Directional Accuracy**
- Compare trade-offs between **classical ML** and **deep learning** for time series forecasting.

---

## 📊 Dataset & Features
- **Source**: [Yahoo Finance API](https://pypi.org/project/yfinance/)  
- **Coverage**: 3 years of daily stock data for all S&P 500 companies.
- **Base Features**: OHLCV (Open, High, Low, Close, Volume)
- **Technical Indicators**:
  - SMA10, SMA50, SMA200
  - RSI14
  - MOM1
  - ROC14
  - MACD
- **Preprocessing**:
  - Rolling windows (252+ days)
  - StandardScaler normalization
  - Windowed feature engineering for temporal modeling

---

## 🧠 Model Architectures

### 🌲 Random Forest
- Flattened `(252 days × 5 OHLCV)` window → `(1260,)`
- MultiOutputRegressor(RandomForestRegressor, n_estimators=100)
- Fast baseline, but **no temporal modeling**.

### 🔄 RNN
- Input: `(252, 12)` (OHLCV + indicators)
- Layers: 2 × RNN(128 units) → Dense(5)
- Strong short-term accuracy, weaker long-term memory.

### 🧬 LSTM
- Input: `(252, 12)`
- Layers: 2 × LSTM(128 units) + Dropout(0.2) → Dense(5)
- Better long-range tracking than RNN.

### 🧠 Transformer
- Input: `(756, 12)`
- Linear embedding → Positional Encoding → 2 × TransformerEncoder(4 heads) → Dense(5)
- Captures **local & global dependencies** with self-attention.
- Best overall performance.

---

## 📈 Results Summary

| Model         | 1 Day  | 1 Week | 1 Month | 6 Months | 1 Year |
|---------------|--------|--------|---------|----------|--------|
| Random Forest | 59.34% | 58.29% | 56.78%  | 45.98%   | 55.91% |
| RNN           | 98.75% | 97.72% | 96.71%  | 94.40%   | 91.64% |
| LSTM          | 96.54% | 96.56% | 94.36%  | 89.77%   | 87.23% |
| Transformer   | **98.80%** | **97.90%** | **96.51%** | **93.68%** | **92.23%** |

---

## 📦 Model Files
Due to GitHub’s file size limits, all trained models are stored externally.  
📥 **[Download Model Files from Google Drive](https://drive.google.com/drive/folders/1GX7uTlhHS2QI5kDmEMNV9fhnAk3MPwMp?usp=sharing)**  

**After download, place them into:**
rf_final/
transformer_final/

---

## 🚀 Running the Project

### Backend

pip install -r requirements.txt

python start_app.py

---

## 🔮 Future Work
- Add macroeconomic features (interest rates, GDP, inflation).
- Integrate sentiment analysis from financial news & social media.
- Explore hybrid statistical + deep learning models.
- Deploy real-time inference via Flask/FastAPI.
- Use financial metrics like Sharpe Ratio for profitability analysis.

## 📝 Self-Reflection
Over the course of this project, I undertook full responsibility for the entire workflow — from sourcing the data to implementing machine learning models and interpreting results. Since this was an individual effort, there were no teammates to divide the work with, which meant I had to wear every hat: data engineer, model builder, troubleshooter, and analyst.

I learned to manage computational resource limits while working with deep models like Transformers, optimize for available hardware, and debug complex architectures.

This experience deepened my understanding of sequence modeling, gave me practical exposure to backend inference + frontend integration, and motivated me to explore transfer learning and deployment in future projects.

Most importantly, it pushed me beyond my comfort zone and boosted my confidence to approach complex AI workflows.

# 📜 License
This code is available for viewing and educational purposes only. You may not use, copy, modify, or distribute it without written permission.

<p align="center"> ⭐ From <a href="https://github.com/shiva-shivanibokka">shiva-shivanibokka</a> </p> <p align="center"> <em>"Engineering Intelligence, One Model at a Time — Pioneering the Future of AI"</em> 🤖🚀 </p> ```
