import os
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import requests
import re

from torch import nn

# ——— Load model class ———
from train_transformer import TimeSeriesTransformer

# ——— Constants ———
ARTIFACT_DIR = os.path.dirname(__file__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(ARTIFACT_DIR, "transformer_multi_horizon.pth")
META_PATH = os.path.join(ARTIFACT_DIR, "transformer_meta.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler_feat.pkl")

# ——— Flask setup ———
app = Flask(__name__)
CORS(app)

# ——— Load artifacts ———
model = TimeSeriesTransformer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

meta = joblib.load(META_PATH)
WINDOW = meta["window"]
HORIZONS = meta["horizons"]
scaler_feat = joblib.load(SCALER_PATH)
scaler_ret = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_ret.pkl"))

# ——— Inference Function ———
def predict_stock_prices(features_scaled):
    """Proper inference using the transformer model following the inference file pattern"""
    try:
        # Convert to tensor and add batch dimension
        x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Get model predictions
        with torch.no_grad():
            output = model(x).cpu().numpy()
        
        # Inverse transform using the return scaler (following inference file pattern)
        output_unscaled = scaler_ret.inverse_transform(output)
        
        print(f"Model output shape: {output.shape}")
        print(f"Unscaled output shape: {output_unscaled.shape}")
        print(f"Horizons: {HORIZONS}")
        
        return output_unscaled.flatten()
        
    except Exception as e:
        print(f"Error in inference: {e}")
        return None

# ——— Feature engineering ———
def compute_technicals(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    # Handle division by zero
    rs = rs.fillna(0)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['MOM_1'] = df['Close'].diff(1)
    df['ROC_14'] = df['Close'].pct_change(14).fillna(0)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    return df.dropna()



# ——— Recommendation System ———
def generate_recommendation(current_price, predictions, technical_data):
    """Generate buy/hold/sell recommendation based on multiple factors"""
    
    # Calculate price changes
    price_changes = {}
    for horizon, pred_price in predictions.items():
        price_changes[horizon] = ((pred_price - current_price) / current_price) * 100
    
    # Technical analysis signals
    sma_50 = technical_data.get('sma_50', [])
    sma_200 = technical_data.get('sma_200', [])
    
    if sma_50 and sma_200:
        current_sma_50 = float(sma_50[-1]) if sma_50 else float(current_price)
        current_sma_200 = float(sma_200[-1]) if sma_200 else float(current_price)
        
        # Golden/Death cross
        golden_cross = current_sma_50 > current_sma_200
        price_above_sma50 = float(current_price) > current_sma_50
        price_above_sma200 = float(current_price) > current_sma_200
    else:
        golden_cross = False
        price_above_sma50 = True
        price_above_sma200 = True
    
    # Scoring system
    score = 0
    reasons = []
    
    # Price prediction factor (60% weight)
    short_term_change = price_changes.get('1d', 0)
    long_term_change = price_changes.get('1y', 0)
    
    if short_term_change > 2:
        score += 2
        reasons.append(f"Strong short-term growth predicted (+{short_term_change:.1f}%)")
    elif short_term_change > 0:
        score += 1
        reasons.append(f"Moderate short-term growth (+{short_term_change:.1f}%)")
    elif short_term_change < -2:
        score -= 2
        reasons.append(f"Significant short-term decline predicted ({short_term_change:.1f}%)")
    elif short_term_change < 0:
        score -= 1
        reasons.append(f"Moderate short-term decline ({short_term_change:.1f}%)")
    
    if long_term_change > 10:
        score += 3
        reasons.append(f"Excellent long-term outlook (+{long_term_change:.1f}%)")
    elif long_term_change > 5:
        score += 2
        reasons.append(f"Good long-term potential (+{long_term_change:.1f}%)")
    elif long_term_change < -10:
        score -= 3
        reasons.append(f"Poor long-term outlook ({long_term_change:.1f}%)")
    elif long_term_change < -5:
        score -= 2
        reasons.append(f"Concerning long-term trend ({long_term_change:.1f}%)")
    
    # Technical analysis factor (40% weight)
    if golden_cross:
        score += 1
        reasons.append("Golden cross detected (bullish signal)")
    
    if price_above_sma50 and price_above_sma200:
        score += 1
        reasons.append("Price above both moving averages")
    elif not price_above_sma50 and not price_above_sma200:
        score -= 1
        reasons.append("Price below both moving averages")
    
    # Generate recommendation
    if score >= 3:
        recommendation = "BUY"
        confidence = "High"
    elif score >= 1:
        recommendation = "BUY"
        confidence = "Medium"
    elif score >= -1:
        recommendation = "HOLD"
        confidence = "Medium"
    elif score >= -3:
        recommendation = "SELL"
        confidence = "Medium"
    else:
        recommendation = "SELL"
        confidence = "High"
    
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "price_changes": price_changes
    }

# ——— Company Fundamentals ———
def get_company_fundamentals(ticker):
    """Get company fundamentals using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        # Get key metrics - Top 10 most important fundamentals
        fundamentals = {
            "company_info": {
                "name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "market_cap": info.get('marketCap', 0),
                "website": info.get('website', 'N/A'),
                "country": info.get('country', 'N/A')
            },
            "valuation_metrics": {
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "forward_pe": info.get('forwardPE', 'N/A'),
                "price_to_book": info.get('priceToBook', 'N/A'),
                "price_to_sales": info.get('priceToSalesTrailing12Months', 'N/A')
            },
            "financial_metrics": {
                "revenue": info.get('totalRevenue', 0),
                "net_income": info.get('netIncomeToCommon', 0),
                "total_cash": info.get('totalCash', 0),
                "total_debt": info.get('totalDebt', 0),
                "debt_to_equity": info.get('debtToEquity', 'N/A'),
                "current_ratio": info.get('currentRatio', 'N/A')
            },
            "dividend_info": {
                "dividend_rate": info.get('dividendRate', 0),
                "dividend_yield": info.get('dividendYield', 0)
            },
            "growth_metrics": {
                "revenue_growth": info.get('revenueGrowth', 'N/A'),
                "earnings_growth": info.get('earningsGrowth', 'N/A'),
                "profit_margins": info.get('profitMargins', 'N/A')
            },
            "trading_info": {
                "beta": info.get('beta', 'N/A'),
                "fifty_two_week_change": info.get('52WeekChange', 'N/A'),
                "shares_outstanding": info.get('sharesOutstanding', 0)
            }
        }
        
        # Format large numbers
        for category in fundamentals.values():
            if isinstance(category, dict):
                for key, value in category.items():
                    if isinstance(value, (int, float)) and value != 0:
                        if value >= 1e9:
                            category[key] = f"${value/1e9:.2f}B"
                        elif value >= 1e6:
                            category[key] = f"${value/1e6:.2f}M"
                        elif value >= 1e3:
                            category[key] = f"${value/1e3:.2f}K"
                        elif key in ['pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book', 'price_to_sales', 
                                   'enterprise_to_revenue', 'enterprise_to_ebitda', 'debt_to_equity', 
                                   'current_ratio', 'quick_ratio', 'payout_ratio', 'revenue_growth', 
                                   'earnings_growth', 'profit_margins', 'operating_margins', 'beta', 
                                   'fifty_two_week_change', 'shares_percent_shares_out']:
                            if isinstance(value, (int, float)):
                                category[key] = f"{value:.2f}"
        
        return fundamentals
        
    except Exception as e:
        return {
            "error": f"Failed to fetch fundamentals: {str(e)}",
            "company_info": {},
            "valuation_metrics": {},
            "financial_metrics": {},
            "dividend_info": {},
            "growth_metrics": {},
            "trading_info": {}
        }

# ——— Enhanced Prediction Route ———
@app.route("/api/predict/<ticker>", methods=["GET"])
def predict(ticker):
    try:
        yf_sym = ticker.upper().replace(".", "-")
        df = yf.download(yf_sym, period="5y", interval="1d", progress=False)

        if df.shape[0] < WINDOW + 1:
            return jsonify({"error": "Not enough data for prediction."})

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = compute_technicals(df)
        df = df.dropna()

        if df.shape[0] < WINDOW:
            return jsonify({"error": "Insufficient technicals for inference."})

        features = df[["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50",
                       "SMA_200", "RSI_14", "MOM_1", "ROC_14", "MACD"]].tail(WINDOW).values

        features_scaled = scaler_feat.transform(features)
        
        # Use proper inference function
        output = predict_stock_prices(features_scaled)
        
        if output is None:
            return jsonify({"error": "Failed to generate predictions."})
        
        close_price = float(df["Close"].iloc[-1])
        preds = {k: round(close_price * (1 + output[i]), 2) for i, k in enumerate(HORIZONS)}
        
        # Get technical data for chart
        df_plot = df.tail(252)
        technical_data = {
            "close": df_plot["Close"].values.tolist(),
            "sma_50": df_plot["SMA_50"].values.tolist(),
            "sma_200": df_plot["SMA_200"].values.tolist(),
            "dates": [d.strftime('%Y-%m-%d') for d in df_plot.index]
        }
        
        # Generate recommendation with proper error handling
        try:
            recommendation = generate_recommendation(close_price, preds, technical_data)
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            recommendation = {
                "recommendation": "HOLD",
                "confidence": "Low",
                "score": 0,
                "reasons": ["Unable to generate recommendation due to data issues"],
                "price_changes": {}
            }
        
        # Get company fundamentals
        fundamentals = get_company_fundamentals(ticker)

        return jsonify({
            "ticker": ticker.upper(),
            "current_price": float(round(close_price, 2)),
            "predictions": {k: float(v) for k, v in preds.items()},
            "recommendation": recommendation,
            "technical_data": technical_data,
            "fundamentals": fundamentals
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ——— Health check route ———
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000) 