import yfinance as yf
df = yf.download("AAPL", period="3y", interval="1d")
print(df)
