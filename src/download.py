import yfinance as yf

# Downloading Apple stock data
df = yf.download("AAPL", start = "2018-01-01", end = "2025-01-01")
df.columns = df.columns.droplevel(1)
df.reset_index(inplace=True)
print(df.head())

# Save to CSV
df.to_csv("AAPL.csv", index=False)

