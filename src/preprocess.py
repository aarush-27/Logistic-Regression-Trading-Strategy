import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

csv_path = BASE_DIR / "data" / "AAPL.csv"

print(csv_path)

df = pd.read_csv(csv_path)

df["Return"] = (
    df["Close"]-df["Close"].shift(1)
                )/df["Close"].shift(1)

df["MA5"] = df["Close"].rolling(5).mean()

df["MA20"] = df["Close"].rolling(20).mean()

df["Volatility"] = df["Return"].rolling(10).std()

df["Change"] = df["Close"].diff()
df["Gain"] = df["Change"].clip(lower = 0)
df["Loss"] = (-df["Change"]).clip(lower=0)
df["AvgGain"] = df["Gain"].rolling(14).mean()
df["AvgLoss"] = df["Loss"].rolling(14).mean()
df["RS"] = df["AvgGain"]/df["AvgLoss"]
df["RSI"] = 100-(100/(1+df["RS"]))
df["FutureReturn"] = df["Return"].shift(-1)

df["Target"] = (df["Close"].shift(-1)>df["Close"]).astype(int)

df = df.dropna()

X = df[["Return","RSI"]]
y = df["Target"]
returns = df["FutureReturn"]

X = X.to_numpy()
y = y.to_numpy().reshape(-1,1)
returns = returns.to_numpy().reshape(-1,1)