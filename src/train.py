from preprocess import X, y, returns
from logistic import train, predict
import numpy as np
import matplotlib.pyplot as plt
from metrics import *
from backtest import *

split = int(0.8*len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

returns_train = returns[:split]
returns_test = returns[split:]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

w, b, losses = train(X_train, y_train)

predictions = predict(X_test, w, b)
positions = generate_positions(predictions)
strategy_returns = compute_strategy_returns(positions,returns_test)

equity = equity_curve(strategy_returns)

print(f"Total Return        : {100*total_return(equity):.2f}%")
print(f"Annual Return       : {100*annual_return(equity):.2f}%")
print(f"Annual Volatility   : {100*annual_volatility(strategy_returns):.2f}%")
print(f"Sharpe Ratio        : {sharpe_ratio(strategy_returns,equity):.2f}")
print(f"Maximum Drawdown    : {100*max_drawdown(equity):.2f}%")
print(f"Win Rate            : {100*win_rate(strategy_returns):.2f}%")
print(f"Average Win         : {100*average_win(strategy_returns):.2f}%")
print(f"Average Loss        : {100*average_loss(strategy_returns):.2f}%")
print(f"Profit Factor       : {profit_factor(strategy_returns):.2f}")

print(predictions[:10])

print(w)

print(b)

TP, TN, FP, FN = confusion_matrix(y_test, predictions)
acc = accuracy(TP, TN, FP, FN)
prec = precision(TP, FP)
rec = recall(TP, FN)
f1 = f1_score(prec, rec)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("TP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

print(np.mean(y_train))

print("Training positive rate:", np.mean(y_train))
print("Test positive rate:", np.mean(y_test))

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(equity)
plt.title("Logistic Regression Trading Strategy")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()