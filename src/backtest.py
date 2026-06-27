import numpy as np
import matplotlib.pyplot as plt

def generate_positions (predictions):
    positions = np.where(predictions == 1,1,-1)
    return positions

def compute_strategy_returns (positions,returns):
    strategy_returns = positions.flatten()*returns.flatten()
    return strategy_returns

def equity_curve(strategy_returns):
    equity = np.cumprod(1+strategy_returns)
    return equity

def total_return(equity):
    return equity[-1]-1

def annual_return(equity):

    years = len(equity)/252

    return equity[-1]**(1/years) - 1

def annual_volatility(strategy_returns):

    daily_vol = np.std(strategy_returns)

    return daily_vol*np.sqrt(252)

def sharpe_ratio(strategy_returns, equity):

    annual_ret = annual_return(equity)

    annual_vol = annual_volatility(strategy_returns)

    return annual_ret/annual_vol

def max_drawdown(equity):

    running_max = np.maximum.accumulate(equity)

    drawdown = (equity-running_max)/running_max

    return np.min(drawdown)

def win_rate(strategy_returns):

    wins = np.sum(strategy_returns > 0)

    return wins/len(strategy_returns)

def average_win(strategy_returns):

    wins = strategy_returns[strategy_returns>0]

    return np.mean(wins)

def average_loss(strategy_returns):

    losses = strategy_returns[strategy_returns<0]

    return np.mean(losses)

def profit_factor(strategy_returns):

    gross_profit = np.sum(strategy_returns[strategy_returns>0])

    gross_loss = -np.sum(strategy_returns[strategy_returns<0])

    return gross_profit/gross_loss

