import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
np.random.seed(0)

import cvxpy as cp
from cvxpy import constraints

stock_tickers = ["TCS.NS","INFY.NS","RELIANCE.NS","ITC.NS","HDFCBANK.NS"]

raw = yf.download(stock_tickers, period = "3y",auto_adjust=True)
prices = raw["Close"]
prices = prices.dropna()
prices= prices[stock_tickers]
prices

plt.figure(figsize=(10,5))
for one_stock in prices.columns:
    plt.plot(prices.index, prices[one_stock],label = one_stock)
plt.xlabel("date")
plt.ylabel("price")
plt.legend()
plt.show()

daily_returns = prices.pct_change().dropna()
daily_returns.head()

average_daily_return = daily_returns.mean()* trading_days_per_year
average_daily_return
mu= average_daily_return*trading_days_per_year
mu


yearly_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
yearly_volatility
print("Yearly risk of each stock:\n",yearly_volatility*100)

Sigma = daily_returns.cov()*trading_days_per_year
Sigma

stock_names = list(prices.columns)
plt.figure(figsize=(6,5))
plt.imshow(Sigma.values, cmap ="Blues")
plt.colorbar(label="covariance")

plt.xticks(range(len(stock_names)), stock_names,rotation = 45, ha = "right")
plt.yticks(range(len(stock_names)), stock_names)
plt.title("Covariance Matrix Sigma(Darker = Move together more)")
plt.tight_layout()
plt.show()

number_of_stocks = len(stock_names)

equal_weights = np.ones(number_of_stocks)/number_of_stocks
print(mu)
portfolio_return = equal_weights @ mu
print(portfolio_return)
#Markowitz's portfolio variance formula
portfolio_variance = equal_weights @ Sigma.values @ equal_weights
print(portfolio_variance)
portfolio_risk = np.sqrt(portfolio_variance)
print(portfolio_risk)

average_individual_risk = yearly_volatility.mean()
print(average_individual_risk)

w = cp.Variable(number_of_stocks)
target_return = mu.mean()
portfolio_variance = cp.quad_form(w,Sigma.values)


constraints = [cp.sum(w)==1 ,w>=0, mu.values @ w>= target_return]
problem = cp.Problem(cp.Minimize(portfolio_variance),constraints)
problem.solve()
print("Best weights for this target:", (w.value * 100))
print("Resulting risk:",np.sqrt(w.value @ Sigma.values @ w.value))
print("Resulting return:",mu.values @ w.value)

target_returns= np.linspace(mu.min(),mu.max(),40)
target_returns

def best_weights_for_target(target):
  w = cp.Variable(number_of_stocks)
  constraints = [cp.sum(w)==1 ,w>=0, mu.values @ w>= target]
  problem = cp.Problem(cp.Minimize(portfolio_variance),constraints)
  problem.solve()
  if w.value is None:
    return None
  return w.value

frontier_risk = []
frontier_return = []
for target in target_returns:
  weights = best_weights_for_target(target)
  this_return = weights@ mu
  this_risk = np.sqrt(weights @Sigma.values @ weights)
  frontier_return.append(this_return)
  frontier_risk.append(this_risk)
frontier_risk = np.array(frontier_risk)
frontier_return = np.array(frontier_return)
print("We Computed",len(frontier_risk),"Portfolio along the frontier.")

w = cp.Variable(number_of_stocks)
constraints = [cp.sum(w)==1 ,w>=0]
problem = cp.Problem(cp.Minimize(cp.quad_form(w,Sigma.values)),constraints)
problem.solve()
print(w.value)

min_var_weights = w.value
min_var_return = min_var_weights @ mu
min_var_risk = np.sqrt(min_var_weights @ Sigma.values @ min_var_weights)
print("Min Variance Portfolio:")
print("Min Variance Return:",round(min_var_return*100,2),"%")
print("Min Variance Risk:", round(min_var_risk*100 , 2),"%")
