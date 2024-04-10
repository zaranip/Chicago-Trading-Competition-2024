import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = pd.DataFrame(train_data.copy())
        self.returns = self.train_data.pct_change().dropna()
        self.num_assets = len(self.train_data.columns)
        self.init_guess = np.array(self.num_assets * [1. / self.num_assets])
        self.cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # self.bounds = tuple((0, 1) for x in range(self.num_assets))

    def objective_sharpe(self, weights, data):
        prices = data.to_numpy()
        returns = np.diff(prices, axis=0) / prices[:-1]
        weighted_returns = np.dot(returns, weights)
        portfolio_returns = np.sum(weighted_returns)


        # Calculate the daily return of the portfolio
        sharpe_ratio = np.mean(weighted_returns) / np.std(weighted_returns)
        
        # capital = [1]
        # for i in range(len(data)-7, len(data) - 1):
        #     shares = capital[-1] * weights / np.array(data.iloc[i, :])
        #     balance = capital[-1] - np.dot(shares, np.array(data.iloc[i, :]))
        #     net_change = np.dot(shares, np.array(data.iloc[i + 1, :]))
        #     capital.append(balance + net_change)
        # capital = np.array(capital)
        # returns = (capital[1:] - capital[:-1]) / capital[:-1]


        # Add a regularization term to encourage diversification
        # diversification_penalty = np.sum(np.square(weights - 1 / self.num_assets))
        

        return -sharpe_ratio + 0.001* np.std(weighted_returns)

    def allocate_portfolio(self, asset_prices):
        self.running_price_paths = self.running_price_paths.append(asset_prices, ignore_index=True)
        self.train_data = self.train_data.append(asset_prices, ignore_index=True)
        self.returns = self.train_data.pct_change().dropna()

        # Perform optimization with different optimization algorithms
        best_weights = None
        best_sharpe_ratio = -np.inf

        optimization_methods = ['SLSQP', 'L-BFGS-B', 'TNC', 'CG']
        for method in optimization_methods:
            opt_results = minimize(self.objective_sharpe, self.init_guess, args=self.train_data,
                                method=method, constraints=self.cons)
            weights = opt_results.x

            sharpe_ratio = -self.objective_sharpe(weights, self.train_data)
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                best_weights = weights

        self.init_guess = best_weights
        best_weights = best_weights / np.max(abs(best_weights)) if np.max(abs(best_weights)) > 1 else best_weights
        best_weights = best_weights / np.sum(abs(best_weights)) if np.sum(abs(best_weights)) > 0 else best_weights
        print(best_weights)
        return best_weights
    
def grading(train_data, test_data):
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0, len(test_data)):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")

    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        print("Return: ", capital[-1] - capital[0])
        print(np.mean(returns), np.std(returns))
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
print(sharpe)

# plt.figure(figsize=(10, 6), dpi=80)
# plt.title("Capital")
# plt.plot(np.arange(len(TEST)), capital)
# plt.show()

# plt.figure(figsize=(10, 6), dpi=80)
# plt.title("Weights")
# plt.plot(np.arange(len(TEST)), weights)
# plt.legend(TEST.columns)
# plt.show()