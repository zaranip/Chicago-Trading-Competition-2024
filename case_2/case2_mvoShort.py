import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        
        # Calculate daily returns
        self.returns = self.train_data.pct_change().dropna()
        
        # Calculate expected returns and covariance matrix
        self.exp_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def mvo_with_short(self, risk_free_rate=0.02, max_short=-0.5):
        num_assets = len(self.exp_returns)
        
        def portfolio_stats(weights):
            portfolio_return = np.dot(weights, self.exp_returns)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return -sharpe_ratio
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((max_short, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]
        
        optimal_weights = optimize.minimize(portfolio_stats, initial_weights, method='SLSQP',
                                            bounds=bounds, constraints=constraints)
        
        return optimal_weights.x
        
    def allocate_portfolio(self, asset_prices):
        self.running_price_paths = self.running_price_paths.append(pd.Series(asset_prices, index=self.train_data.columns), ignore_index=True)
        
        # Update train_data with the new asset prices
        self.train_data = self.running_price_paths.copy()
        
        # Calculate daily returns
        self.returns = self.train_data.pct_change().dropna()
        
        # Calculate expected returns and covariance matrix
        self.exp_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        # Perform mean-variance optimization with short positions
        optimal_weights = self.mvo_with_short()
        
        return optimal_weights

def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

sharpe, capital, weights = grading(TRAIN, TEST)
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()