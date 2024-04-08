import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.num_assets = 6
        self.population_size = 100
        self.iterations = 50
        self.update_range = 0.008 * np.pi
        self.quantum_matrix = np.random.dirichlet(np.ones(2 * self.num_assets), size=1).flatten()

    def generate_portfolio(self):
        p = self.quantum_matrix.reshape(self.num_assets, 2)
        if not np.allclose(np.sum(p, axis=1), 1):
            raise ValueError("Probabilities do not sum to 1")
        p = p.flatten()
        portfolio = np.random.choice([0, 1], size=(self.population_size, self.num_assets), p=p)
        portfolio = portfolio * 2 - 1  # Convert 0 to -1
        return portfolio
    
    def evaluate_portfolio(self, portfolio):
        returns = np.zeros(self.population_size)
        for i in range(self.population_size):
            weights = portfolio[i]
            capital = [1]
            for j in range(len(self.train_data) - 1):
                shares = capital[-1] * weights / np.array(self.train_data.iloc[j, :])
                balance = capital[-1] - np.dot(shares, np.array(self.train_data.iloc[j, :]))
                net_change = np.dot(shares, np.array(self.train_data.iloc[j+1, :]))
                capital.append(balance + net_change)
            capital = np.array(capital)
            returns[i] = (capital[-1] - capital[0]) / capital[0]
        return returns
    
    def update_quantum_matrix(self, best_portfolio, worst_portfolio):
        best_portfolio_extended = np.repeat(best_portfolio, 2)  # Repeat each weight for buy and sell
        worst_portfolio_extended = np.repeat(worst_portfolio, 2)  # Repeat each weight for buy and sell
        self.quantum_matrix += self.update_range * (best_portfolio_extended - worst_portfolio_extended)
        self.quantum_matrix = np.clip(self.quantum_matrix, 0, 1)

    
    def allocate_portfolio(self, asset_prices):
        self.running_price_paths = pd.concat([self.running_price_paths, pd.Series(asset_prices).to_frame().T], ignore_index=True)        
        for _ in range(self.iterations):
            portfolio = self.generate_portfolio()
            returns = self.evaluate_portfolio(portfolio)
            best_index = np.argmax(returns)
            worst_index = np.argmin(returns)
            best_portfolio = portfolio[best_index]
            worst_portfolio = portfolio[worst_index]
            self.update_quantum_matrix(best_portfolio, worst_portfolio)
        
        best_portfolio = portfolio[best_index]
        weights = best_portfolio / np.sum(np.abs(best_portfolio))
        
        return weights

def grading(train_data, test_data):
    # Grading Script
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