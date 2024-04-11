import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)
SYMBOLS = data.keys()
print(SYMBOLS)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
        #  def __init__(self, train_data, eps=0.5, C=500, variant=0, max_weight=0.3):

    def __init__(self, train_data, eps=0.5, C=500, variant=0, max_weight=0.25):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.eps = eps
        self.C = C
        self.variant = variant
        self.max_weight = max_weight
        self.last_b = self.init_weights(train_data.columns)

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def update(self, b, x, eps, C):
        x_mean = np.mean(x)
        le = max(0.0, np.dot(b, x) - eps)
        if self.variant == 0:
            lam = le / np.linalg.norm(x - x_mean) ** 2
        elif self.variant == 1:
            lam = min(C, le / np.linalg.norm(x - x_mean) ** 2)
        elif self.variant == 2:
            lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / C)
        lam = min(100000, lam)
        b = b - lam * (x - x_mean)
        b = self.simplex_proj(b)
        b = np.minimum(b, self.max_weight)
        b /= np.sum(b)
        return b

    def simplex_proj(self, b):
        m = len(b)
        bget = False
        s = sorted(b, reverse=True)
        tmpsum = 0.0
        for ii in range(m - 1):
            tmpsum = tmpsum + s[ii]
            tmax = (tmpsum - 1) / (ii + 1)
            if tmax >= s[ii + 1]:
                bget = True
                break
        if not bget:
            tmax = (tmpsum + s[m - 1] - 1) / m
        return np.maximum(b - tmax, 0.0)

    def allocate_portfolio(self, asset_prices):
        new_data = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_data], ignore_index=True)
        self.last_b = self.update(self.last_b, asset_prices / asset_prices.mean(), self.eps, self.C)
        return self.last_b

def grading(train_data, test_data):
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
        print(np.mean(returns))
        print(np.std(returns))
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
    print(weights)
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