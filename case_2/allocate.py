import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from universal import tools, algos
from universal.algos import CRP

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

''' We recommend that you change your train and test split '''
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        ''' Anything data you want to store between days must be stored in a class field '''
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.alloc = algos.OLMAR(window=5, eps=10)
        #BestMarkowitz()
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
    def allocate_portfolio(self, asset_prices):
        ''' asset_prices: np array of length 6, prices of the 6 assets on a particular day
            weights: np array of length 6, portfolio allocation for the next day
        '''        
        ### TODO Implement your code here
        result = self.alloc.run(asset_prices)
        weights = result.weights
        
        return weights

def grading(train_data, test_data):
    ''' Grading Script '''
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

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()