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
        
        # Initialize the population
        self.population_size = 50
        self.num_assets = self.train_data.shape[1]
        self.population = np.random.uniform(-1, 1, (self.population_size, self.num_assets))
        
        # Set the number of generations
        self.generations = 100
        
        # Set the mutation rate
        self.mutation_rate = 0.1
        
    def trend_ratio(self, weights):
        # Calculate the trend ratio for a given set of weights
        weighted_returns = np.sum(self.train_data.pct_change().dropna() * weights, axis=1)
        expected_return = np.mean(weighted_returns)
        risk = np.std(weighted_returns)
        
        if risk != 0:
            ratio = expected_return / risk
            # Ensure the ratio is non-negative
            return max(0, ratio)
        else:
            return 0
        
    def allocate_portfolio(self, asset_prices):
        self.running_price_paths = self.running_price_paths.append(pd.Series(asset_prices, index=self.train_data.columns), ignore_index=True)
        
        # Update the training data
        self.train_data = self.running_price_paths.copy()
        
        # Evolve the population
        for _ in range(self.generations):
            # Evaluate the fitness of each individual
            fitness_scores = [self.trend_ratio(weights) for weights in self.population]
            
            # Select the parents for reproduction
            parent_indices = np.random.choice(self.population_size, size=2, replace=False, p=fitness_scores/np.sum(fitness_scores))
            parents = self.population[parent_indices]
            
            # Perform crossover
            child = np.mean(parents, axis=0)
            
            # Perform mutation
            if np.random.rand() < self.mutation_rate:
                mutation_index = np.random.randint(self.num_assets)
                child[mutation_index] = np.random.uniform(-1, 1)
            
            # Replace the worst individual with the child
            worst_index = np.argmin(fitness_scores)
            self.population[worst_index] = child
        
        # Select the best individual as the portfolio weights
        best_index = np.argmax([self.trend_ratio(weights) for weights in self.population])
        weights = self.population[best_index]
        
        return weights


def grading(train_data, test_data): 
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