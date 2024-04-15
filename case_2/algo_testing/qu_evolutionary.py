import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# This code is based off of https://www.hindawi.com/journals/mpe/2017/4197914/ 

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)


class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        
        # Initialize GA parameters
        self.population_size = 50
        self.num_generations = 100
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        
        # Calculate market capitalization for each industry sector
        self.industry_market_cap = self.train_data.groupby('industry_sector')['market_cap'].sum()
        
    def calculate_priority(self, stock):
        beta_std_error = stock['beta_std_error']
        avg_trading_amount = stock['avg_trading_amount']
        avg_market_cap = stock['avg_market_cap']
        
        v1, v2, v3 = 1, 1, 1  # Weights for priority calculation
        
        priority = v1 * (1 / beta_std_error) + v2 * avg_trading_amount + v3 * avg_market_cap
        return priority
    
    def select_stocks(self):
        selected_stocks = []
        remaining_sectors = self.industry_market_cap.copy()
        
        while len(selected_stocks) < 30:  # Select 30 stocks for the portfolio
            sector = remaining_sectors.idxmax()
            sector_stocks = self.train_data[self.train_data['industry_sector'] == sector]
            
            priorities = sector_stocks.apply(self.calculate_priority, axis=1)
            selected_stock = priorities.idxmax()
            
            selected_stocks.append(selected_stock)
            remaining_sectors.drop(sector, inplace=True)
        
        return selected_stocks
    
    def optimize_weights(self, selected_stocks):
        # Implement GA optimization for weights
        population = np.random.dirichlet(np.ones(len(selected_stocks)), size=self.population_size)
        best_weights = None
        best_fitness = float('-inf')
        
        for generation in range(self.num_generations):
            fitness_scores = []
            
            for weights in population:
                portfolio_return = np.sum(self.train_data.loc[selected_stocks].pct_change().mean() * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.train_data.loc[selected_stocks].pct_change().cov(), weights)))
                fitness = portfolio_return / portfolio_std
                fitness_scores.append(fitness)
            
            if np.max(fitness_scores) > best_fitness:
                best_fitness = np.max(fitness_scores)
                best_weights = population[np.argmax(fitness_scores)]
            
            # Selection
            selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=fitness_scores/np.sum(fitness_scores))
            population = population[selected_indices]
            
            # Crossover
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    parent1, parent2 = population[i], population[i+1]
                    child1, child2 = self.crossover(parent1, parent2)
                    population[i], population[i+1] = child1, child2
            
            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    population[i] = self.mutate(population[i])
        
        return best_weights
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    
    def mutate(self, individual):
        mutation_point = np.random.randint(len(individual))
        individual[mutation_point] = np.random.uniform(0, 1)
        individual /= np.sum(individual)
        return individual
    
    def allocate_portfolio(self, asset_prices):
        self.running_price_paths = self.running_price_paths.append(pd.Series(asset_prices, index=self.train_data.columns), ignore_index=True)
        
        selected_stocks = self.select_stocks()
        optimized_weights = self.optimize_weights(selected_stocks)
        
        weights = pd.Series(optimized_weights, index=selected_stocks)
        weights = weights.reindex(self.train_data.columns, fill_value=0)
        
        return weights.values


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
        net_change = np.dot(shares, np.array(test_data.iloc[i+1, :]))
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