import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from itertools import product


data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class HMPPSO:
    def __init__(self, num_particles, num_sub_populations, max_iter, w, c1, c2):
        self.num_particles = 20
        self.num_sub_populations = 7
        self.max_iter = 30
        self.w = 0.8
        self.c1 = 1.2
        self.c2 = 1.2
        self.particles = []
        self.best_position = None
        self.best_fitness = -np.inf

    def initialize_particles(self, num_assets):
        self.particles = np.random.dirichlet(np.ones(num_assets), size=self.num_particles)
        self.velocities = np.zeros((self.num_particles, num_assets))
        self.fitness = np.zeros(self.num_particles)
        self.best_position = self.particles[0]
        self.best_fitness = -np.inf

    def evaluate_fitness(self, positions):
        returns = np.dot(positions, self.returns)
        risk = np.sqrt(np.diag(np.dot(positions, np.dot(self.cov_matrix, positions.T))))
        sharpe_ratio = np.zeros_like(returns)
        mask = risk != 0
        sharpe_ratio[mask] = returns[mask] / risk[mask]
        sharpe_ratio[~mask] = returns[~mask]
        return sharpe_ratio

    def update_velocities(self, best_positions):
        r1 = np.random.random((self.num_particles, 1))
        r2 = np.random.random((self.num_particles, 1))
        cognitive_velocity = self.c1 * r1 * (self.particles - self.velocities)
        social_velocity = self.c2 * r2 * (best_positions[np.arange(self.num_particles) % len(best_positions)] - self.particles)
        self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity

    def update_positions(self):
        self.particles = self.particles + self.velocities
        self.particles = self.particles / np.sum(self.particles, axis=1, keepdims=True)

    def optimize(self, returns, cov_matrix):
        self.returns = returns
        self.cov_matrix = cov_matrix
        num_assets = len(returns)
        self.initialize_particles(num_assets)

        for _ in range(self.max_iter):
            self.fitness = self.evaluate_fitness(self.particles)
            best_index = np.argmax(self.fitness)
            if self.fitness[best_index] > self.best_fitness:
                self.best_position = self.particles[best_index]
                self.best_fitness = self.fitness[best_index]
            best_positions = self.particles[np.arange(self.num_particles) % self.num_sub_populations == 0]
            self.update_velocities(best_positions)
            self.update_positions()

        return self.best_position

class Allocator:
    def __init__(self, train_data, num_particles, num_sub_populations, max_iter, w, c1, c2):
        self.train_data = train_data
        self.train_data = train_data
        self.num_particles = num_particles
        self.num_sub_populations = num_sub_populations
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.hmppso = HMPPSO(self.num_particles, self.num_sub_populations, self.max_iter, self.w, self.c1, self.c2)

    def allocate_portfolio(self, asset_prices):
        # returns = np.array(asset_prices) / np.array(self.train_data.iloc[-1])
        # cov_matrix = np.cov(self.train_data.T)
        # weights = self.hmppso.optimize(returns, cov_matrix)
        # return weights
        returns = np.array(asset_prices) / np.array(self.train_data.iloc[-1])
        cov_matrix = np.cov(self.train_data.T)
        weights = self.hmppso.optimize(returns, cov_matrix)
        return weights


def grading(train_data, test_data):
    weights = np.full(shape=(len(test_data.index), len(data.columns)), fill_value=0.0)
    alloc = Allocator(train_data, num_particles, num_sub_populations, max_iter, w, c1, c2)
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
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    return sharpe, capital, weights

# # Define the range of values for each hyperparameter
# num_particles_range = [20, 30, 40]
# num_sub_populations_range = [3, 5, 7]
# max_iter_range = [20, 25, 30]
# w_range = [0.6, 0.7, 0.8]
# c1_range = [1.2, 1.4, 1.6]
# c2_range = [1.2, 1.4, 1.6]

# # Create a list of all possible combinations of hyperparameters
# param_combinations = list(product(num_particles_range, num_sub_populations_range, max_iter_range, w_range, c1_range, c2_range))

# best_sharpe = -np.inf
# best_params = None

# # Perform grid search
# for params in param_combinations:
#     num_particles, num_sub_populations, max_iter, w, c1, c2 = params
#     sharpe, _, _ = grading(TRAIN, TEST, num_particles, num_sub_populations, max_iter, w, c1, c2)
#     if sharpe > best_sharpe:
#         best_sharpe = sharpe
#         best_params = params

# print("Best Sharpe Ratio:", best_sharpe)
# print("Best Hyperparameters:", best_params)

# # Use the best hyperparameters to allocate the portfolio and plot the results
# num_particles, num_sub_populations, max_iter, w, c1, c2 = best_params
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