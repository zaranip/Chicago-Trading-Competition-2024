import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os, sys
from universal import tools, algos
from universal.algos import CRP

def log_to_file(log_file_path):
    """
    Redirects the console output to a log file.
    Prints to both the console and the log file.
    """
    class Logger(object):
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log_file = open(log_file, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)

        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    log_file_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)

    sys.stdout = Logger(log_file_path)

log_to_file("final/final_metric.txt")



data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)
SYMBOLS = data.keys()

# TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

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

# class Allocator():
#     def __init__(self, data, train_data):
#         ''' Anything data you want to store between days must be stored in a class field '''
#         self.running_price_paths = train_data.copy()
#         self.data = data
#         self.train_data = train_data.copy()
#         self.i = 0
#         self.alloc = algos.BestMarkowitz()
        
#         # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
#     def allocate_portfolio(self, asset_prices):
#         ''' asset_prices: np array of length 6, prices of the 6 assets on a particular day
#             weights: np array of length 6, portfolio allocation for the next day
#         '''        
#         ### TODO Implement your code here
#         self.train_data = self.train_data.append(asset_prices, ignore_index=True)
#         result = self.alloc.run(self.data)
#         weights = result.weights.iloc[self.i,:]
        
#         self.i += 1
#         return weights


# class HMPPSO:
#     def __init__(self):
#         self.num_particles = 20
#         self.num_sub_populations = 7
#         self.max_iter = 30
#         self.w = 0.8
#         self.c1 = 1.2
#         self.c2 = 1.2
#         self.particles = []
#         self.best_position = None
#         self.best_fitness = -np.inf

#     def initialize_particles(self, num_assets):
#         self.particles = np.random.dirichlet(np.ones(num_assets), size=self.num_particles)
#         self.velocities = np.zeros((self.num_particles, num_assets))
#         self.fitness = np.zeros(self.num_particles)
#         self.best_position = self.particles[0]
#         self.best_fitness = -np.inf

#     def evaluate_fitness(self, positions):
#         returns = np.dot(positions, self.returns)
#         risk = np.sqrt(np.diag(np.dot(positions, np.dot(self.cov_matrix, positions.T))))
#         sharpe_ratio = np.zeros_like(returns)
#         mask = risk != 0
#         sharpe_ratio[mask] = returns[mask] / risk[mask]
#         sharpe_ratio[~mask] = returns[~mask]
#         return sharpe_ratio

#     def update_velocities(self, best_positions):
#         r1 = np.random.random((self.num_particles, 1))
#         r2 = np.random.random((self.num_particles, 1))
#         cognitive_velocity = self.c1 * r1 * (self.particles - self.velocities)
#         social_velocity = self.c2 * r2 * (best_positions[np.arange(self.num_particles) % len(best_positions)] - self.particles)
#         self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity

#     def update_positions(self):
#         self.particles = self.particles + self.velocities
#         self.particles = self.particles / np.sum(self.particles, axis=1, keepdims=True)

#     def optimize(self, returns, cov_matrix):
#         self.returns = returns
#         self.cov_matrix = cov_matrix
#         num_assets = len(returns)
#         self.initialize_particles(num_assets)

#         for _ in range(self.max_iter):
#             self.fitness = self.evaluate_fitness(self.particles)
#             best_index = np.argmax(self.fitness)
#             if self.fitness[best_index] > self.best_fitness:
#                 self.best_position = self.particles[best_index]
#                 self.best_fitness = self.fitness[best_index]
#             best_positions = self.particles[np.arange(self.num_particles) % self.num_sub_populations == 0]
#             self.update_velocities(best_positions)
#             self.update_positions()

#         return self.best_position

# class Allocator:
#     def __init__(self, data, train_data):
#         self.data = data
#         self.train_data = train_data
#         self.hmppso = HMPPSO()

#     def allocate_portfolio(self, asset_prices):
#         # returns = np.array(asset_prices) / np.array(self.train_data.iloc[-1])
#         # cov_matrix = np.cov(self.train_data.T)
#         # weights = self.hmppso.optimize(returns, cov_matrix)
#         # return weights
#         returns = np.array(asset_prices) / np.array(self.train_data.iloc[-1])
#         cov_matrix = np.cov(self.train_data.T)
#         weights = self.hmppso.optimize(returns, cov_matrix)
#         return weights


def grading(data, method):
    for window_size in [2520]:
        avg_log_size = 252
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in [0.75, 0.8, 0.85]:
            train_test_ratio = k
            sharpe_ratios = []
            temp = []
            capitals = []
            weights_list = []
            print("Method: ", method, "Train Test Ratio: ", train_test_ratio, "Window Size: ", window_size)

            for i in range(data.shape[0]-window_size+1):
                window_data = data.iloc[i:i+window_size]

                window_train_data = window_data.iloc[:int(train_test_ratio*len(window_data))]
                window_test_data = window_data.iloc[int(train_test_ratio*len(window_data)):]

                weights = np.full(shape=(len(window_test_data.index), len(SYMBOLS)), fill_value=0.0)
                alloc = Allocator(window_train_data)

                for j in range(len(window_test_data)):
                    weights[j, :] = alloc.allocate_portfolio(window_test_data.iloc[j, :])
                    if np.sum(weights < -1) or np.sum(weights > 1):
                        raise Exception("Weights Outside of Bounds")

                capital = [1]
                for j in range(len(window_test_data) - 1):
                    asset_prices = np.array(window_test_data.iloc[j, :])
                    shares = capital[-1] * weights[j] / asset_prices
                    balance = capital[-1] - np.dot(shares, asset_prices)
                    net_change = np.dot(shares, np.array(window_test_data.iloc[j+1, :]))
                    capital.append(balance + net_change)

                capital = np.array(capital)
                returns = (capital[1:] - capital[:-1]) / capital[:-1]

                if np.std(returns) != 0:
                    sharpe = np.mean(returns) / np.std(returns)
                else:
                    sharpe = 0

                window_sharpe = sharpe * np.sqrt(252)
                sharpe_ratios.append(window_sharpe)
                capitals.append(capital)
                weights_list.append(weights)
                temp.append(window_sharpe)
                if i % avg_log_size == 0 or i == data.shape[0]-window_size:
                    res = 0
                    if len(temp) > 0:
                        res = np.mean(temp)
                        while len(temp)>0:
                            temp.pop()

                    print("Window Sharpe Ratio:", res)

            ax.plot(sharpe_ratios, label=f"Train Test Ratio: {k}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(f"Sharpe Ratios for Window Size: {window_size}")
        ax.legend()

        # Save the plot
        output_dir = "final/graphs"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"sharpe_ratios_window_{window_size}.png"))
        plt.close(fig)

    return sharpe_ratios, capitals, weights_list

# Set the window size (k)
method = "PAMR FINAL"
sharpe_ratios, capitals, weights_list = grading(data, method)
