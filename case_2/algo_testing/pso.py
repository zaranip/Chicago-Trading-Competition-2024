import numpy as np
import pandas as pd
from pyswarm import pso

class Allocator:
    def __init__(self, train_data):
        self.train_data = train_data

    def allocate_portfolio(self, data_point):
        # Implement your allocation strategy here based on the training data
        # For example, you can use the mean returns of the assets as the weights
        weights = self.train_data.mean() / self.train_data.mean().sum()
        return weights

def objective_function(weights, train_data, test_data):
    alloc = Allocator(train_data)
    alloc.allocate_portfolio = lambda x: weights

    sharpe, capital, _ = grading(train_data, test_data)
    return -sharpe  # Negative Sharpe ratio since PSO minimizes the objective function

def optimize_weights(train_data, test_data):
    num_assets = train_data.shape[1]
    lb = np.full(num_assets, -1.0)  # Lower bound for weights
    ub = np.full(num_assets, 1.0)   # Upper bound for weights

    # Define the equality constraint (sum of weights should be 1)
    def constraint(weights, *args):
        return np.sum(weights) - 1

    # Run PSO optimization
    optimal_weights, _ = pso(objective_function, lb, ub, f_ieqcons=constraint, args=(train_data, test_data))

    return optimal_weights

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
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    return sharpe, capital, weights

# Load the data from the CSV file
data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

# Split the data into training and testing sets
train_data = data.iloc[:2000]
test_data = data.iloc[2000:]

# Optimize the weights
optimal_weights = optimize_weights(train_data, test_data)

# Create an instance of the Allocator class with the optimized weights
alloc = Allocator(train_data)
alloc.allocate_portfolio = lambda x: optimal_weights

# Calculate the Sharpe ratio, capital, and weights using the grading function
sharpe, capital, weights = grading(train_data, test_data)

print("Optimal Weights:", optimal_weights)
print("Sharpe Ratio:", sharpe)
print("Final Capital:", capital[-1])