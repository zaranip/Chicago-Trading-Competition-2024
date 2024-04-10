import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean

data = pd.read_csv('Case 2 Data 2024.csv')
symbols = data.columns[1:]  # Exclude the first column (presumably the date or index)
num_data_points = len(data)

# Calculate returns
returns = data.iloc[:, 1:].pct_change().dropna()  # Exclude the first column from returns calculation

def objective_sharpe(weights, returns):
    return -np.dot(weights, returns.mean()) / np.sqrt(np.dot(weights.T, np.dot(returns.cov() * returns.shape[0], weights)))

def max_drawdown(return_series):
    comp_ret = (1 + return_series).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak) - 1
    return dd.min()

def detailed_portfolio_statistics(weights, data):
    portfolio_returns = data.dot(weights)
    
    # General descriptive statistics
    mean_return_annualized = gmean(portfolio_returns + 1)**data.shape[0] - 1
    std_dev_annualized = portfolio_returns.std() * np.sqrt(data.shape[0])
    skewness = skew(portfolio_returns)
    kurt = kurtosis(portfolio_returns)
    max_dd = max_drawdown(portfolio_returns)
    count = len(portfolio_returns)
    
    # Optimization Metrics
    risk_free_rate = 0.00
    sharpe_ratio = (mean_return_annualized - risk_free_rate) / std_dev_annualized
    conf_level = 0.05
    cvar = mean_return_annualized - std_dev_annualized * norm.ppf(conf_level)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std_dev = downside_returns.std() * np.sqrt(data.shape[0])
    sortino_ratio = mean_return_annualized / downside_std_dev
    variance = std_dev_annualized ** 2
    
    return mean_return_annualized, std_dev_annualized, skewness, kurt, max_dd, count, sharpe_ratio, cvar, sortino_ratio, variance

# Names of the Statistics
statistics_names = ['Annual Return', 'Annualized Volatility', 'Skewness', 'Kurtosis', 'Max Drawdown', 'Data Count', 'Sharpe Ratio', 'CVaR', 'Sortino Ratio', 'Variance']

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Money Limits
bounds = tuple((0, 1) for x in range(len(symbols)))

# Optimizations
init_guess = np.array(len(symbols) * [1. / len(symbols)])

# Store k values and test Sharpe ratios
k_values = []
test_sharpe_ratios = []
interval = 50
for k in range(1, num_data_points, 50):
    print(f"Train-Test Split with k = {k}")
    
    # Split the data into train and test sets
    train_data = returns.iloc[:k]
    test_data = returns.iloc[k:]
    
    opt_results_sharpe = minimize(objective_sharpe, init_guess, args=train_data, method='CG', bounds=bounds, constraints=cons)
    optimal_weights_sharpe = opt_results_sharpe.x
    
    # Evaluate the portfolio on the test data
    test_statistics = detailed_portfolio_statistics(optimal_weights_sharpe, test_data)
    test_sharpe_ratio = test_statistics[6]  # Index 6 corresponds to Sharpe Ratio
    
    k_values.append(k)
    test_sharpe_ratios.append(test_sharpe_ratio)
    
    print("Optimal Weights (Sharpe):")
    for symbol, weight in zip(symbols, optimal_weights_sharpe):
        if weight < 1e-4:
            print(f"{symbol}: Basically 0%")
        else:
            print(f"{symbol}: {weight*100:.2f}%")
    
    print("\nDescriptive Statistics (Sharpe):")
    for name, stat in zip(statistics_names, test_statistics):
        print(f"{name}: {stat:.4f}")
    
    print("-----------------------------")

# Plot k values vs. test Sharpe ratios
plt.figure(figsize=(10, 6))
plt.plot(k_values, test_sharpe_ratios, marker='o')
plt.xlabel('k')
plt.ylabel('Test Sharpe Ratio')
plt.title('k vs. Test Sharpe Ratio')
plt.grid(True)
plt.show()

# Find the best value of k based on the highest test Sharpe ratio
best_k = k_values[np.argmax(test_sharpe_ratios)]
print(f"Best value of k: {best_k}")