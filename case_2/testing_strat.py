import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean
import os, sys

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

log_to_file("case_2/trial_5/mc_chungus_results.txt")

data = pd.read_csv('Case 2 Data 2024.csv')
symbols = data.columns[1:]  # Exclude the first column (presumably the date or index)

# Calculate returns
returns = data.iloc[:, 1:].pct_change().dropna()  # Exclude the first column from returns calculation
num_data_points = len(returns)

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

window_sizes = [63, 126, 252, 2520]
#['Powell', 'L-BFGS-B', 'TNC']
for window_size in window_sizes: 
    print(f"Window Size: {window_size}")
    for method in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
        print(f"Method: {method}\n\n")
        sharpe_ratios = []
        
        for i in range(0, num_data_points, window_size):
            start_index = i
            end_index = min(i + window_size, num_data_points)
            
            data_window = returns.iloc[start_index:end_index]
            
            opt_results_sharpe = minimize(objective_sharpe, init_guess, args=data_window, method=method, bounds=bounds, constraints=cons)
            optimal_weights_sharpe = opt_results_sharpe.x
            
            # Evaluate the portfolio on the data window
            window_statistics = detailed_portfolio_statistics(optimal_weights_sharpe, data_window)
            window_sharpe_ratio = window_statistics[6]  # Index 6 corresponds to Sharpe Ratio
            
            sharpe_ratios.append(window_sharpe_ratio)
        
        # Calculate mean, variance, and standard deviation of Sharpe ratios
        mean_sharpe = np.mean(sharpe_ratios)
        var_sharpe = np.var(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        print(f"Sharpe Ratios: {sharpe_ratios}")
        print(f"Mean Sharpe Ratio: {mean_sharpe:.4f}")
        print(f"Variance of Sharpe Ratios: {var_sharpe:.4f}")
        print(f"Standard Deviation of Sharpe Ratios: {std_sharpe:.4f}")
        print("\n")
        
        # Plot Sharpe ratios
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(sharpe_ratios)), sharpe_ratios, marker='o')
        plt.text(100, 1, f"Mean sharpe: {mean_sharpe}", fontsize=12)
        plt.text(0.1, 15, f"Mean Var: {var_sharpe}", fontsize=12)
        plt.text(0.1, 30, f"Mean Std: {std_sharpe}", fontsize=12)
        plt.xlabel('Window')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Sharpe Ratios - {method}')
        plt.grid(True)
        plt.savefig(f"case_2/trial_5/{window_size}_{method}_sharpe_ratios_graph.png")
        plt.close()