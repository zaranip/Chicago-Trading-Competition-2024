import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('Case 2 Data 2024.csv')
symbols = data.columns[1:]  # Exclude the first column (presumably the date or index)

# Calculate returns
returns = data.iloc[:, 1:].pct_change().dropna()  # Exclude the first column from returns calculation

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
train_returns = train_data.iloc[:, 1:].pct_change().dropna()
test_returns = test_data.iloc[:, 1:].pct_change().dropna()

def objective_sharpe(weights, returns): 
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -portfolio_return / portfolio_volatility

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Money Limits
bounds = tuple((0, 1) for x in range(len(symbols)))

# Optimizations
init_guess = np.array(len(symbols) * [1. / len(symbols)])
print(init_guess.shape)

opt_results_sharpe = minimize(objective_sharpe, init_guess, args=(train_returns,), method='SLSQP', bounds=bounds, constraints=cons)

optimal_weights_sharpe = opt_results_sharpe.x

num_assets = len(symbols)
num_portfolios = 50000

np.random.seed(101)

print("line 45\n")

cov_matrix = train_returns.cov() * 252
mean_returns = train_returns.mean() * 252

port_returns = []
port_volatility = []
sharpe_ratio = []
all_weights = []  

for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns_portfolio = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sr = returns_portfolio / volatility
    sharpe_ratio.append(sr)
    port_returns.append(returns_portfolio)
    port_volatility.append(volatility)
    all_weights.append(weights)  

plt.figure(figsize=(12, 8))
plt.scatter(port_volatility, port_returns, c=sharpe_ratio, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

opt_returns_sharpe = np.sum(test_returns.mean() * optimal_weights_sharpe) * 252
opt_volatility_sharpe = np.sqrt(np.dot(optimal_weights_sharpe.T, np.dot(test_returns.cov() * 252, optimal_weights_sharpe)))
opt_portfolio_sharpe = plt.scatter(opt_volatility_sharpe, opt_returns_sharpe, color='r', s=50, label='Sharpe')

plt.legend(loc='upper right')

plt.show()

print("line 72\n")

def max_drawdown(return_series):
    comp_ret = (1 + return_series).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak) - 1
    return dd.min()

def detailed_portfolio_statistics(weights, returns):
    portfolio_returns = np.sum(returns * weights, axis=1)
    
    # General descriptive statistics
    mean_return_annualized = np.mean(portfolio_returns) * 252
    std_dev_annualized = np.std(portfolio_returns) * np.sqrt(252)
    skewness = skew(portfolio_returns)
    kurt = kurtosis(portfolio_returns)
    max_dd = max_drawdown(portfolio_returns)
    count = len(portfolio_returns)
    
    print("\nMean Return Annualized: ")
    print(mean_return_annualized)
    print("\nStandard Deviation Annualized: ")
    print(std_dev_annualized)

    # Optimization Metrics
    risk_free_rate = 0.00
    sharpe_ratio = (mean_return_annualized - risk_free_rate) / std_dev_annualized
    print("\nSharpe Ratio: ")
    print(sharpe_ratio)
    conf_level = 0.05
    cvar = mean_return_annualized - std_dev_annualized * norm.ppf(conf_level)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std_dev = np.std(downside_returns) * np.sqrt(252)
    sortino_ratio = mean_return_annualized / downside_std_dev
    variance = std_dev_annualized ** 2 
    
    return mean_return_annualized, std_dev_annualized, skewness, kurt, max_dd, count, sharpe_ratio, cvar, sortino_ratio, variance

# Calculate Sharpe Ratio on test data
statistics_sharpe = detailed_portfolio_statistics(optimal_weights_sharpe, test_returns)

# Names of the Statistics
statistics_names = ['Annual Return', 'Annualized Volatility', 'Skewness', 'Kurtosis', 'Max Drawdown', 'Data Count', 'Sharpe Ratio', 'CVaR', 'Sortino Ratio', 'Variance']

portfolio_data = {
    'Sharpe': {
        'weights': optimal_weights_sharpe,
        'statistics': statistics_sharpe
    },   
}

print("line 120\n")

for method, data in portfolio_data.items():
    print("\n")
    print("========================================================================================================")
    print("\n")
    print(f"Money made in this portfolio {method}:")
    print("\n")
    for symbol, weight in zip(symbols, data['weights']):
        if weight < 1e-4:  
            print(f"{symbol}: Basically 0%")
        else:
            print(f"{symbol}: {weight*100:.2f}%")

    print("\n")
    print(f"Descriptive Statistics of {method}:")
    print("\n")
    for name, stat in zip(statistics_names, data['statistics']):
        print(f"{name}: {stat:.2f}")