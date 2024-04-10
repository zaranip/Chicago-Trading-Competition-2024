import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.symbols = self.data.columns[1:]
        self.returns = self.data.iloc[:, 1:].pct_change().dropna()

    def split_data(self, test_size=0.3, random_state=None):
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        train_returns = train_data.iloc[:, 1:].pct_change().dropna()
        test_returns = test_data.iloc[:, 1:].pct_change().dropna()
        return train_returns, test_returns

class OptimizationObjective:
    def __init__(self, returns):
        self.returns = returns

    def sharpe_ratio(self, weights):
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        return -portfolio_return / portfolio_volatility

class PortfolioOptimizer:
    def __init__(self, symbols, returns):
        self.symbols = symbols
        self.returns = returns
        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        self.bounds = tuple((0, 1) for _ in range(len(symbols)))

    def optimize(self, objective_func):
        init_guess = np.array(len(self.symbols) * [1. / len(self.symbols)])
        opt_results = minimize(objective_func, init_guess, method='SLSQP', bounds=self.bounds, constraints=self.constraints)
        return opt_results.x

class PortfolioSimulator:
    def __init__(self, returns, num_portfolios):
        self.returns = returns
        self.num_portfolios = num_portfolios
        self.num_assets = len(returns.columns)
        self.cov_matrix = returns.cov() * 252
        self.mean_returns = returns.mean() * 252

    def simulate_portfolios(self):
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        all_weights = []

        np.random.seed(101)

        for _ in range(self.num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            returns_portfolio = np.sum(self.mean_returns * weights)
            volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sr = returns_portfolio / volatility
            sharpe_ratio.append(sr)
            port_returns.append(returns_portfolio)
            port_volatility.append(volatility)
            all_weights.append(weights)

        return port_returns, port_volatility, sharpe_ratio, all_weights

class PortfolioVisualizer:
    def __init__(self, port_returns, port_volatility, sharpe_ratio, opt_returns, opt_volatility):
        self.port_returns = port_returns
        self.port_volatility = port_volatility
        self.sharpe_ratio = sharpe_ratio
        self.opt_returns = opt_returns
        self.opt_volatility = opt_volatility

    def plot_efficient_frontier(self):
        plt.figure(figsize=(12, 8))
        plt.scatter(self.port_volatility, self.port_returns, c=self.sharpe_ratio, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        opt_portfolio = plt.scatter(self.opt_volatility, self.opt_returns, color='r', s=50, label='Sharpe')
        plt.legend(loc='upper right')

        plt.show()

class PortfolioStatistics:
    def __init__(self, returns):
        self.returns = returns

    def max_drawdown(self, return_series):
        comp_ret = (1 + return_series).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret/peak) - 1
        return dd.min()

    def detailed_statistics(self, weights):
        portfolio_returns = np.sum(self.returns * weights, axis=1)

        mean_return_annualized = gmean(portfolio_returns + 1)**252 - 1
        std_dev_annualized = portfolio_returns.std() * np.sqrt(252)
        skewness = skew(portfolio_returns)
        kurt = kurtosis(portfolio_returns)
        max_dd = self.max_drawdown(portfolio_returns)
        count = len(portfolio_returns)

        risk_free_rate = 0.00
        sharpe_ratio = (mean_return_annualized - risk_free_rate) / std_dev_annualized
        conf_level = 0.05
        cvar = mean_return_annualized - std_dev_annualized * norm.ppf(conf_level)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std_dev = downside_returns.std() * np.sqrt(252)
        sortino_ratio = mean_return_annualized / downside_std_dev
        variance = std_dev_annualized ** 2

        return mean_return_annualized, std_dev_annualized, skewness, kurt, max_dd, count, sharpe_ratio, cvar, sortino_ratio, variance
    
def main():
    data_processor = DataProcessor('Case 2 Data 2024.csv')
    symbols = data_processor.symbols
    train_returns, test_returns = data_processor.split_data(test_size=0, random_state=42)

    optimization_objective = OptimizationObjective(train_returns)
    portfolio_optimizer = PortfolioOptimizer(symbols, train_returns)
    optimal_weights_sharpe = portfolio_optimizer.optimize(optimization_objective.sharpe_ratio)

    portfolio_simulator = PortfolioSimulator(train_returns, num_portfolios=50000)
    port_returns, port_volatility, sharpe_ratio, all_weights = portfolio_simulator.simulate_portfolios()

    opt_returns_sharpe = np.sum(train_returns.mean() * optimal_weights_sharpe) * 252
    opt_volatility_sharpe = np.sqrt(np.dot(optimal_weights_sharpe.T, np.dot(train_returns.cov() * 252, optimal_weights_sharpe)))

    portfolio_visualizer = PortfolioVisualizer(port_returns, port_volatility, sharpe_ratio, opt_returns_sharpe, opt_volatility_sharpe)
    portfolio_visualizer.plot_efficient_frontier()

    portfolio_statistics = PortfolioStatistics(test_returns)
    statistics_sharpe = portfolio_statistics.detailed_statistics(optimal_weights_sharpe)

    statistics_names = ['Annual Return', 'Annualized Volaltility', 'Skewness', 'Kurtosis', 'Max Drawdown', 'Data Count', 'Sharpe Ratio', 'CVaR', 'Sortino Ratio', 'Variance']

    portfolio_data = {
        'Sharpe': {
            'weights': optimal_weights_sharpe,
            'statistics': portfolio_statistics.detailed_statistics(optimal_weights_sharpe)
        },
    }

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
            print(f"{name}: {stat*100 if name != 'Data Count' else stat:.2f}")

if __name__ == "__main__":
    main()