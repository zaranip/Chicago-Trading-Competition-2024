import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean

data = pd.read_csv('Case 2 Data 2024.csv')
symbols = data.columns[1:]  # Exclude the first column (presumably the date or index)
# Calcular los retornos
returns = data.iloc[:, 1:].pct_change().dropna()  # Exclude the first column from returns calculation

def objective_sharpe(weights): 
    print("objective_sharpe\n")
    return -np.dot(weights, returns.mean()) / np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Los límites para los pesos
bounds = tuple((0, 1) for x in range(len(symbols)))

# Optimización
init_guess = np.array(len(symbols) * [1. / len(symbols)])
print(init_guess.shape)

opt_results = minimize(objective_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

optimal_weights = opt_results.x
opt_results_sharpe = minimize(objective_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

optimal_weights_sharpe = opt_results_sharpe.x

# Graficar la frontera eficiente
port_returns = []
port_volatility = []
sharpe_ratio = []
all_weights = []  # almacena los pesos de todas las carteras simuladas

num_assets = len(symbols)
num_portfolios = 50000

np.random.seed(101)

print("line 45\n")

for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns_portfolio = np.dot(weights, returns.mean()) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sr = returns_portfolio / volatility
    sharpe_ratio.append(sr)
    port_returns.append(returns_portfolio)
    port_volatility.append(volatility)
    all_weights.append(weights)  # registra los pesos para esta cartera

plt.figure(figsize=(12, 8))
plt.scatter(port_volatility, port_returns, c=sharpe_ratio, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

opt_returns_sharpe = np.dot(optimal_weights_sharpe, returns.mean()) * 252
opt_volatility_sharpe = np.sqrt(np.dot(optimal_weights_sharpe.T, np.dot(returns.cov() * 252, optimal_weights_sharpe)))
opt_portfolio_sharpe = plt.scatter(opt_volatility_sharpe, opt_returns_sharpe, color='r', s=50, label='Sharpe')

plt.legend(loc='upper right')

plt.show()

print("line 72\n")

# Función para calcular el drawdown máximo
def max_drawdown(return_series):
    comp_ret = (1 + return_series).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak) - 1
    return dd.min()

# Función para calcular estadísticas descriptivas más detalladas
def detailed_portfolio_statistics(weights):
    portfolio_returns = returns.dot(weights)
    
    # Estadísticas descriptivas generales
    mean_return_annualized = gmean(portfolio_returns + 1)**252 - 1
    std_dev_annualized = portfolio_returns.std() * np.sqrt(252)
    skewness = skew(portfolio_returns)
    kurt = kurtosis(portfolio_returns)
    max_dd = max_drawdown(portfolio_returns)
    count = len(portfolio_returns)
    
    # Métricas de optimización
    risk_free_rate = 0.00
    sharpe_ratio = (mean_return_annualized - risk_free_rate) / std_dev_annualized
    print("\nSharpe Ratio: ")
    print(sharpe_ratio)
    conf_level = 0.05
    cvar = mean_return_annualized - std_dev_annualized * norm.ppf(conf_level)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std_dev = downside_returns.std() * np.sqrt(252)
    sortino_ratio = mean_return_annualized / downside_std_dev
    variance = std_dev_annualized ** 2 
    
    return mean_return_annualized, std_dev_annualized, skewness, kurt, max_dd, count, sharpe_ratio, cvar, sortino_ratio, variance

# Calcular estadísticas para cada portafolio
statistics_sharpe = detailed_portfolio_statistics(optimal_weights_sharpe)

# Nombres de las estadísticas
statistics_names = ['Retorno anualizado', 'Volatilidad anualizada', 'Skewness', 'Kurtosis', 'Max Drawdown', 'Conteo de datos', 'Sharpe Ratio', 'CVaR', 'Ratio Sortino', 'Varianza']

# Diccionario que asocia los nombres de los métodos de optimización con los pesos óptimos y las estadísticas
portfolio_data = {
    'Sharpe': {
        'weights': optimal_weights_sharpe,
        'statistics': detailed_portfolio_statistics(optimal_weights_sharpe)
    },   
}

print("line 120\n")

# Imprimir los pesos y las estadísticas para cada método de optimización
for method, data in portfolio_data.items():
    print("\n")
    print("========================================================================================================")
    print("\n")
    print(f"Pesos del portafolio óptimo para {method}:")
    print("\n")
    for symbol, weight in zip(symbols, data['weights']):
        if weight < 1e-4:  # considera cualquier peso menor que 0.01% como cero
            print(f"{symbol}: prácticamente 0%")
        else:
            print(f"{symbol}: {weight*100:.2f}%")

    print("\n")
    print(f"Estadísticas descriptivas del portafolio óptimo para {method}:")
    print("\n")
    for name, stat in zip(statistics_names, data['statistics']):
        print(f"{name}: {stat*100 if name != 'Conteo de datos' else stat:.2f}")

print("\n")
print("========================================================================================================")