import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

class Allocator:
    def __init__(self, train_data):
        self.train_data = train_data
        self.models = self.train_models()

    def train_models(self):
        models = {}
        for asset in self.train_data.columns:
            # Time Series Analysis (ARIMA)
            model_arima = ARIMA(self.train_data[asset], order=(1, 1, 1))
            model_arima_fit = model_arima.fit()
            models[f'{asset}_arima'] = model_arima_fit

            # Machine Learning (Linear Regression)
            X = np.arange(len(self.train_data)).reshape(-1, 1)
            y = self.train_data[asset].values.reshape(-1, 1)
            model_lr = LinearRegression()
            model_lr.fit(X, y)
            models[f'{asset}_lr'] = model_lr

        return models

    def predict_returns(self, asset, forecast_horizon):
        # Time Series Analysis (ARIMA)
        arima_forecast = self.models[f'{asset}_arima'].forecast(forecast_horizon)

        # Machine Learning (Linear Regression)
        X_future = np.arange(len(self.train_data), len(self.train_data) + forecast_horizon).reshape(-1, 1)
        lr_forecast = self.models[f'{asset}_lr'].predict(X_future)

        # Combine forecasts (equal weights)
        forecast = (arima_forecast + lr_forecast) / 2

        return forecast

    def optimize_portfolio(self, returns, min_weight=-1, max_weight=1):
        def sharpe_ratio(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T), weights)))
            sharpe = portfolio_return / portfolio_std_dev
            return -sharpe  # Minimize negative Sharpe ratio

        num_assets = len(returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights equals 1
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        initial_guess = np.ones(num_assets) / num_assets  # Equal weights as initial guess

        result = minimize(sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimized_weights = result.x

        return optimized_weights

    def momentum_strategy(self, returns, lookback_period):
        returns_lookback = returns.iloc[-lookback_period:]
        momentum_scores = returns_lookback.mean()
        weights = np.where(momentum_scores > 0, 1, -1)  # Long positive momentum, short negative momentum
        weights /= np.abs(weights).sum()  # Normalize weights
        return weights

    def allocate_portfolio(self, test_data_row):
        forecast_horizon = 1  # Adjust as needed
        predicted_returns = []
        for asset in test_data_row.index:
            predicted_return = self.predict_returns(asset, forecast_horizon)
            predicted_returns.append(predicted_return[0])

        # Optimize portfolio weights
        # optimized_weights = self.optimize_portfolio(np.array(predicted_returns))

        # Momentum strategy
        lookback_period = 60  # Adjust as needed
        momentum_weights = self.momentum_strategy(self.train_data, lookback_period)

        # Combine optimized weights and momentum weights
        # combined_weights = (optimized_weights + momentum_weights) / 2

        # Use momentum weights as the final weights
        final_weights = momentum_weights

        return final_weights
    
data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

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
        print("Return: ", capital[-1] - capital[0])
        print(np.mean(returns), np.std(returns))
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    return sharpe, capital, weights



sharpe, capital, weights = grading(TRAIN, TEST)
print(sharpe)