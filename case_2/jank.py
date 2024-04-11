import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0)

TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)


def calc_returns(prices):
    returns = prices.pct_change().iloc[1:]
    return returns

def calc_momentum(returns, window):
    momentum = returns.rolling(window=window, min_periods=1).mean()
    return momentum

def calc_granger_tilt(returns):
    granger_matrix = pd.DataFrame(np.zeros((len(returns.columns), len(returns.columns))), index=returns.columns, columns=returns.columns)
    granger_matrix.loc['A', 'E'] = 1
    granger_matrix.loc['A', 'F'] = 1
    granger_matrix.loc['F', 'A'] = 1
    granger_matrix.loc['D', 'C'] = 1
    granger_matrix.loc['F', 'C'] = 1
    
    granger_tilt = returns.dot(granger_matrix)
    return granger_tilt

def detect_regime(returns, window):
    rolling_corr = returns.rolling(window=window, min_periods=1).corr()
    rolling_vol = returns.rolling(window=window, min_periods=1).std()
    
    rolling_corr_reset = rolling_corr.reset_index(level=1, drop=True)
    rolling_vol_reset = rolling_vol.reset_index(drop=True)
    
    regime = (rolling_corr_reset.fillna(method='ffill') > 0.5) & (rolling_vol_reset.fillna(method='ffill') > rolling_vol_reset.fillna(method='ffill').quantile(0.75))
    regime = regime.astype(int)
    
    return regime

def volatility_weighting(returns, window):
    # Calculate the initial standard deviation using the first 'window' periods
    initial_std = returns.iloc[:window].std()

    # Create a new DataFrame to store the rolling standard deviation
    returns_std = pd.DataFrame(index=returns.index, columns=returns.columns, data=np.nan)

    # Fill the initial 'window' periods with the initial standard deviation
    returns_std.iloc[:window] = initial_std.values

    # Calculate the rolling standard deviation for the remaining periods
    for i in range(window, len(returns)):
        returns_std.iloc[i] = returns.iloc[i - window:i].std()

    # Replace any remaining NaN values with the initial standard deviation
    returns_std = returns_std.fillna(initial_std)

    inv_vol = 1 / returns_std
    vol_weights = inv_vol.divide(inv_vol.sum(axis=1), axis='rows')

    return vol_weights

def normalize_weights(weights):
    abs_sum = weights.abs().sum()
    if abs_sum == 0 or np.isnan(abs_sum):
        # If the sum of absolute weights is zero or NaN, return equal weights
        return pd.Series(data=np.full(len(weights), 1/len(weights)), index=weights.index)
    else:
        return weights / abs_sum


class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        
    def allocate_portfolio(self, asset_prices):
        self.running_price_paths.loc[len(self.running_price_paths)] = asset_prices
        
        returns = calc_returns(TRAIN)
        #print(returns)
        
        momentum_short = calc_momentum(returns, window=10)
        momentum_long = calc_momentum(returns, window=50)
        
        granger_tilt = calc_granger_tilt(returns)
        regime = detect_regime(returns, window=100)

        print("Momentum short: ", momentum_short)
        print("Momentum long: ", momentum_long)
        print("Granger tilt: ", granger_tilt)
        print("Regime: ", regime)
        
        weights = {}
        for asset in returns.columns:
            weight = momentum_long[asset] * granger_tilt[asset] * regime[asset]
            weights[asset] = weight
        
        vol_weights = volatility_weighting(returns, window=50) 

        print("Volatility", vol_weights)
        
        raw_weights = pd.Series(weights)
        raw_weights = pd.to_numeric(raw_weights, errors='coerce')
        
        tilted_weights = raw_weights * vol_weights.iloc[-1]
        normalized_weights = normalize_weights(tilted_weights)
        print(normalized_weights)
        # Reindex normalized_weights to match the order of asset columns in test_data
        normalized_weights = normalized_weights.reindex(asset_prices.index)
        
        return normalized_weights.to_numpy(dtype=float)


def grading(train_data, test_data): 
    best = 0
    best_weight = np.zeros(6)

    counter = 0
    greater = 0
    THRESH = 0.138
    interval = 100
    while True:
        weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
        alloc = Allocator(train_data)
        for i in range(0,len(test_data)):
            weights[i,:] = np.random.rand(6) #alloc.allocate_portfolio(test_data.iloc[i,:])
            #print(weights[i,:])
            if np.sum(weights < -1) or np.sum(weights > 1):
                raise Exception("Weights Outside of Bounds")
        
        capital = [1]
        for i in range(len(test_data) - 1):
            shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
            balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
            net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
            capital.append(balance + net_change)
        capital = np.array(capital)
        returns = (capital[1:] - capital[:-1]) / capital[:-1]
        
        if np.std(returns) != 0:
            sharpe = np.mean(returns) / np.std(returns)
        else:
            sharpe = 0
        
        if sharpe > THRESH:
            greater += 1
        counter += 1

        if counter % interval == 0:
            print("Counter: ", counter)
            print("Greater: ", greater)
            print(f"Prob of random sharpe > {THRESH}: ", greater / counter)
            print("\n")
    
    
    print("Best sharpe: ", best)
    print("Best weight: ", best_weight)
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