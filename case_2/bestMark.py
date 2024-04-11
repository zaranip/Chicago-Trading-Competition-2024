import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from cvxopt import matrix, solvers

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
   def __init__(self, train_data, global_sharpe=None, sharpe=None, **kwargs):
       self.global_sharpe = global_sharpe
       self.sharpe = sharpe
       self.opt_markowitz_kwargs = kwargs

       self.running_price_paths = train_data.copy()
       
       self.train_data = train_data.copy()

   def freq(self, ix: pd.Index) -> float:
       assert isinstance(ix, pd.Index), "freq method only accepts pd.Index object"

       assert len(ix) > 1, "Index must contain more than one item"

       if not ix.is_monotonic_increasing:
           ix = ix.sort_values()

       if isinstance(ix, pd.DatetimeIndex):
           days = (ix[-1] - ix[0]).days
           return len(ix) / float(days) * 365.0
       else:
           return 252.0

   def opt_markowitz(self, mu, sigma, long_only=True, reg=0.0, rf_rate=0.02, q=1.0, max_leverage=1.0):
       keep = ~(mu.isnull() | (np.diag(sigma) < 0.00000001))

       mu = mu[keep]
       sigma = sigma.loc[keep, keep]

       m = len(mu)

       sigma = sigma.fillna(0.0)

       sigma = np.matrix(sigma)
       mu = np.matrix(mu).T

       sigma += np.eye(m) * reg

       if not long_only:
           sigma_inv = np.linalg.inv(sigma)
           b = q / 2 * (1 + rf_rate) * sigma_inv @ (mu - rf_rate)
           b = np.ravel(b)
       else:
           def maximize(mu, sigma, r, q):
               n = len(mu)

               P = 2 * matrix((sigma - r * mu * mu.T + (n * r) ** 2) / (1 + r))
               q = matrix(-mu) * q
               G = matrix(-np.eye(n))
               h = matrix(np.zeros(n))

               if max_leverage is None or max_leverage == float("inf"):
                   sol = solvers.qp(P, q, G, h)
               else:
                   A = matrix(np.ones(n)).T
                   b = matrix(np.array([float(max_leverage)]))
                   sol = solvers.qp(P, q, G, h, A, b)

               return np.squeeze(sol["x"])

           b = maximize(mu, sigma, rf_rate, q)

       b = pd.Series(b, index=keep.index[keep])
       b = b.reindex(keep.index).fillna(0.0)

       return b

   def allocate_portfolio(self, asset_prices):
       self.running_price_paths = self.running_price_paths.append(asset_prices, ignore_index=True)

       X = self.running_price_paths

       freq = self.freq(X.index)

       R = X.pct_change().dropna()

       sigma = R.cov() * freq

       if self.sharpe:
           mu = pd.Series(np.sqrt(np.diag(sigma)), X.columns) * pd.Series(self.sharpe).reindex(X.columns)
       elif self.global_sharpe:
           mu = pd.Series(np.sqrt(np.diag(sigma)) * self.global_sharpe, X.columns)
       else:
           mu = R.mean() * freq

       self.b = self.opt_markowitz(mu, sigma, **self.opt_markowitz_kwargs)

       return self.b

def grading(train_data, test_data):
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

   risk_free_rate = 0.02
   excess_returns = returns - risk_free_rate
   if np.std(excess_returns) != 0:
       sharpe = np.mean(excess_returns) / np.std(excess_returns)
   else:
       sharpe = 0

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