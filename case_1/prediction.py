import math
import sympy as sp
from sympy import symbols, exp, sqrt, pi, lambdify, diff, solve, atan, cos, gamma, loggamma, log
import collections
import numpy as np
from sklearn.linear_model import GammaRegressor
import statsmodels.api as sm

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import sympy as sp

class GammaDistribution:
    def __init__(self, data):
        self.data = data
        self.kde = gaussian_kde(data)
        self.a, self.loc, self.scale = self._fit_params()
    
    def _fit_params(self):
        def mse(params):
            a, loc, scale = params
            x = np.linspace(min(self.data), max(self.data), 100)
            pdf_values = stats.gamma.pdf(x, a, loc=loc, scale=scale)
            kde_values = self.kde(x)
            return np.mean((pdf_values - kde_values)**2)
        
        initial_params = stats.gamma.fit(self.data)
        result = minimize(mse, initial_params, method='Nelder-Mead')
        return result.x
    
    def pdf(self, x):
        if isinstance(x, sp.Symbol):
            print(x)
            return (x**(self.a - 1) * sp.exp(-x / self.scale)) / (self.scale**self.a * sp.gamma(self.a))
        else:
            return stats.gamma.pdf(x, self.a, loc=self.loc, scale=self.scale)
    
    def deriv_pdf(self, x):
        if isinstance(x, sp.Symbol):
            a, scale = self.a, self.scale
            return sp.diff((x**(a - 1) * sp.exp(-x / scale)) / (scale**a * sp.gamma(a)), x)
        else:
            a, scale = self.a, self.scale
            print(a, scale)
            return (x**(a - 2) * np.exp(-x / scale) * (a - 1 - x / scale)) / (scale**a * np.math.gamma(a))

class LognormDistribution:
    def __init__(self, data):
        self.data = data
        self.kde = gaussian_kde(data)
        self.s, self.loc, self.scale = self._fit_params()

    def _fit_params(self):
        def mse(params):
            s, loc, scale = params
            x = np.linspace(min(self.data), max(self.data), 100)
            pdf_values = stats.lognorm.pdf(x, s, loc=loc, scale=scale)
            kde_values = self.kde(x)
            return np.mean((np.log(pdf_values) - np.log(kde_values))**2)

        initial_params = stats.lognorm.fit(self.data)
        result = minimize(mse, initial_params, method='Nelder-Mead')
        print("optimized")
        return result.x

    def log_pdf(self, x):
        if isinstance(x, sp.Symbol):
            return -(sp.log(x) - self.loc)**2 / (2 * self.s**2) - sp.log(x) - sp.log(self.s) - 0.5 * sp.log(2 * sp.pi)
        else:
            x = np.where(x == 0, np.finfo(float).eps, x)  # Replace 0 with a small value
            return stats.lognorm.logpdf(x, self.s, loc=self.loc, scale=self.scale)

    def log_deriv_pdf(self, x):
        if isinstance(x, sp.Symbol):
            s, loc = self.s, self.loc
            return sp.diff(-(sp.log(x) - loc)**2 / (2 * s**2) - sp.log(x) - sp.log(s) - 0.5 * sp.log(2 * sp.pi), x)
        else:
            s, loc = self.s, self.loc
            x = np.where(x == 0, np.finfo(float).eps, x)  # Replace 0 with a small value
            return -(np.log(x) - loc) / (s**2 * x) - 1 / x

    def pdf(self, x):
        return sp.exp(self.log_pdf(x))

    def deriv_pdf(self, x):
        return sp.exp(self.log_deriv_pdf(x))
    

class GammaKDE:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.k = symbols('k')
        self.theta = symbols('theta')
        self.pdf_expr = None
        self.pdf_func = None
        self.deriv_pdf_expr = None
        self.deriv_pdf_func = None

    def fit(self):
        # Estimate shape and scale parameters
        data_mean = np.mean(self.data)
        data_var = np.var(self.data)
        self.k_value = data_mean**2 / data_var
        self.theta_value = data_var / data_mean

        # Calculate the PDF expression
        self.pdf_expr = (self.k**self.k * self.theta**(-self.k) * gamma(self.k) * (self.k - 1) * exp(-self.k * self.theta))
        self.pdf_func = lambdify([self.k, self.theta], self.pdf_expr, 'numpy')

    def pdf(self, x, k=None, theta=None):
        if k is None:
            k = self.k_value
        if theta is None:
            theta = self.theta_value

        log_pdf_expr = k * log(k) - k * log(theta) + (k - 1) * log(x) - x / theta - loggamma(k)

        if isinstance(x, (int, float)):
            return float(exp(log_pdf_expr.subs([(self.k, k), (self.theta, theta)])))
        else:
            return exp(log_pdf_expr.subs([(self.k, k), (self.theta, theta)]))

    def deriv_pdf(self, x, k=None, theta=None):
        if self.deriv_pdf_expr is None:
            self.deriv_pdf_expr = diff(self.pdf_expr, self.k)
            self.deriv_pdf_func = lambdify([self.k, self.theta], self.deriv_pdf_expr, 'numpy')

        if k is None:
            k = self.k_value
        if theta is None:
            theta = self.theta_value

        if isinstance(x, (int, float)):
            return self.deriv_pdf_func(k, theta).subs([(self.k, k), (self.theta, theta)])
        else:
            return self.deriv_pdf_expr.subs([(self.k, k), (self.theta, theta)])
        
class GaussianKDE:
    def __init__(self, data, bandwidth=None):
        self.data = data
        self.n = len(data)
        if bandwidth is None:
            self.bandwidth = self._scotts_rule()
        else:
            self.bandwidth = bandwidth
        self.x = symbols('x')
        self.pdf_expr = None
        self.pdf_func = None
        self.deriv_pdf_expr = None
        self.deriv_pdf_func = None

    def _scotts_rule(self):
        return 1.06 * np.std(self.data) * self.n ** (-1/5)

    def _gaussian_kernel(self, x):
        return exp(-(x - self.x)**2 / (2 * self.bandwidth**2)) / (sqrt(2 * pi) * self.bandwidth)

    def fit(self):
        # Calculate the PDF expression
        self.pdf_expr = sum(self._gaussian_kernel(xi) for xi in self.data) / self.n

        # Create a lambdified function for the PDF
        self.pdf_func = lambdify(self.x, self.pdf_expr)

    def pdf(self, x):
        if isinstance(x, (int, float)):
            return self.pdf_func(x)
        else:
            return self.pdf_expr.subs(self.x, x)

    def deriv_pdf(self, x):
        if self.deriv_pdf_expr is None:
            self.deriv_pdf_expr = diff(self.pdf_expr, self.x)
            self.deriv_pdf_func = lambdify(self.x, self.deriv_pdf_expr)

        if isinstance(x, (int, float)):
            return self.deriv_pdf_func(x)
        else:
            return self.deriv_pdf_expr.subs(self.x, x)   


class HistPred:
    def __init__(self, symbol, sample):
        self.symbol = symbol
        self.sample = sample
        self.kde = LognormDistribution(sample)
        self.log_f = self.kde.log_pdf
        self.log_d_f = self.kde.log_deriv_pdf
        self.V = self.find_var()

    def name(self):
        return self.symbol

    def grad(self, x, mu_k):
        log_exp = self.log_d_f(x) + log(mu_k * cos(atan(exp(self.log_d_f(x)))) * sqrt(1 + exp(2 * self.log_f(x))))
        print(log_exp)
        return log_exp

    def solver(self, x, k):
        x=5000
        y = symbols('y', real=True)
        mu_k = self.find_mu(k)
        print(f"solving for {x} with lookahead {k} and mu {mu_k}")
        #self.grad(y, mu_k) - self.grad(x, mu_k)
        result = solve(self.grad(y, mu_k) - self.grad(x, mu_k), simplify=False)
        print("result is", result)
        return result[0]

    def find_mu(self, k):
        return 1 / (1 + log(1 + self.V * k))

    def find_var(self):
        return np.var(self.sample)


class RoundPred():
    def __init__(self, symbol):
        self.symbol = symbol
        self.alpha = 0.2
        self.prices = []
        self.soft_average = 0
        self.volume = 0
        self.bids = {}
        self.asks = {}
        self.book = []
        pass
    
    def name(self):
        return self.symbol
    
    def get_current_price(self):
        return self.prices[-1]
    
    def get_bid_prices(self):
        return list(self.bids.keys())

    def get_asks_prices(self):
        return list(self.asks.keys())

    def update(self, order_book):

        self.volume = sum(v for _, v in order_book.bids.items() if v != 0) 
        + sum(v for _, v in order_book.asks.items() if v != 0)
            
        self.bids = dict((k,v) for k, v in order_book.bids.items() if v != 0)
        self.asks = dict((k,v) for k, v in order_book.asks.items() if v != 0)
        self.book = list(self.bids.keys()) + (list(self.asks.keys()))
        # print(self.book)
        price = self.predict_naive()
        self.prices.append(price)
        self.soft_average = (1-self.alpha)*self.soft_average + self.alpha*price if len(self.prices) > 1 else price
    
    def predict_naive(self):
        # print("Printing book", self.book)
        return np.mean(self.book) if len(self.book) > 0 else 0
    
    def predict_window(self, book):
        pass
    
    def average(self):
        return self.soft_average

class Prediction():
    def __init__(self, symbol, sample):
        self.symbol = symbol
        self.hist = HistPred(symbol, sample)
        self.round = RoundPred(symbol)
        self.weight = 0.3
        self.fade = 1
        pass

    def name(self):
        return self.symbol
    
    def update(self, book):
        self.round.update(book)

    def predict(self, k):
        x = self.round.get_current_price()
        return (1-self.weight) * self.round.average() + self.weight*self.hist.solver(x, k)
    
    def bid(self, pred):
        # implemented penny in
        bids = [bid for bid in self.round.get_bid_prices() if bid < pred]
        return min(bids, key=lambda x: abs(x-pred)) + 1
    
    def ask(self, pred):
        # implemented penny out
        asks = [ask for ask in self.round.get_asks_prices() if ask > pred]
        return min(asks, key=lambda x: abs(x-pred)) - 1
    
    def __str__(self):
        # TODO: fill in the representation for informative print outs
        return f"A predictor of {self.symbol}"
if __name__ == "__main__":
    pass
    