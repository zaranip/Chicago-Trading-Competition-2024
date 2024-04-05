import math
from sympy import diff, solve, symbols
import sympy as sp
import numpy as np
from scipy import stats
from sklearn.linear_model import GammaRegressor
import statsmodels.api as sm

class Model():
    def gamma_distribution(a,b):
        def gamma(x):
            if isinstance(x, sp.Expr):
                return b**a / sp.gamma(a) * x**(a-1) * sp.exp(-b*x)
            else:
                return b**a / math.gamma(a) * x**(a-1) * math.exp(-b*x)
        return gamma
    def fit_gamma(kde):
        clf = GammaRegressor()
        X = kde.support
        y = kde.density
        clf.fit(X.reshape(-1, 1), y)
        best_alpha = clf.coef_[0]
        best_beta = clf.coef_[1]
        
        return Model.gamma_distribution(best_alpha, best_beta)

class HistPred():
    def __init__(self, symbol, sample):
        self.symbol = symbol
        self.f = self.find_f(sample)
        self.d_f = self.find_d_f()
        self.V = self.find_var(sample)
        pass
    
    def name(self):
        return self.symbol
    
    def grad(self, x, mu_k):
        if isinstance(x, sp.Expr):
            # x_val = float(x.evalf())
            x_val = x
            return self.d_f(x_val) + mu_k * sp.cos(sp.atan(self.d_f(x_val))) * sp.sqrt(1 + self.f.evaluate(x_val)**2)
        else:
            return self.d_f(x) + mu_k * math.cos(math.atan(self.d_f(x))) * math.sqrt(1 + self.f.evaluate(x)**2)    
        
    def solver(self, x, k):
        mu_k = self.find_mu(k)
        y = sp.Symbol('y')
        eq = sp.Eq(self.grad(y, mu_k), self.grad(x, mu_k))
        sol = sp.solveset(eq, y, domain=sp.Reals)
        if isinstance(sol, sp.sets.EmptySet):
            return None
        elif isinstance(sol, sp.sets.FiniteSet):
            return float(sol.args[0])
        else:
            return float(sp.min(sol))
    
    def find_mu(self, k):
        return 1/(1+math.log(1+self.V*k))
    
    def find_var(self, sample):
        return np.var(sample)
        
    def find_f(self, sample):
        kde = sm.nonparametric.KDEUnivariate(sample)
        kde.fit()
        # return Model.fit_gamma(kde)
        return kde
    
    def find_d_f(self, dx=1e-6):
        x = sp.Symbol('x')
        f_expr = self.f
        d_f_expr = (f_expr.subs(x, x + dx) - f_expr.subs(x, x - dx)) / (2 * dx)
        d_f_callable = sp.lambdify(x, d_f_expr, 'numpy')

        def d_f(x_val):
            return d_f_callable(x_val)
        return d_f

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

    def update(self, order_books):
        self.volume = sum(v for _, v in self.order_books[self.symbol].bids.items() if v != 0) 
        + sum(v for _, v in self.order_books[self.symbol].asks.items() if v != 0)
            
        self.bids = dict((k,v) for k, v in order_books.bids.items() if v != 0)
        self.asks = dict((k,v) for k, v in order_books.asks.items() if v != 0)
        self.book = list(self.bids.keys()).extend(list(self.asks.keys()))
        price = self.predict_naive()
        self.prices.append(price)
        self.soft_average = (1-self.alpha)*self.soft_average + self.alpha*price if len(self.prices) > 1 else price
    
    def predict_naive(self):
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
    