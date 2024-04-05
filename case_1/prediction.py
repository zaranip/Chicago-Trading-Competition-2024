import math
from sympy import diff, solve, symbols
import sympy as sp
import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
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
        params = {'alpha': np.linspace(0.1, 10, 100),
                  'beta': np.linspace(0.1, 10, 100)}
        
        grid_search = GridSearchCV(KernelDensity(), params, cv=5)
        grid_search.fit(kde.support, kde.density)
        
        best_alpha = grid_search.best_params_['alpha']
        best_beta = grid_search.best_params_['beta']
        
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
        return Model.fit_gamma(kde)
    
    def find_d_f(self, dx=1e-6):
        x = sp.Symbol('x')
        def f_callable(x_val):
            return self.f.evaluate(x_val)

        f_expr = f_callable(x)
        d_f_expr = (f_expr.subs(x, x + dx) - f_expr.subs(x, x - dx)) / (2 * dx)
        d_f_callable = sp.lambdify(x, d_f_expr, 'numpy')

        def d_f(x_val):
            return d_f_callable(x_val)
        return d_f

class RoundPred():
    def __init__(self, symbol):
        self.alpha = 0.2
        self.prices = []
        self.soft_average = 0
        pass
    
    def name(self):
        return self.symbol
    
    def get_current(self):
        return self.prices[-1]

    def update(self, book):
        price = self.predict_naive(book)
        self.prices.append(price)
        self.soft_average = (1-self.alpha)*self.soft_average + self.alpha*price if len(self.prices) > 1 else price
    
    def predict_naive(self, book):
        return np.mean(book) if len(book) > 0 else 0
    
    def predict_window(self, book):
        # TODO: implement the sliding window approach
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
        x = self.round.get_current()
        return (1-self.weight) * self.round.average() + self.weight*self.hist.solver(x, k)
    
    def __repr__(self):
        # TODO: fill in the representation for informative print outs
        return self.symbol
    
    
if __name__ == "__main__":
    pass
    