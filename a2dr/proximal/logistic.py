import numpy as np
from scipy import sparse
from scipy.special import expit
from scipy.optimize import minimize

def prox_logistic(v, t, x0 = None, y = None):
    """Returns the proximal operator for f(x) = \sum_i log(1 + exp(-y_i*x_i)), where y is a given vector quantity,
       solved using the Newton-CG method from scipy.optimize.minimize. The function defaults to y_i = -1 for all i,
       so that f(x) = \sum_i log(1 + e^x_i).
    """
    if x0 is None:
        x0 = v   # np.random.randn(*v.shape)
    if y is None:
        y = -np.ones(v.shape)

    # g(x) = \sum_i log(1 + exp(-y_i*x_i)) + 1/(2*t)*||x - v||_2^2
    def fun(x, y, v, t):
        # expit(x) = 1/(1 + exp(-x))
        return -np.sum(np.log(expit(np.multiply(y,x)))) + 1.0/(2*t)*np.sum((x - v)**2)

    # dg(x)/dx_i = -y_i/(1 + exp(y_i*x_i)) + (1/t)*(x_i - v_i)
    def jac(x, y, v, t):
        return -np.multiply(y, expit(-np.multiply(y,x))) + (1.0/t)*(x - v)

    # d^2g(x)/dx_i^2 = y_i^2*exp(y_i*x_i)/(1 + exp(y_i*x_i))^2 + 1/t
    def hess(x, y, v, t):
        return np.diag(np.multiply(np.multiply(y**2, np.exp(np.multiply(y,x))), expit(-np.multiply(y,x))**2) + 1.0/t)

    res = minimize(fun, x0, args=(y, v, t), method='Newton-CG', jac=jac, hess=hess)
    # res = minimize(fun, x0, args=(y, v, t), method='Newton-CG', jac=jac)
    if not res.success:
        warnings.warn(res.message)
    return res.x[0] if res.x.size == 1 else res.x