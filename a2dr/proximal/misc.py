import numpy as np
import warnings
from scipy import sparse
from scipy.special import expit
from scipy.optimize import minimize
from a2dr.proximal.projection import proj_simplex
from a2dr.proximal.composition import prox_scale

def prox_logistic(v, t = 1, x0 = None, y = None, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where 
    :math:`f(x) = \\sum_i \\log(1 + \\exp(-y_i*x_i))`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(v):
        v = np.array([v])
    if x0 is None:
        # x0 = np.random.randn(*v.shape)
        x0 = v
    if y is None:
        y = -np.ones(v.shape)
    return prox_scale(prox_logistic_base, x0, y, *args, **kwargs)(v, t)

def prox_max(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\max_i x_i`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(v):
        v = np.array([v])
    return prox_scale(prox_max_base, *args, **kwargs)(v, t)

def prox_logistic_base(v, t, x0, y):
    """Proximal operator of :math:`f(x) = \\sum_i \\log(1 + \\exp(-y_i*x_i))`, where y is a given vector quantity,
    solved using the Newton-CG method from scipy.optimize.minimize. The function defaults to y_i = -1 for all i,
    so that :math:`f(x) = \\sum_i \\log(1 + \\exp(x_i))`.
    """
    # Treat matrices elementwise.
    v_shape = v.shape
    v = v.flatten(order='C')
    x0 = x0.flatten(order='C')
    y = y.flatten(order='C')

    # Only works on dense vectors.
    if sparse.issparse(v):
        v = v.todense()
    if sparse.issparse(x0):
        x0 = x0.todense()
    if sparse.issparse(y):
        y = y.todense()

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
    return res.x[0] if res.x.size == 1 else res.x.reshape(v_shape, order='C')

def prox_max_base(v, t):
    """Proximal operator of :math:`f(x) = \\max_i x_i`.
    """
    return v - t*proj_simplex(v/t)
