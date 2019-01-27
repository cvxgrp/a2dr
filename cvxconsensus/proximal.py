import cvxpy
from cvxpy.atoms.affine.unary_operators import NegExpression
import numpy as np
import scipy as sp

def proj_simplex(x, r = 1):
    """Project x onto a simplex with upper bound r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 3.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    x_decr = np.sort(x, axis = None)[::-1]
    x_cumsum = np.cumsum(x_decr)
    denom = 1 + np.arange(len(x_decr))
    theta = (x_cumsum - r)/denom
    x_diff = x_decr - theta
    idx = np.squeeze(np.argwhere(x_diff > 0))[-1]
    return np.maximum(x - theta[idx], 0)

# Project x onto the l1 ball with radius r.
def proj_l1(x, r = 1):
    """Project x onto the l1-ball with radius r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 4.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    beta = proj_simplex(np.abs(x), r)
    return np.sign(x) * beta

def prox_func(f, x, rho):
    """Returns the proximal operator for simple functions evaluated at x with scaling factor rho.
       \prox_{\rho * f}(x) = \argmin_y f(y) + 1/(2*\rho)*||y - x||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    if isinstance(f, cvxpy.Constant):
        return x
    elif isinstance(f, cvxpy.norm1):
        return np.maximum(np.abs(x) - rho, 0) * np.sign(x)
    elif isinstance(f, cvxpy.norm2):
        return np.maximum(1 - rho/np.norm(x,2), 0) * x
    elif isinstance(f, cvxpy.norm_inf):
        return x - rho*proj_l1(x/rho)
    elif isinstance(f, cvxpy.sum_squares):
        return (1 / (1 + rho/2)) * x
    elif isinstance(f, cvxpy.abs):
        return max(x - 1/rho, 0) + min(x + 1/rho, 0)
    elif isinstance(f, cvxpy.entr):
        return (sp.special.lambertw(rho*x - 1) * np.log(rho)) / rho
    elif isinstance(f, cvxpy.exp):
        return x - sp.special.lambertw(np.exp(x - np.log(rho)))
    elif isinstance(f, cvxpy.huber):
        return x * rho / (1 + rho) if np.abs(x) < (1 + 1/rho) else x - np.sign(x) / rho
    elif isinstance(f, cvxpy.square):
        return rho * x / (1 + rho)
    elif isinstance(f, cvxpy.max):
        return x - rho*proj_simplex(x/rho)

def prox_func_mat(f, A, rho):
    """Returns the proximal operator for matrix functions evaluated at A with scaling factor rho.
       \prox_{\rho * f}(A) = \argmin_Y f(Y) + 1/(2*\rho)*||Y - A||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    U, s, Vt = np.linalg.svd(A, full_matrices = False)
    if isinstance(f, cvxpy.normNuc):
        s_new = np.maximum(s - rho, 0)
    elif isinstance(f, cvxpy.lambda_max):
        s_new = s - prox_func(cvxpy.max, s, rho)
    elif isinstance(f, cvxpy.trace):
        s_new = np.full(s.shape, rho)
    elif isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det):
        s_new = (s + np.sqrt(s^2 + 4*rho))/2
    return U.dot(np.diag(s_new)).dot(Vt)
