import cvxpy
import numpy as np
import scipy as sp

def prox_func(f, x, rho):
    if isinstance(f, cvxpy.Constant):
        return x
    elif isinstance(f, cvxpy.norm1):
        return np.maximum(np.abs(x) - rho, 0) * np.sign(x)
    elif isinstance(f, cvxpy.norm2):
        return np.maximum(1 - rho/np.norm(x,2), 0) * x
    elif isinstance(f, cvxpy.normNuc):
        u, s, vt = np.linalg.svd(x, full_matrices = False)
        s_new = np.maximum(s - rho, 0)
        return u.dot(np.diag(s_new)).dot(vt)
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
