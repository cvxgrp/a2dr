import numpy as np
import scipy as sp
import cvxpy
from cvxpy import Constant, Variable
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.binary_operators import MulExpression

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

def proj_l1(x, r = 1):
    """Project x onto the l1-ball with radius r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 4.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    beta = proj_simplex(np.abs(x), r)
    return np.sign(x) * beta

def prox_func(f, u, rho, constr = []):
    if (np.isscalar(u) or len(u.shape) <= 1) and len(constr) == 0:
        return prox_func_scalar(f, u, rho)
    else:
        return prox_func_matrix(f, u, rho, constr)

def prox_func_scalar(f, u, rho):
    """Returns the proximal operator for simple functions evaluated at u with scaling factor rho.
       \prox_{\rho * f}(u) = \argmin_x f(x) + 1/(2*\rho)*||x - u||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    if isinstance(f, Constant):
        return u
    elif isinstance(f, cvxpy.norm1):
        return np.maximum(np.abs(u) - rho, 0) * np.sign(u)
    elif isinstance(f, cvxpy.Pnorm) and f.p == 2:
        return np.maximum(1 - rho / np.linalg.norm(u, 2), 0) * u
    elif isinstance(f, cvxpy.norm_inf):
        return u - rho * proj_l1(u / rho)
    elif isinstance(f, cvxpy.quad_over_lin) and f.args[1].value == 1:
        return (1 / (1 + rho/2)) * u
    elif isinstance(f, cvxpy.abs):
        return max(u - 1 / rho, 0) + min(u + 1 / rho, 0)
    elif isinstance(f, cvxpy.entr):
        return (sp.special.lambertw(rho * u - 1) * np.log(rho)) / rho
    elif isinstance(f, cvxpy.exp):
        return u - sp.special.lambertw(np.exp(u - np.log(rho)))
    elif isinstance(f, cvxpy.huber):
        return u * rho / (1 + rho) if np.abs(u) < (1 + 1 / rho) else u - np.sign(u) / rho
    elif isinstance(f, cvxpy.power) and f.p == 2:
        return rho * u / (1 + rho)
    elif isinstance(f, cvxpy.max):
        return u - rho * proj_simplex(u / rho)
    else:
        raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))

def prox_func_matrix(f, A, rho, constr = []):
    """Returns the proximal operator for matrix functions evaluated at A with scaling factor rho.
       \prox_{\rho * f}(A) = \argmin_Y f(Y) + 1/(2*\rho)*||Y - A||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    U, s, Vt = np.linalg.svd(A, full_matrices = False)
    if isinstance(f, cvxpy.normNuc):
        s_new = np.maximum(s - rho, 0)
    elif isinstance(f, cvxpy.Pnorm) and f.p == 2 and \
            isinstance(f.args[0], cvxpy.reshape) and isinstance(f.args[0].args[0], Variable) and \
            f.args[0].shape == (f.args[0].args[0].size,):
        prox_vec = prox_func_scalar(f, np.asarray(A).ravel(), rho)
        return np.reshape(prox_vec, A.shape)
    elif isinstance(f, cvxpy.sigma_max):
        s_new = s - prox_func_scalar(cvxpy.max(s), s, rho)
    elif isinstance(f, cvxpy.trace):
        if len(f.args) == 1 and isinstance(f.args[0], Variable):
            return A - np.diag(np.full(s.shape, rho))
        elif isinstance(f.args[0], MulExpression) and \
                isinstance(f.args[0].args[0], Variable) and isinstance(f.args[0].args[1], Constant):
            return A - rho*f.args[0].args[1].value.T
        else:
            raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
    elif isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det):
        s_new = (s + np.sqrt(s**2 + 4*rho))/2
    elif isinstance(f, cvxpy.atoms.affine.sum.Sum) and \
            len(f.args) == 1 and isinstance(f.args[0], cvxpy.abs):
        return np.maximum(np.abs(A) - rho, 0) * np.sign(A)
    elif isinstance(f, cvxpy.Pnorm) and f.p == 2:
        return np.maximum(1 - rho / np.linalg.norm(A,"fro"), 0) * A
    elif isinstance(f, Constant) and \
            len(constr) == 1 and isinstance(constr[0], cvxpy.constraints.PSD) and \
            isinstance(constr[0].args[0], Variable) and constr[0].args[0].is_symmetric() and \
            A.shape == A.T.shape and np.allclose(A, A.T, 1e-8):
        s_new = np.maximum(s, 0)
    else:
        raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
    return U.dot(np.diag(s_new)).dot(Vt)
