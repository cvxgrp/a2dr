import cvxpy
import numpy as np
import scipy as sp
from cvxpy import Constant, Variable, Parameter, Problem, Minimize
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxconsensus.utilities import flip_obj

class ProxOperator(object):
    def __init__(self, problem, rho_init):
        self.problem = problem
        self.variables = problem.variables()

        if len(self.variables) == 0:   # Only constants.
            self.is_simple = True
            self.prox = prox_func_vector(problem.objective.args[0], problem.constraints)
            self.var_map = {}
        elif len(self.variables) == 1:   # Single variable.
            x_var = self.variables[0]
            is_scalar = len(x_var.shape) <= 1
            self.is_simple = is_simple_prox(problem.objective.args[0], problem.constraints, is_scalar)
            if self.is_simple:
                prox_func = prox_func_vector if is_scalar else prox_func_matrix
                self.prox = prox_func(problem.objective.args[0], problem.constraints)
                self.var_map = {x_var.id: {"x": x_var, "y": Parameter(x_var.shape, value=np.zeros(x_var.shape)),
                                           "rho": Parameter(value=rho_init[x_var.id], nonneg=True)}}
            else:
                self.prox, self.var_map = self.build_prox_problem(problem, rho_init)
        else:   # Multiple variables.
            self.is_simple = False
            self.prox, self.var_map = self.build_prox_problem(problem, rho_init)

    @staticmethod
    def build_prox_problem(problem, rho_init):
        var_map = {}
        objective = flip_obj(problem).args[0]

        # Add penalty for each variable.
        for x_var in problem.variables():
            x_id = x_var.id
            x_shape = x_var.shape
            var_map[x_id] = {"x": x_var, "y": Parameter(x_shape, value=np.zeros(x_shape)),
                             "rho": Parameter(value=rho_init[x_id], nonneg=True)}
            objective += (var_map[x_id]["rho"] / 2.0) * cvxpy.sum_squares(x_var - var_map[x_id]["y"])
        prox = Problem(Minimize(objective), problem.constraints)
        return prox, var_map

    def solve(self, *args, **kwargs):
        if len(self.variables) == 0:
            return
        elif self.is_simple and len(self.variables) == 1:
            x_var = self.variables[0]
            v_map = self.var_map[x_var.id]
            x_var.value = self.prox(v_map["y"].value, v_map["rho"].value)
        else:
            self.prox.solve(*args, **kwargs)

def is_ortho_invar(f):
    return isinstance(f, (cvxpy.normNuc, cvxpy.sigma_max)) or \
           (isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det))

def is_simple_prox(f, constr, is_scalar):
    if is_scalar:
        return len(constr) == 0 and ((isinstance(f, (Constant, cvxpy.norm1, cvxpy.norm_inf, cvxpy.abs, cvxpy.entr, cvxpy.exp, cvxpy.huber, cvxpy.max))) or \
                  (isinstance(f, (cvxpy.Pnorm, cvxpy.power)) and f.p == 2) or (isinstance(f, cvxpy.quad_over_lin) and f.args[1].value == 1))
    else:
        return len(constr) == 0 and (isinstance(f, Constant) or \
                  (isinstance(f, cvxpy.atoms.affine.sum.Sum) and len(f.args) == 1 and isinstance(f.args[0], cvxpy.abs)) or \
                  (isinstance(f, cvxpy.Pnorm) and f.p == 2 and isinstance(f.args[0], cvxpy.reshape) and isinstance(f.args[0].args[0], Variable) and \
                       f.args[0].shape == (f.args[0].args[0].size,)) or \
                  (isinstance(f, cvxpy.trace) and (len(f.args) == 1 and isinstance(f.args[0], Variable)) or \
                       (isinstance(f.args[0], MulExpression) and isinstance(f.args[0].args[0], Variable) and isinstance(f.args[0].args[1], Constant))) or \
                  (is_ortho_invar(f) and (isinstance(f, (cvxpy.normNuc, cvxpy.sigma_max)) or \
                                          isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det)))) or \
              len(constr) == 1 and isinstance(f, Constant) and \
                  (isinstance(constr[0], cvxpy.constraints.PSD) and isinstance(constr[0].args[0], Variable) and constr[0].args[0].is_symmetric())

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

def prox_func_vector(f, constr = []):
    """Returns the proximal operator for simple functions evaluated at u with scaling factor rho.
       \prox_{\rho * f}(u) = \argmin_x f(x) + 1/(2*\rho)*||x - u||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    if len(constr) == 0:
        if isinstance(f, Constant):
            return lambda u, rho: u
        elif isinstance(f, cvxpy.norm1):
            return lambda u, rho: np.maximum(np.abs(u) - rho, 0) * np.sign(u)
        elif isinstance(f, cvxpy.Pnorm) and f.p == 2:
            return lambda u, rho: np.maximum(1 - rho / np.linalg.norm(u, 2), 0) * u
        elif isinstance(f, cvxpy.norm_inf):
            return lambda u, rho: u - rho * proj_l1(u / rho)
        elif isinstance(f, cvxpy.quad_over_lin) and f.args[1].value == 1:
            return lambda u, rho: (1 / (1 + rho/2)) * u
        elif isinstance(f, cvxpy.abs):
            return lambda u, rho: np.maximum(u - 1 / rho, 0) + np.minimum(u + 1 / rho, 0)
        elif isinstance(f, cvxpy.entr):
            return lambda u, rho: (sp.special.lambertw(rho * u - 1) * np.log(rho)) / rho
        elif isinstance(f, cvxpy.exp):
            return lambda u, rho: u - sp.special.lambertw(np.exp(u - np.log(rho)))
        elif isinstance(f, cvxpy.huber):
            return lambda u, rho: u * rho / (1 + rho) if np.abs(u) < (1 + 1 / rho) else u - np.sign(u) / rho
        elif isinstance(f, cvxpy.power) and f.p == 2:
            return lambda u, rho: rho * u / (1 + rho)
        elif isinstance(f, cvxpy.max):
            return lambda u, rho: u - rho * proj_simplex(u / rho)
        else:
            raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
    else:
        raise NotImplementedError("Vector constraints are unimplemented")

def prox_func_matrix(f, constr = []):
    """Returns the proximal operator for matrix functions evaluated at A with scaling factor rho.
       \prox_{\rho * f}(A) = \argmin_Y f(Y) + 1/(2*\rho)*||Y - A||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    if len(constr) == 0:
        if isinstance(f, Constant):
            return lambda A, rho: A
        if isinstance(f, cvxpy.atoms.affine.sum.Sum) and \
           len(f.args) == 1 and isinstance(f.args[0], cvxpy.abs):
            return lambda A, rho: np.maximum(np.abs(A) - rho, 0) * np.sign(A)
        elif isinstance(f, cvxpy.trace):
            if len(f.args) == 1 and isinstance(f.args[0], Variable):
                return lambda A, rho: A - np.diag(np.full((A.shape[0],), rho))
            elif isinstance(f.args[0], MulExpression) and \
                    isinstance(f.args[0].args[0], Variable) and isinstance(f.args[0].args[1], Constant):
                return lambda A, rho: A - rho * f.args[0].args[1].value.T
            else:
                raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
        elif isinstance(f, cvxpy.Pnorm) and f.p == 2 and \
                isinstance(f.args[0], cvxpy.reshape) and isinstance(f.args[0].args[0], Variable) and \
                f.args[0].shape == (f.args[0].args[0].size,):
            def prox(A, rho):
                u = np.asarray(A).ravel()
                prox_vec = np.maximum(1 - rho / np.linalg.norm(u, 2), 0) * u
                return np.reshape(prox_vec, A.shape)
            return prox
        elif is_ortho_invar(f):
            if isinstance(f, cvxpy.normNuc):
                prox_diag = lambda s, rho: np.maximum(s - rho, 0)
            elif isinstance(f, cvxpy.sigma_max):
                prox_diag = lambda s, rho: rho * proj_simplex(s / rho)
            elif isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det):
                prox_diag = lambda s, rho: (s + np.sqrt(s ** 2 + 4 * rho)) / 2
            else:
                raise ValueError("Unimplemented orthogonally invariant function {0}".format(f.__class__.__name__))
            def prox(A, rho):
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_new = prox_diag(s, rho)
                return U.dot(np.diag(s_new)).dot(Vt)
            return prox
        else:
            raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
    elif len(constr) == 1 and isinstance(f, Constant):
        if isinstance(constr[0], cvxpy.constraints.PSD) and \
              isinstance(constr[0].args[0], Variable) and constr[0].args[0].is_symmetric():
            def prox(A, rho):
                if not (A.shape == A.T.shape and np.allclose(A, A.T, 1e-8)):
                    raise ValueError("A must be a symmetric matrix")
                U, s, Vt = np.linalg.svd(A, full_matrices = False)
                s_new = np.maximum(s, 0)
                return U.dot(np.diag(s_new)).dot(Vt)
            return prox
        else:
            raise ValueError("Unsupported constraint instance {0}".format(constr[0].__class.__.__name__))
    else:
        raise NotImplementedError("Multiple constraints are unimplemented")

