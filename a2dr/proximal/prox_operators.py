"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy
import warnings
import numpy as np
import numpy.linalg as LA
import scipy as sp
from scipy.optimize import minimize
from scipy.special import expit
from scipy import sparse
import scipy.sparse.linalg

from cvxpy import *
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.add_expr import AddExpression
from a2dr.utilities import flip_obj

class ProxOperator(object):
    def __init__(self, problem, y_vals = {}, rho_vals = {}, use_cvxpy = False):
        self.problem = problem
        self.variables = problem.variables()

        if use_cvxpy:
            self.is_simple = False
            self.prox, self.var_map = self.build_prox_problem(problem, y_vals, rho_vals)
        else:
            if len(self.variables) == 0:   # Only constants.
                self.is_simple = True
                self.var_map = {}
                self.prox = prox_func_vector(problem.objective.args[0], problem.constraints)
            elif len(self.variables) == 1:   # Single variable.
                x_var = self.variables[0]
                is_scalar = len(x_var.shape) <= 1
                objective = flip_obj(problem).args[0]
                if x_var.id not in rho_vals.keys():
                    rho_vals[x_var.id] = 1.0

                # Factor out non-negative scalar constant c from objective and map rho -> rho/c.
                is_scaled, expr, scale = sep_nonneg_scaled(objective.expr)
                if is_scaled:
                    if scale > 0:
                        objective = Minimize(expr)
                        rho_vals[x_var.id] /= scale
                    elif scale == 0:
                        objective = Minimize(0)
                    problem = Problem(objective, problem.constraints)

                # Check if objective has simple proximal operator form.
                self.is_simple = is_simple_prox(objective.expr, problem.constraints, self.variables, is_scalar)
                if self.is_simple:
                    prox_func = prox_func_vector if is_scalar else prox_func_matrix
                    self.prox = prox_func(objective.expr, problem.constraints)
                    self.var_map = {x_var.id: {"x": x_var, "y": Parameter(x_var.shape, value = y_vals.get(x_var.id, np.zeros(x_var.shape))),
                                               "rho": Parameter(value = rho_vals[x_var.id], nonneg = True)}}
                else:
                    self.prox, self.var_map = self.build_prox_problem(problem, y_vals, rho_vals)
            else:   # Multiple variables.
                self.is_simple = False
                self.prox, self.var_map = self.build_prox_problem(problem, y_vals, rho_vals)

    @property
    def status(self):
        return cvxpy.settings.OPTIMAL if self.is_simple else self.prox.status

    @staticmethod
    def build_prox_problem(problem, y_vals = {}, rho_vals = {}):
        var_map = {}
        objective = flip_obj(problem).args[0]

        # Add penalty for each variable.
        for x_var in problem.variables():
            x_id = x_var.id
            x_shape = x_var.shape
            var_map[x_id] = {"x": x_var, "y": Parameter(x_shape, value = y_vals.get(x_id, np.zeros(x_shape))),
                             "rho": Parameter(value = rho_vals.get(x_id, 1.0), nonneg = True)}
            objective += (var_map[x_id]["rho"] / 2.0) * cvxpy.sum_squares(x_var - var_map[x_id]["y"])
        prox = Problem(Minimize(objective), problem.constraints)
        return prox, var_map

    def solve(self, *args, **kwargs):
        if len(self.variables) == 0:
            return
        elif self.is_simple:
            x_var = self.variables[0]
            v_map = self.var_map[x_var.id]
            x_var.value = self.prox(v_map["y"].value, v_map["rho"].value)
        else:
            self.prox.solve(*args, **kwargs)

def sep_nonneg_scaled(f):
    """Is this function multiplied by a non-negative scalar constant? If so, factor out scalar."""
    def is_nonneg_scalar(arg):
        return arg.is_constant() and arg.value is not None and len(np.unique(arg.value)) == 1 and arg.is_nonneg()

    if isinstance(f, multiply):
        if is_nonneg_scalar(f.args[0]):
            return True, f.args[1], np.unique(f.args[0].value)[0]
        elif is_nonneg_scalar(f.args[1]):
            return True, f.args[0], np.unique(f.args[1].value)[0]
        else:
            return False, None, None
    else:
        return False, None, None

def is_ortho_invar(f):
    """Is this function orthogonally invariant?"""
    return (isinstance(f, (cvxpy.normNuc, cvxpy.sigma_max)) and isinstance(f.args[0], Variable)) or \
           (isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det) and isinstance(f.args[0].args[0], Variable))

def is_symm_constr(c):
    """Is this a symmetric constraint, X == X.T?"""
    return isinstance(c, cvxpy.Zero) and isinstance(c.args[0], AddExpression) and \
            isinstance(c.args[0].args[0], Variable) and isinstance(c.args[0].args[1], NegExpression) and \
            isinstance(c.args[0].args[1].args[0], cvxpy.atoms.affine.transpose.transpose) and \
            isinstance(c.args[0].args[1].args[0].args[0], Variable) and c.args[0].args[1].args[0].args[0].id == c.args[0].args[0].id

def is_simple_prox(f, constr, vars, is_scalar):
    """Does this function/constraint have a simple proximal operator?"""
    if len(vars) > 1:   # Single variable problems only.
        return False

    # Reject if variable has attribute constraints.
    for key, value in vars[0].attributes.items():
        if (key == "sparsity" and value is not None) or value:
            return False

    if is_scalar:
        return len(constr) == 0 and ((isinstance(f, (Constant, cvxpy.norm1, cvxpy.norm_inf, cvxpy.abs, cvxpy.entr, cvxpy.exp, cvxpy.huber, cvxpy.max))) or \
                  (isinstance(f, (cvxpy.Pnorm, cvxpy.power)) and f.p == 2) or (isinstance(f, cvxpy.quad_over_lin) and f.args[1].value == 1)) and \
                  (len(f.args) == 0 or isinstance(f.args[0], cvxpy.Variable))
    else:
        return len(constr) == 0 and (isinstance(f, Constant) or \
                  (isinstance(f, cvxpy.atoms.affine.sum.Sum) and len(f.args) == 1 and isinstance(f.args[0], cvxpy.abs) and isinstance(f.args[0].args[0], Variable)) or \
                  (isinstance(f, cvxpy.trace) and (len(f.args) == 1 and isinstance(f.args[0], Variable)) or \
                       (isinstance(f.args[0], MulExpression) and isinstance(f.args[0].args[0], Variable) and isinstance(f.args[0].args[1], Constant))) or \
                  (isinstance(f, cvxpy.Pnorm) and f.p == 2 and isinstance(f.args[0], cvxpy.reshape) and isinstance(f.args[0].args[0], Variable) and \
                        f.args[0].shape == (f.args[0].args[0].size,)) or \
                   is_ortho_invar(f)) # or \
               # len(constr) == 2 and isinstance(f, Constant) and \
               #   ((isinstance(constr[0], cvxpy.constraints.PSD) and is_symm_constr(constr[1])) or \
               #    (isinstance(constr[1], cvxpy.constraints.PSD) and is_symm_constr(constr[0])))

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
    # idx = np.squeeze(np.argwhere(x_diff > 0))[-1]
    idx = np.argwhere(x_diff > 0)[0][-1]
    return np.maximum(x - theta[idx], 0)

def proj_l1(x, r = 1):
    """Project x onto the l1-ball with radius r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 4.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    beta = proj_simplex(np.abs(x), r)
    return np.sign(x) * beta

def prox_logistic(u, rho, x0 = None, y = None):
    """Returns the proximal operator for f(x) = \sum_i log(1 + exp(-y_i*x_i)), where y is a given vector quantity,
       solved using the Newton-CG method from scipy.optimize.minimize. The function defaults to y_i = -1 for all i,
       so that f(x) = \sum_i log(1 + e^x_i).
    """
    if x0 is None:
        x0 = u#np.random.randn(*u.shape)
    if y is None:
        y = -np.ones(u.shape)

    # g(x) = \sum_i log(1 + exp(-y_i*x_i)) + (\rho/2)*||x - u||_2^2
    def fun(x, y, u, rho):
        # expit(x) = 1/(1 + exp(-x))
        return -np.sum(np.log(expit(np.multiply(y,x)))) + (rho/2)*np.sum((x - u)**2)

    # dg(x)/dx_i = -y_i/(1 + exp(y_i*x_i)) + \rho*(x_i - u_i)
    def jac(x, y, u, rho):
        return -np.multiply(y,expit(-np.multiply(y,x))) + rho*(x - u)

    # d^2g(x)/dx_i^2 = y_i^2*exp(y_i*x_i)/(1 + exp(y_i*x_i))^2 + rho
    def hess(x, y, u, rho):
        return np.diag(np.multiply(np.multiply(y**2, np.exp(np.multiply(y,x))), expit(-np.multiply(y,x))**2) + rho)

    res = minimize(fun, x0, args=(y, u, rho), method='Newton-CG', jac=jac, hess=hess)
    #res = minimize(fun, x0, args=(y, u, rho), method='Newton-CG', jac=jac)
    if not res.success:
        warnings.warn(res.message)
    return res.x[0] if res.x.size == 1 else res.x

def prox_norm1(alpha = 1.0):
    return lambda v, t: (v - t*alpha).maximum(0) - (-v - t*alpha).maximum(0) if sparse.issparse(v) else \
                        np.maximum(v - t*alpha,0) - np.maximum(-v - t*alpha,0)

def prox_norm2(alpha = 1.0):
    def prox_norm2_inner(v, t):
        if np.linalg.norm(v) == 0:
            return np.zeros(len(v))
        elif sparse.issparse(v):
            return (1 - t*alpha*1.0/(sparse.linalg.norm(v,'fro'))).maximum(0) * v
        else: 
            return np.maximum(1 - t*alpha*1.0/(LA.norm(v,2)),0) * v 
            
    return lambda v, t: prox_norm2_inner(v, t)

def prox_norm_inf(bound):
    if bound < 0:
        raise ValueError("bound must be a non-negative scalar.")
    return lambda v, t: v.minimum(bound).maximum(-bound) if sparse.issparse(v) else \
                        np.maximum(np.minimum(v, bound), -bound)

def prox_nuc_norm(alpha = 1.0, order = 'C'):
    def prox(Q, t):
        U, s, Vt = np.linalg.svd(Q, full_matrices=False)
        s_new = np.maximum(s - t*alpha, 0)
        Q_new = U.dot(np.diag(s_new)).dot(Vt)
        return Q_new.ravel(order=order)
    return prox

def prox_group_lasso(alpha = 1.0):
    prox_inner = prox_norm2(alpha)
    return lambda Q, t: np.concatenate([prox_inner(Q[:,j], t) for j in range(Q.shape[1])])

def prox_quad_form(Q):
    if sparse.issparse(Q):
        if not np.all(LA.eigvals(Q.todense()) >= 0):
            raise Exception("Q must be a positive semidefinite matrix.")
        return lambda v, t: sparse.linalg.lsqr(Q + (1/t)*sparse.eye(v.shape[0]), v/t, atol=1e-16, btol=1e-16)[0]
    else:
        if not np.all(LA.eigvals(Q) >= 0):
            raise Exception("Q must be a positive semidefinite matrix.")
        return lambda v, t: LA.lstsq(Q + (1/t)*np.eye(v.shape[0]), v/t, rcond=None)[0]
    
def prox_square(v, t):
    return v/(1.0+2*t)

def prox_sat(c, x_max):
    def sat(u, b):
        return np.maximum(np.minimum(u, b), -b)
    return lambda v, t: sat(v/(1.0+2*t*c), x_max)

def prox_sat_pos(c, x_max):
    def sat_pos(u, b):
        return np.maximum(np.minimum(u, b), 0)
    return lambda v, t: sat_pos(v/(1.0+2*t*c), x_max)

    
def prox_sum_squares(X, y, type = "lsqr"):
    n = X.shape[1]
    if type == "lsqr":
        X = sparse.csr_matrix(X)
        def prox(v, t):
            A = sparse.vstack([X, 1/np.sqrt(2*t)*sparse.eye(n)])
            b = np.concatenate([y, 1/np.sqrt(2*t)*v])
            return sparse.linalg.lsqr(A, b, atol=1e-16, btol=1e-16)[0]
    elif type == "lstsq":
        def prox(v, t):
            A = np.vstack([X, 1/np.sqrt(2*t)*np.eye(n)])
            b = np.concatenate([y, 1/np.sqrt(2*t)*v])
            return LA.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError("Algorithm type not supported:", type)
    return prox

def prox_qp(Q, q, F, g):
    # check warmstart/parameter mode -- make sure the problem reduction is only done once
    n = Q.shape[0]
    I = np.eye(n)
    v_par = Parameter(n)
    t_par = Parameter(nonneg=True)
    x = Variable(n)
    obj = quad_form(x, Q) + sum_squares(x)/2/t_par + (q-v_par/t_par)*x
    constr = [F * x <= g]
    prob = Problem(Minimize(obj), constr)
    def prox_qp1(v, t):
        v_par.value, t_par.value = v, t
        prob.solve()
        return x.value
    return prox_qp1

def prox_neg_log_det_aff(Q, S, t, order = 'C'):
    Q_symm = (Q + Q.T) / 2.0
    Q_symm_S = Q_symm - t*S
    s, u = LA.eigh(Q_symm_S)
    s_new = (s + np.sqrt(s**2 + 4.0*t))/2
    Q_new = u.dot(np.diag(s_new)).dot(u.T)
    return Q_new.ravel(order=order)

def prox_func_vector(f, constr = []):
    """Returns the proximal operator for simple functions evaluated at u with scaling factor rho.
       \prox_{(1/\rho) * f}(u) = \argmin_x f(x) + (\rho/2)*||x - u||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    if len(constr) == 0:
        if isinstance(f, Constant):
            return lambda u, rho: u
        elif isinstance(f, cvxpy.norm1):
            return lambda u, rho: np.maximum(np.abs(u) - 1.0/rho, 0) * np.sign(u)
        elif isinstance(f, cvxpy.Pnorm) and f.p == 2:
            return lambda u, rho: np.maximum(1 - 1.0 / (rho * np.linalg.norm(u, 2)), 0) * u
        elif isinstance(f, cvxpy.norm_inf):
            return lambda u, rho: u - (1.0 / rho) * proj_l1(rho * u)
        elif isinstance(f, cvxpy.quad_over_lin) and f.args[1].value == 1:
            return lambda u, rho: (1 / (1 + 1.0/(2 * rho))) * u
        elif isinstance(f, cvxpy.abs):
            return lambda u, rho: np.maximum(u - rho, 0) + np.minimum(u + rho, 0)
        elif isinstance(f, cvxpy.entr):
            return lambda u, rho: -(sp.special.lambertw(u / rho - 1) * np.log(rho)) * rho
        elif isinstance(f, cvxpy.exp):
            return lambda u, rho: u - sp.special.lambertw(np.exp(u + np.log(rho)))
        elif isinstance(f, cvxpy.huber):
            return lambda u, rho: u * 1.0 / (1 + rho) if np.abs(u) < (1 + rho) else u - np.sign(u) * rho
        elif isinstance(f, cvxpy.power) and f.p == 2:
            return lambda u, rho: u / (1 + rho)
        elif isinstance(f, cvxpy.max):
            return lambda u, rho: u - proj_simplex(rho * u) / rho
        else:
            raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
    else:
        raise NotImplementedError("Vector constraints are unimplemented")

def prox_func_matrix(f, constr = []):
    """Returns the proximal operator for matrix functions evaluated at A with scaling factor rho.
       \prox_{(1/\rho) * f}(A) = \argmin_Y f(Y) + (\rho/2)*||Y - A||_2^2

       References:
       1) N. Parikh and S. Boyd (2013). "Proximal Algorithms." https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
       2) A. Beck (2017). "First-Order Methods in Optimization." https://archive.siam.org/books/mo25/mo25_ch6.pdf
    """
    if len(constr) == 0:
        if isinstance(f, Constant):
            return lambda A, rho: A
        if isinstance(f, cvxpy.atoms.affine.sum.Sum) and len(f.args) == 1 and isinstance(f.args[0], cvxpy.abs) and \
                isinstance(f.args[0].args[0], Variable):
            return lambda A, rho: np.maximum(np.abs(A) - 1.0/rho, 0) * np.sign(A)
        elif isinstance(f, cvxpy.trace):
            if len(f.args) == 1 and isinstance(f.args[0], Variable):
                return lambda A, rho: A - np.diag(np.full((A.shape[0],), 1.0/rho))
            elif isinstance(f.args[0], MulExpression) and \
                    isinstance(f.args[0].args[0], Variable) and isinstance(f.args[0].args[1], Constant):
                return lambda A, rho: A - f.args[0].args[1].value.T / rho
            else:
                raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
        elif isinstance(f, cvxpy.Pnorm) and f.p == 2 and \
                isinstance(f.args[0], cvxpy.reshape) and isinstance(f.args[0].args[0], Variable) and \
                f.args[0].shape == (f.args[0].args[0].size,):
            def prox(A, rho):
                u = np.asarray(A).ravel()
                u_2norm = np.linalg.norm(u, 2)
                if u_2norm < 1e-8:
                    return A
                else:
                    prox_vec = np.maximum(1 - 1.0 / (rho * u_2norm), 0) * u
                    return np.reshape(prox_vec, A.shape)
            return prox
        elif is_ortho_invar(f):
            if isinstance(f, cvxpy.normNuc):
                prox_diag = lambda s, rho: np.maximum(s - 1.0/rho, 0)
            elif isinstance(f, cvxpy.sigma_max):
                prox_diag = lambda s, rho: 1.0/rho * proj_simplex(rho * s)
            elif isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det):
                prox_diag = lambda s, rho: (s + np.sqrt(s ** 2 + 4.0/rho)) / 2
            else:
                raise ValueError("Unimplemented orthogonally invariant function {0}".format(f.__class__.__name__))

            def prox_inner(A, rho):
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_new = prox_diag(s, rho)
                return U.dot(np.diag(s_new)).dot(Vt)

            if isinstance(f, NegExpression) and isinstance(f.args[0], cvxpy.log_det):
                def prox(A, rho):
                    A_symm = (A + A.T) / 2.0
                    if not (np.allclose(A, A_symm)):
                        raise Exception("Proximal operator for negative log-determinant only operates on symmetric matrices.")
                    w, v = np.linalg.eig(A_symm)
                    w_new = np.maximum(w, 0)
                    return v.dot(np.diag(w_new)).dot(v.T)
            else:
                prox = prox_inner
            return prox
        else:
            raise ValueError("Unsupported atom instance {0}".format(f.__class__.__name__))
    elif len(constr) == 2 and isinstance(f, Constant) and \
            ((isinstance(constr[0], cvxpy.constraints.PSD) and is_symm_constr(constr[1])) or \
             (isinstance(constr[1], cvxpy.constraints.PSD) and is_symm_constr(constr[0]))):
        def prox(A, rho):
            A_symm = (A + A.T)/2.0
            if not np.allclose(A, A_symm):
                raise Exception("Proximal operator for positive semidefinite cone only operates on symmetric matrices.")
            w, v = np.linalg.eig(A_symm)
            w_new = np.maximum(w, 0)
            return v.dot(np.diag(w_new)).dot(v.T)
        return prox
    else:
        raise NotImplementedError("Multiple constraints are unimplemented except for [X >> 0, X == X.T]")

