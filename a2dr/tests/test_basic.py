import numpy as np
import scipy as sp
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt

from cvxpy import *
from scipy import sparse

from a2dr import a2dr
from a2dr.proximal.prox_operators import prox_logistic
from a2dr.tests.base_test import BaseTest

def prox_norm1(alpha = 1.0):
    return lambda v, t: (v - t*alpha).maximum(0) - (-v - t*alpha).maximum(0) if sparse.issparse(v) else \
                        np.maximum(v - t*alpha,0) - np.maximum(-v - t*alpha,0)

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

class TestBasic(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8  # specify these in all examples?
        self.eps_abs = 1e-6
        self.MAX_ITER = 1000

    def test_unconstrained(self):
        # minimize ||y - X\beta||_2^2.

        # Problem data.
        m, n = 100, 80
        density = 0.1
        X = sparse.random(m, n, density=density, data_rvs=np.random.randn)
        y = np.random.randn(m)
        prox_list = [prox_sum_squares(X, y)]

        # Solve with NumPy.
        np_result = LA.lstsq(X.todense(), y, rcond=None)
        np_beta = np_result[0]
        np_obj = np.sum(np_result[1])

        # Solve with DRS.
        drs_result = a2dr(prox_list, n_list=[n], anderson=False, max_iter=self.MAX_ITER)
        drs_beta = drs_result["x_vals"][-1]
        drs_obj = np.sum((y - X.dot(drs_beta))**2)
        print("Finish DRS.")

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, n_list=[n], anderson=True, max_iter=self.MAX_ITER)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
        print("Finish A2DR.")
        self.compare_total(drs_result, a2dr_result)
        # self.assertAlmostEqual(np_obj, drs_obj)
        # self.assertAlmostEqual(np_obj, a2dr_obj)

    def test_ols(self):
        # minimize ||y - X\beta||_2^2 with respect to \beta >= 0.
        m = 100
        n = 10
        N = 4  # Number of splits.
        beta_true = np.array(np.arange(-n / 2, n / 2) + 1)
        X = np.random.randn(m, n)
        y = X.dot(beta_true) + np.random.randn(m)

        # Split problem.
        X_split = np.split(X, N)
        y_split = np.split(y, N)
        p_list = [prox_sum_squares(X_sub, y_sub) for X_sub, y_sub in zip(X_split, y_split)]
        v_init = N * [np.random.randn(n)]

        # Solve with NumPy.
        np_beta = []
        np_obj = 0
        for i in range(N):
            np_result = LA.lstsq(X_split[i], y_split[i], rcond=None)
            np_beta += [np_result[0]]
            np_obj += np.sum(np_result[1])
        print("NumPy Objective:", np_obj)
        print("NumPy Solution:", np_beta)

        # Solve with DRS (proximal point method).
        drs_result = a2dr(p_list, v_init=v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                          eps_rel=self.eps_rel, anderson=False)
        drs_beta = drs_result["x_vals"]
        drs_obj = np.sum([(yi - Xi.dot(beta)) ** 2 for yi, Xi, beta in zip(y_split, X_split, drs_beta)])
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)

        # Solve with A2DR (proximal point method with Anderson acceleration).
        a2dr_result = a2dr(p_list, v_init=v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_beta = a2dr_result["x_vals"]
        a2dr_obj = np.sum([(yi - Xi.dot(beta)) ** 2 for yi, Xi, beta in zip(y_split, X_split, drs_beta)])
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)

        # Compare results.
        # self.assertAlmostEqual(np_obj, drs_obj)
        # self.assertAlmostEqual(np_obj, a2dr_obj)
        # for i in range(N):
        #    self.assertItemsAlmostEqual(np_beta[i], drs_beta[i])
        #    self.assertItemsAlmostEqual(np_beta[i], a2dr_beta[i])

        # Plot residuals.
        plt.semilogy(range(drs_result["num_iters"]), np.sqrt(drs_result["primal"]**2 + drs_result["dual"]**2), label="DRS")
        plt.semilogy(range(a2dr_result["num_iters"]), np.sqrt(a2dr_result["primal"]**2 + a2dr_result["dual"]**2), label="A2DR")
        plt.title("Residual Norms")
        plt.legend()
        plt.show()

    def test_nnls(self):
        # minimize ||y - X\beta||_2^2 subject to \beta >= 0.

        # Problem data.
        m, n = 100, 80
        density = 0.1
        X = sparse.random(m, n, density=density, data_rvs=np.random.randn)
        y = np.random.randn(m)

        # Convert problem to standard form.
        # f_1(\beta_1) = ||y - X\beta_1||_2^2, f_2(\beta_2) = I(\beta_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [prox_sum_squares(X, y), lambda v, t: v.maximum(0) if sparse.issparse(v) else np.maximum(v, 0)]
        A_list = [sparse.eye(n), -sparse.eye(n)]
        b = np.zeros(n)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finish DRS.')

        # Solve with A2DR.
        t0 = time.time()
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        t1 = time.time()
        a2dr_beta = a2dr_result["x_vals"][-1]
        print('Proportion of nonzero entries = {}'.format(np.sum(a2dr_beta > 0) * 1.0 / len(a2dr_beta)))
        print('Finish A2DR.')
        self.compare_total(drs_result, a2dr_result)

        # Check solution correctness.
        print('A2DR runtime = {}'.format(t1 - t0))
        print('A2DR constraint violation = {}'.format(np.min(a2dr_beta)))
        print('A2DR objective value = {}'.format(np.linalg.norm(X.dot(a2dr_beta) - y)))

    def test_l1_trend_filtering(self):
        # minimize (1/2)||y - x||_2^2 + \alpha*||Dx||_1,
        # where (Dx)_{t-1} = x_{t-1} - 2*x_t + x_{t+1} for t = 2,...,n-1.
        # Reference: https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

        # Problem data.
        n = int(2*10**2)
        y = np.random.randn(n)
        alpha = 0.01*np.linalg.norm(y, np.inf)

        # Form second difference matrix.
        D = sparse.lil_matrix(sparse.eye(n))
        D.setdiag(-2, k = 1)
        D.setdiag(1, k = 2)
        D = D[:(n-2),:]

        # Convert problem to standard form.
        # f_1(x_1) = (1/2)||y - x_1||_2^2, f_2(x_2) = \alpha*||x_2||_1.
        # A_1 = D, A_2 = -I_{n-2}, b = 0.
        prox_list = [lambda v, t: (t*y + v)/(t + 1.0), prox_norm1(alpha)]
        A_list = [D, -sparse.eye(n-2)]
        b = np.zeros(n-2)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result)
        print('Finished A2DR.')
