"""
Copyright 2018 Anqi Fu

This file is part of CVXConsensus.

CVXConsensus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXConsensus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXConsensus. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy as sp
import numpy.linalg as LA
import matplotlib.pyplot as plt
from cvxpy import *
from scipy.optimize import nnls
from cvxconsensus import a2dr
from cvxconsensus.tests.base_test import BaseTest

def prox_sum_squares(X, y, rcond = None):
    n = X.shape[1]
    def prox(v, rho):
        A = np.vstack((X, np.sqrt(rho/2)*np.eye(n)))
        b = np.concatenate((y, np.sqrt(rho/2)*v))
        return LA.lstsq(A, b, rcond=rcond)[0]
    return prox

def prox_neg_log_det(v, rho):
    n = int(np.sqrt(v.shape[0]))
    A = np.reshape(v, (n,n), order='C')
    A_symm = (A + A.T) / 2.0
    if not (np.allclose(A, A_symm) and np.all(LA.eigvals(A_symm) > 0)):
        raise Exception("Proximal operator for negative log-determinant only operates on symmetric positive definite matrices.")
    U, s, Vt = LA.svd(A_symm, full_matrices=False)
    s_new = (s + np.sqrt(s ** 2 + 4.0/rho)) / 2
    A_new = U.dot(np.diag(s_new)).dot(Vt)
    return A_new.ravel(order='C')

def prox_pos_semidef(v, rho):
    n = int(np.sqrt(v.shape[0]))
    A = np.reshape(v, (n,n), order='C')
    A_symm = (A + A.T) / 2.0
    if not np.allclose(A, A_symm):
        raise Exception("Proximal operator for positive semidefinite cone only operates on symmetric matrices.")
    w, v = LA.eig(A_symm)
    w_new = np.maximum(w, 0)
    A_new = v.dot(np.diag(w_new)).dot(v.T)
    return A_new.ravel(order='C')

def prox_quad_form(Q):
    if not np.all(LA.eigvals(Q) >= 0):
        raise Exception("Q must be a positive semidefinite matrix.")
    return lambda v, rho: LA.lstsq(Q + rho*np.eye(v.shape[0]), rho*v, rcond=None)[0]

def prox_norm_inf(bound):
    if bound < 0:
        raise ValueError("bound must be a non-negative scalar.")
    return lambda v, rho: np.maximum(np.minimum(v, bound), -bound)

class TestSolver(BaseTest):
    """Unit tests for internal A2DR solver."""

    def setUp(self):
        np.random.seed(1)
        self.eps_stop = 1e-8
        self.eps_abs = 1e-16
        self.MAX_ITER = 2000

    def test_ols(self):
        # minimize ||y - X\beta||_2^2 with respect to \beta >= 0.
        m = 100
        n = 10
        N = 4  # Number of splits.
        beta_true = np.array(np.arange(-n/2,n/2) + 1)
        X = np.random.randn(m, n)
        y = X.dot(beta_true) + np.random.randn(m)

        # Split problem.
        X_split = np.split(X, N)
        y_split = np.split(y, N)
        p_list = [prox_sum_squares(X_sub, y_sub) for X_sub, y_sub in zip(X_split, y_split)]
        v_init = N*[np.random.randn(n)]

        # Solve with NumPy.
        np_beta = []
        np_obj = 0
        for i in range(N):
            np_result = np.linalg.lstsq(X_split[i], y_split[i], rcond=None)
            np_beta += [np_result[0]]
            np_obj += np.sum(np_result[1])
        print("NumPy Objective:", np_obj)
        print("NumPy Solution:", np_beta)

        # Solve with DRS (proximal point method).
        drs_result = a2dr(p_list, v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, eps_stop=self.eps_stop, anderson=False)
        drs_beta = drs_result["x_vals"]
        drs_obj = np.sum([(yi - Xi.dot(beta))**2 for yi,Xi,beta in zip(y_split,X_split,drs_beta)])
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)

        # Solve with A2DR (proximal point method with Anderson acceleration).
        a2dr_result = a2dr(p_list, v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, eps_stop=self.eps_stop, anderson=True)
        a2dr_beta = a2dr_result["x_vals"]
        a2dr_obj = np.sum([(yi - Xi.dot(beta))**2 for yi, Xi, beta in zip(y_split, X_split, drs_beta)])
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)

        # Compare results.
        self.assertAlmostEqual(np_obj, drs_obj)
        self.assertAlmostEqual(np_obj, a2dr_obj)
        for i in range(N):
            self.assertItemsAlmostEqual(np_beta[i], drs_beta[i])
            self.assertItemsAlmostEqual(np_beta[i], a2dr_beta[i])

        # Plot residuals.
        plt.semilogy(range(drs_result["num_iters"]), drs_result["dual"], label = "DRS")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["dual"], label = "A2DR")
        plt.title("Dual Residuals")
        plt.legend()
        plt.show()

    def test_nnls(self):
        # minimize ||y - X\beta||_2^2 with respect to \beta >= 0.
        m = 100
        n = 10
        N = 4   # Number of splits.
        beta_true = np.array(np.arange(-n/2,n/2) + 1)
        X = np.random.randn(m,n)
        y = X.dot(beta_true) + np.random.randn(m)

        # Solve with SciPy.
        sp_result = nnls(X, y)
        sp_beta = sp_result[0]
        sp_obj = sp_result[1]**2   # SciPy objective is ||y - X\beta||_2.
        print("Scipy Objective:", sp_obj)
        print("SciPy Solution:", sp_beta)

        # Split problem.
        X_split = np.split(X,N)
        y_split = np.split(y,N)
        p_list = [prox_sum_squares(X_sub, y_sub) for X_sub, y_sub in zip(X_split, y_split)]
        p_list += [lambda u, rho: np.maximum(u, 0)]   # Projection onto non-negative orthant.
        v_init = (N + 1)*[np.random.randn(n)]
        A_list = np.hsplit(np.eye(N*n),N) + [-np.vstack(N*(np.eye(n),))]
        b = np.zeros(N*n)

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, eps_stop=self.eps_stop, anderson=False)
        drs_beta = drs_result["x_vals"][-1]
        drs_obj = np.sum((y - X.dot(drs_beta))**2)
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)
        self.assertAlmostEqual(sp_obj, drs_obj)
        self.assertItemsAlmostEqual(sp_beta, drs_beta, places=3)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], \
                            normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, eps_stop=self.eps_stop, anderson=True)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)
        self.assertAlmostEqual(sp_obj, a2dr_obj)
        self.assertItemsAlmostEqual(sp_beta, a2dr_beta, places=3)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], \
                            normalize=True, title="A2DR Residuals", semilogy=True)

        # Compare residuals
        plt.semilogy(range(drs_result["num_iters"]), drs_result["primal"], color="blue", linestyle="--", label="Primal (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["primal"], color="blue", label="Primal (A2DR)")
        plt.semilogy(range(drs_result["num_iters"]), drs_result["dual"], color="darkorange", linestyle="--", label="Dual (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["dual"], color="darkorange", label="Dual (A2DR) ")
        plt.title("Residuals")
        plt.legend()
        plt.show()

    def test_sparse_covariance(self):
        # minimize -log(det(S)) + trace(S*Y) + \alpha*||S||_1
        #   subject to S is PSD, where Y and \alpha >= are parameters.

        # Problem data.
        m = 10  # Dimension of matrix.
        n = m*m   # Length of vectorized matrix.
        K = 1000  # Number of samples.
        A = np.random.randn(m,m)
        A[sp.sparse.rand(m,m,0.85).todense().nonzero()] = 0
        S_true = A.dot(A.T) + 0.05 * np.eye(m)
        R = np.linalg.inv(S_true)
        y_sample = sp.linalg.sqrtm(R).dot(np.random.randn(m,K))
        Y = np.cov(y_sample)

        # Solve with CVXPY.
        S = Variable((m,m), PSD=True)
        alpha = Parameter(nonneg=True)
        obj = -log_det(S) + trace(S*Y) + alpha*norm1(S)
        prob = Problem(Minimize(obj))

        alpha.value = 1.0
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_S = S.value
        print("CVXPY Objective:", cvxpy_obj)
        # print("CVXPY Solution:", cvxpy_S)

        # Split problem as f_1(S) = -log(det(S)), f_2(S) = trace(S*Y),
        #   f_3(S) = \alpha*||S||_1, f_4(S) = I(S is symmetric PSD).
        N = 3
        p_list = [prox_neg_log_det,
                  lambda v, rho: v - Y.ravel(order='C')/rho,
                  lambda v, rho: np.maximum(np.abs(v) - alpha.value/rho, 0) * np.sign(v),
                  prox_pos_semidef]
        S_init = np.random.randn(m,m)
        S_init = S_init.dot(S_init.T)   # Ensure starting point is symmetric PSD.
        v_init = (N + 1)*[S_init.ravel(order='C')]
        A_list = np.hsplit(np.eye(N*n),N) + [-np.vstack(N*(np.eye(n),))]
        b = np.zeros(N*n)

        # Solve with DRS.
        # TODO: This fails because S is no longer PSD after a few iterations.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_stop=self.eps_stop, anderson=False)
        drs_S = drs_result["x_vals"][-1].reshape((m,m), order='C')
        drs_obj = -np.log(np.linalg.det(drs_S)) + np.sum(np.diag(drs_S.dot(Y))) + alpha.value*np.sum(np.abs(drs_S))
        print("DRS Objective:", drs_obj)
        # print("DRS Solution:", drs_S)
        self.assertAlmostEqual(cvxpy_obj, drs_obj)
        self.assertItemsAlmostEqual(cvxpy_S, drs_S, places=3)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], \
                            normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_stop=self.eps_stop, anderson=True)
        a2dr_S = a2dr_result["x_vals"][-1]
        a2dr_obj = -np.log(np.linalg.det(drs_S)) + np.sum(np.diag(drs_S.dot(Y))) + alpha.value*np.sum(np.abs(drs_S))
        print("A2DR Objective:", a2dr_obj)
        # print("A2DR Solution:", a2dr_S)
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_S, a2dr_S, places=3)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], \
                            normalize=True, title="A2DR Residuals", semilogy=True)

        # Compare residuals
        plt.semilogy(range(drs_result["num_iters"]), drs_result["primal"], color="blue", linestyle="--", label="Primal (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["primal"], color="blue", label="Primal (A2DR)")
        plt.semilogy(range(drs_result["num_iters"]), drs_result["dual"], color="darkorange", linestyle="--", label="Dual (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["dual"], color="darkorange", label="Dual (A2DR) ")
        plt.title("Residuals")
        plt.legend()
        plt.show()

    def test_optimal_control(self):
        m = 2
        n = 5
        T = 20
        K = (T + 1)*(m + n)
        u_bnd = 1   # Upper bound on all |u_t|.
        d_eps = 0.01   # Lower bound on eigenvalues of R.

        # Dynamic matrices.
        A = np.random.randn(n,n)
        A = A/np.max(np.abs(np.linalg.eigvals(A)))   # Scale A so largest eigenvalue has magnitude of one.
        B = np.random.randn(n,m)
        c = np.zeros(n)
        x_init = np.random.randn(n)

        # Cost matrices.
        Q = np.random.randn(n,n)
        Q = Q.T.dot(Q)   # Q is positive semidefinite.
        R = np.random.randn(m,m)
        R = R.T.dot(R)
        U, d, Vt = np.linalg.svd(R)
        d = np.maximum(d, d_eps)
        R = U.dot(np.diag(d)).dot(Vt)   # R is positive definite.

        # TODO: Solve with CVXPY for comparison.

        # Form problem matrices.
        D_row = np.hstack([-A, -B, np.eye(n), np.zeros((n,K-(2*n+m)))])
        D_rows = [np.hstack([np.eye(n), np.zeros((n,K-n))])]
        D_rows += [np.roll(D_row, t*(n+m), axis=1) for t in range(T)]
        D = np.vstack(D_rows)
        e_vec = np.concatenate([x_init] + T*[c])

        # Define proximal operators.
        def prox_norm_inf_wrapper(z, rho):
            z_mat = np.reshape(z, (m+n, T+1), order='F')
            u = z_mat[n:,:].ravel(order='F')
            u_prox = prox_norm_inf(u_bnd)(u, rho)
            u_mat = np.reshape(u_prox, (m,T+1), order='F')
            z_mat[n:,:] = u_mat
            return z_mat.ravel(order='F')
        QR_mats = (T+1)*[Q, R]
        QR_diag = sp.linalg.block_diag(*QR_mats)
        p_list = [prox_quad_form(QR_diag), prox_norm_inf_wrapper]

        # Initialize problem.
        v_init = 2*[np.random.randn(K)]
        A_list = [np.vstack([D, np.eye(K)]), np.vstack([np.zeros((D.shape[0],K)), -np.eye(K)])]
        b = np.concatenate([e_vec, np.zeros(K)])

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_stop=self.eps_stop, anderson=False)
        drs_z = drs_result["x_vals"][-1]
        drs_u_mat = np.reshape(drs_z, (m+n, T+1), order='F')[n:,:]
        if(np.max(np.abs(drs_u_mat)) > u_bnd):
            drs_obj = np.inf
        else:
            drs_obj = drs_z.T.dot(QR_diag).dot(drs_z)
        print("DRS Objective:", drs_obj)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], \
                            normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_stop=self.eps_stop, anderson=True)
        a2dr_z = a2dr_result["x_vals"][-1]
        a2dr_u_mat = np.reshape(a2dr_z, (m+n, T+1), order='F')[n:,:]
        if (np.max(np.abs(a2dr_u_mat)) > u_bnd):
            a2dr_obj = np.inf
        else:
            a2dr_obj = a2dr_z.T.dot(QR_diag).dot(a2dr_z)
        print("A2DR Objective:", a2dr_obj)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], \
                            normalize=True, title="A2DR Residuals", semilogy=True)

        # Compare residuals
        plt.semilogy(range(drs_result["num_iters"]), drs_result["primal"], color="blue", linestyle="--",
                     label="Primal (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["primal"], color="blue", label="Primal (A2DR)")
        plt.semilogy(range(drs_result["num_iters"]), drs_result["dual"], color="darkorange", linestyle="--",
                     label="Dual (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["dual"], color="darkorange", label="Dual (A2DR) ")
        plt.title("Residuals")
        plt.legend()
        plt.show()