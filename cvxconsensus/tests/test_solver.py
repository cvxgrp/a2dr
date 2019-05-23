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
from scipy import sparse
from scipy.optimize import nnls
from cvxconsensus import a2dr
from cvxconsensus.proximal.prox_operators import prox_logistic
from cvxconsensus.tests.base_test import BaseTest

def prox_sum_squares(X, y, type = "lsqr"):
    n = X.shape[1]
    if type == "lsqr":
        X = sparse.csc_matrix(X)
        def prox(v, rho):
            A = sparse.vstack((X, np.sqrt(rho/2)*sparse.eye(n)))
            b = np.concatenate((y, np.sqrt(rho/2)*v))
            return sparse.linalg.lsqr(A, b, atol=1e-16, btol=1e-16)[0]
    elif type == "lstsq":
        def prox(v, rho):
           A = np.vstack((X, np.sqrt(rho/2)*np.eye(n)))
           b = np.concatenate((y, np.sqrt(rho/2)*v))
           return LA.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError("Algorithm type not supported:", type)
    return prox

def prox_neg_log_det(A, rho):
    A_symm = (A + A.T) / 2.0
    if not np.allclose(A, A_symm):
        raise Exception("Proximal operator for negative log-determinant only operates on symmetric matrices.")
    w, v = LA.eig(A_symm)
    w_new = (w + np.sqrt(w**2 + 4.0/rho))/2
    A_new = v.dot(np.diag(w_new)).dot(v.T)
    return A_new.ravel(order='C')

def prox_pos_semidef(A, rho):
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

def prox_multi_logistic(y, x0 = None):
    n = y.shape[0]
    if not x0:
        x0 = np.random.randn(n)
    return lambda v, rho: np.array([prox_logistic(v[i], rho, x0[i], y[i]) for i in range(n)])

    # def prox(v, rho):
    #     fun = lambda x: np.sum(-np.log(sp.special.expit(np.multiply(y,x)))) + (rho/2.0)*np.sum((x-v)**2)
    #     jac = lambda x: -y*sp.special.expit(-np.multiply(y,x)) + rho*(x-v)
    #     res = sp.optimize.minimize(fun, x0, method='L-BFGS-B', jac=jac)
    #     return res.x
    # return prox

    # z = Variable(y.shape[0])
    # rho_parm = Parameter(nonneg=True)
    # v_parm = Parameter(y.shape[0])
    # loss = sum(logistic(-multiply(y, z)))
    # reg = (rho_parm/2)*sum_squares(z - v_parm)
    # prob = Problem(Minimize(loss + reg))
    #
    # def prox(v, rho):
    #     rho_parm.value = rho
    #     v_parm.value = v
    #     prob.solve()
    #     return z.value
    # return prox

def prox_norm1(lam = 1.0):
    return lambda v, rho: np.maximum(v-lam/rho,0) - np.maximum(-v-lam/rho,0)

def prox_norm2(lam = 1.0):
    return lambda u, rho: np.maximum(1 - 1.0/(lam/rho * LA.norm(u, 2)), 0) * u

def prox_norm_inf(bound):
    if bound < 0:
        raise ValueError("bound must be a non-negative scalar.")
    return lambda v, rho: np.maximum(np.minimum(v, bound), -bound)

def prox_nuc_norm(lam = 1.0):
    def prox(A, rho):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_new = np.maximum(s - lam/rho, 0)
        A_new = U.dot(np.diag(s_new)).dot(Vt)
        return A_new.ravel(order='C')
    return prox

def prox_group_lasso(lam = 1.0):
    prox_inner = prox_norm2(lam)
    return lambda A, rho: np.concatenate([prox_inner(A[:,j], rho) for j in range(A.shape[1])])

class TestSolver(BaseTest):
    """Unit tests for internal A2DR solver."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8
        self.eps_abs = 1e-6
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
            np_result = LA.lstsq(X_split[i], y_split[i], rcond=None)
            np_beta += [np_result[0]]
            np_obj += np.sum(np_result[1])
        print("NumPy Objective:", np_obj)
        print("NumPy Solution:", np_beta)

        # Solve with DRS (proximal point method).
        drs_result = a2dr(p_list, v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                          eps_rel=self.eps_rel, anderson=False)
        drs_beta = drs_result["x_vals"]
        drs_obj = np.sum([(yi - Xi.dot(beta))**2 for yi,Xi,beta in zip(y_split,X_split,drs_beta)])
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)

        # Solve with A2DR (proximal point method with Anderson acceleration).
        a2dr_result = a2dr(p_list, v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                           eps_rel=self.eps_rel, anderson=True)
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
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                          eps_rel=self.eps_rel, anderson=False)
        drs_beta = drs_result["x_vals"][-1]
        drs_obj = np.sum((y - X.dot(drs_beta))**2)
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)
        self.assertAlmostEqual(sp_obj, drs_obj)
        self.assertItemsAlmostEqual(sp_beta, drs_beta, places=3)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)
        self.assertAlmostEqual(sp_obj, a2dr_obj)
        self.assertItemsAlmostEqual(sp_beta, a2dr_beta, places=3)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_l1_trend_filtering(self):
        # minimize (1/2)||y - x||_2^2 + lam*||Dx||_1,
        # where (Dx)_{t-1} = x_{t-1} - 2*x_t + x_{t+1} for t = 2,...,n-1.
        # Reference: https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

        # Problem data.
        n = 100
        lam = 1.0
        y = np.random.randn(n)

        # Solve with CVXPY.
        x = Variable(n)
        obj = sum_squares(y - x)/2 + lam*norm1(diff(x,2))
        prob = Problem(Minimize(obj))
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_x = x.value
        print("CVXPY Objective:", cvxpy_obj)
        print("CVXPY Solution:", cvxpy_x)

        # Split problem as f_1(x_1) = (1/2)||y - x_1||_2^2, f_2(x_2) = lam*||x_2||_1,
        # subject to the constraint D*x_1 = x_2.
        arr = np.concatenate([np.array([1,-2,1]), np.zeros(n-3)])
        arrs = [np.roll(arr, i) for i in range(n-2)]
        D = np.vstack(arrs)

        p_list = [lambda v, rho: (y + rho*v)/(1.0 + rho), prox_norm1(lam)]
        v_init = [np.zeros(n), np.zeros(n-2)]
        A_list = [D, -np.eye(n-2)]
        b = np.zeros(n-2)

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                          eps_rel=self.eps_rel, anderson=False)
        drs_x = drs_result["x_vals"][0]
        drs_obj = np.sum((y - drs_x)**2)/2 + lam*np.sum(np.abs(np.diff(drs_x,2)))
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_x)
        self.assertAlmostEqual(cvxpy_obj, drs_obj)
        self.assertItemsAlmostEqual(cvxpy_x, drs_x)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_x = a2dr_result["x_vals"][0]
        a2dr_obj = np.sum((y - a2dr_x)**2)/2 + lam*np.sum(np.abs(np.diff(a2dr_x,2)))
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_x)
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_x, a2dr_x)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_multi_task_logistic(self):
        # minimize \sum_i log(1 + exp(-y_i*z_i)) + lam_1*||\theta||_{2,1} + \lam_2*||\theta||_*
        #    subject to Z = X\theta, ||.||_{2,1} = group lasso, ||.||_* = nuclear norm.

        # Problem data.
        K = 5    # Number of tasks.
        n = 20    # Number of features.
        m = 100   # Number of samples.
        X = np.random.randn(m,n)
        theta_true = np.random.randn(n,K)
        Z_true = X.dot(theta_true)
        Y = 2*(Z_true > 0) - 1   # y = 1 or -1.

        def calc_obj(theta, lam):
            obj = np.sum(-np.log(sp.special.expit(np.multiply(Y, X.dot(theta)))))
            reg = lam[0]*np.sum([LA.norm(theta[:,k], 2) for k in range(K)])
            reg += lam[1]*LA.norm(theta, ord='nuc')
            return obj + reg

        def prox_logistic_wrapper(y):
            def prox(v, rho):
                z_prox = prox_multi_logistic(y)(v[:m], rho)
                return np.concatenate([z_prox, v[m:]])
            return prox

        # Solve with CVXPY.
        theta = Variable((n,K))
        lam = Parameter(2, nonneg=True)
        loss = sum(logistic(-multiply(Y, X*theta)))
        reg = lam[0]*sum(norm(theta, 2, axis=0)) + lam[1]*normNuc(theta)
        prob = Problem(Minimize(loss + reg))

        lam.value = [1.0, 1.0]
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_theta = theta.value
        print("CVXPY Objective:", cvxpy_obj)

        # Split problem as f_k(z_k) = \sum_i log(1 + exp(-Y_{ik}*Z_{ik})) for k = 1,...,K
        # f_{K+1}(\theta) = lam1*||\theta||_{2,1}, f_{K+2}(\theta) = lam2*||\theta||_*.
        M = m*K + 2*n*K
        p_list = [prox_logistic_wrapper(Y[:,k]) for k in range(K)]
        p_list += [lambda v, rho: prox_group_lasso(lam[0].value)(np.reshape(v, (n,K), order='C'), rho),
                   lambda v, rho: prox_nuc_norm(lam[1].value)(np.reshape(v, (n,K), order='C'), rho)]
        v_init = K*[np.zeros(m+n)] + 2*[np.random.randn(n*K)]

        # Form constraint matrices.
        E1_mat = np.zeros((m*K,m+n))
        E1_mat[:m,:] = np.hstack([-np.eye(m), X])
        E2_mat = np.zeros((n*K,m+n))
        E2_mat[:n,m:] = np.eye(n)
        E_mats = []
        for k in range(K):
            E1 = np.roll(E1_mat, k*m, axis=0)
            E2 = np.roll(E2_mat, k*n, axis=0)
            E = np.vstack([E1, E2, np.zeros((n*K,m+n))])
            E_mats.append(E)

        T1_mat = np.vstack([np.zeros((m*K,n*K)), -np.eye(n*K), np.eye(n*K)])
        T2_mat = np.vstack([np.zeros(((m+n)*K,n*K)), np.eye(n*K)])
        A_list = E_mats + [T1_mat, T2_mat]
        b = np.zeros(M)

        # Solve transformed problem with CVXPY.
        # zt_list = [Variable(m+n) for k in range(K)]
        # theta1 = Variable(n*K)
        # theta2 = Variable(n*K)
        #
        # loss = 0
        # Ax = T1_mat*theta1 + T2_mat*theta2
        # for k in range(K):
        #     Ax += E_mats[k]*zt_list[k]
        #     loss += sum(logistic(-multiply(Y[:,k], zt_list[k][:m])))
        # reg1 = lam[0]*sum(norm(reshape(theta1, (n,K)), 2, axis=0))
        # reg2 = lam[1]*normNuc(reshape(theta2, (n,K)))
        # prob = Problem(Minimize(loss + reg1 + reg2), [Ax == b])
        # prob.solve()
        # print("CVXPY Transformed Objective:", prob.value)
        # self.assertAlmostEqual(prob.value, cvxpy_obj, places=3)

        # Solve with DRS.
        # TODO: Proximal operator for logistic function is failing.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_rel=self.eps_rel, anderson=False)
        drs_theta = drs_result["x_vals"][-1].reshape((n, K), order='C')
        drs_obj = calc_obj(drs_theta, lam.value)
        print("DRS Objective:", drs_obj)
        # self.assertAlmostEqual(cvxpy_obj, drs_obj)
        # self.assertItemsAlmostEqual(cvxpy_theta, drs_theta)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_theta = a2dr_result["x_vals"][-1].reshape((n, K), order='C')
        a2dr_obj = calc_obj(a2dr_theta, lam.value)
        print("DRS Objective:", a2dr_obj)
        # self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        # self.assertItemsAlmostEqual(cvxpy_theta, a2dr_theta)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_sparse_covariance(self):
        # minimize -log(det(S)) + trace(S*Y) + \alpha*||S||_1
        #   subject to S is PSD, where Y and \alpha >= are parameters.

        # Problem data.
        m = 10    # Dimension of matrix.
        n = m*m   # Length of vectorized matrix.
        K = 1000  # Number of samples.
        A = np.random.randn(m,m)
        A[sp.sparse.rand(m,m,0.85).todense().nonzero()] = 0
        S_true = A.dot(A.T) + 0.05*np.eye(m)
        R = LA.inv(S_true)
        y_sample = sp.linalg.sqrtm(R).dot(np.random.randn(m,K))
        Y = np.cov(y_sample)

        # Solve with CVXPY.
        S = Variable((m,m), PSD=True)
        alpha = Parameter(nonneg=True)
        obj = -log_det(S) + trace(S*Y) + alpha*norm1(S)
        prob = Problem(Minimize(obj))

        alpha.value = 1.0
        prob.solve(eps = self.eps_abs)
        cvxpy_obj = prob.value
        cvxpy_S = S.value
        print("CVXPY Objective:", cvxpy_obj)

        # Split problem as f_1(S) = -log(det(S)), f_2(S) = trace(S*Y),
        #   f_3(S) = \alpha*||S||_1, f_4(S) = I(S is symmetric PSD).
        # TODO: Due to computational error, need to relax/disable symmetry check in prox for f_1 and f_4.
        N = 3
        p_list = [lambda v, rho: prox_neg_log_det(np.reshape(v, (m,m), order='C'), rho),
                  lambda v, rho: v - Y.ravel(order='C')/rho,
                  lambda v, rho: np.maximum(np.abs(v) - alpha.value/rho, 0) * np.sign(v),
                  lambda v, rho: prox_pos_semidef(np.reshape(v, (m,m), order='C'), rho)]
        S_init = np.random.randn(m,m)
        S_init = S_init.T.dot(S_init)   # Ensure starting point is symmetric PSD.
        v_init = (N + 1)*[S_init.ravel(order='C')]
        A_list = np.hsplit(np.eye(N*n),N) + [-np.vstack(N*(np.eye(n),))]
        b = np.zeros(N*n)

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_rel=self.eps_rel, anderson=False)
        drs_S = drs_result["x_vals"][-1].reshape((m,m), order='C')
        drs_obj = -LA.slogdet(drs_S)[1] + np.sum(np.diag(drs_S.dot(Y))) + alpha.value*np.sum(np.abs(drs_S))
        print("DRS Objective:", drs_obj)
        self.assertAlmostEqual(cvxpy_obj, drs_obj, places=3)
        self.assertItemsAlmostEqual(cvxpy_S, drs_S, places=3)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_S = a2dr_result["x_vals"][-1].reshape((m,m), order='C')
        a2dr_obj = -LA.slogdet(a2dr_S)[1] + np.sum(np.diag(a2dr_S.dot(Y))) + alpha.value*np.sum(np.abs(a2dr_S))
        print("A2DR Objective:", a2dr_obj)
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_S, a2dr_S)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_single_commodity_flow(self):
        # Problem data.
        m = 25    # Number of sources.
        n = 100   # Number of flows.
        A = np.random.randint(-1,2,(m,n))

        # Flow cost = \sum_j \psi*(x_j - x0)^2 for 0 <= x_j <= x_max.
        psi = 0.5
        x0 = 0.5
        x_max = 1

        # Source costs.
        # Generator cost = \sum_i \phi*s_i^2 for 0 <= s_i <= s_max.
        m_free = 10  # Number of generators free to set.
        m_off = 5    # Number of generators that are off (s_i = 0).
        m_pin = 5    # Number of generators that are pinned to max (s_i = s_max).
        m_gen = m_off + m_pin + m_free
        phi = 0.5
        s_max = 1

        # Load cost = I(s_i = load_i) where load_i < 0.
        m_load = m - m_gen   # Number of loads.
        load = np.full(m_load, -1)

        # Solve with CVXPY
        x = Variable(n)
        s_free = Variable(m_free)
        s_off = Variable(m_off)
        s_pin = Variable(m_pin)
        s_load = Variable(m_load)
        s = vec(vstack([reshape(s_free, (m_free,1)), reshape(s_off, (m_off,1)), \
                        reshape(s_pin, (m_pin,1)), reshape(s_load, (m_load,1))]))
        obj = phi*sum_squares(s_free) + psi*sum_squares(x - x0)
        constr = [A*x + s == 0, sum(s) == 0, x >= 0, x <= x_max, s_free >= 0, s_free <= s_max,
                  s_off == 0, s_pin == s_max, s_load == load]
        prob = Problem(Minimize(obj), constr)
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_x = x.value
        cvxpy_s = s.value
        print("CVXPY Objective:", cvxpy_obj)

        # Initialize problem.
        p_list = [lambda x,rho: (2*psi*x0 + rho*x)/(2*psi + rho),
                  lambda s,rho: np.concatenate([rho*s[:m_free]/(2*phi + rho), s[m_free:]]),
                  lambda x,rho: np.minimum(np.maximum(x,0), x_max),
                  lambda s,rho: np.concatenate([np.minimum(np.maximum(s[:m_free],0), s_max), s[m_free:]])]
        v_init = 2*[np.zeros(n), np.zeros(m)]
        D = np.vstack([np.hstack([A, np.eye(m), np.zeros((m,m+n))]),
                       np.hstack([np.zeros((1,n)), np.ones((1,m)), np.zeros((1,m+n))]),
                       np.hstack([np.zeros((m_off,n + m_free)), np.eye(m_off), np.zeros((m_off, m_pin + m_load + m+n))]),
                       np.hstack([np.zeros((m_pin,n + m_free + m_off)), np.eye(m_pin), np.zeros((m_pin, m_load + m+n))]),
                       np.hstack([np.zeros((m_load,n + m_free + m_off + m_pin)), np.eye(m_load), np.zeros((m_load, m+n))]),
                       np.hstack([np.eye(n), np.zeros((n,m)), -np.eye(n), np.zeros((n,m))]),
                       np.hstack([np.zeros((m,n)), np.eye(m), np.zeros((m,n)), -np.eye(m)])
                      ])
        A_list = [D[:,:n], D[:,n:(n+m)], D[:,(n+m):(2*n+m)], D[:,(2*n+m):(2*n+2*m)]]
        b = np.concatenate([np.zeros(m+1), np.zeros(m_off), np.full(m_pin, s_max), load, np.zeros(m+n)])

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_rel=self.eps_rel, anderson=False)
        drs_x = drs_result["x_vals"][0]
        drs_s = drs_result["x_vals"][1]
        drs_obj = phi*np.sum(drs_s[:m_free]**2) + psi*np.sum((drs_x - x0)**2)
        print("DRS Objective:", drs_obj)
        self.assertAlmostEqual(cvxpy_obj, drs_obj)
        self.assertItemsAlmostEqual(cvxpy_x, drs_x)
        self.assertItemsAlmostEqual(cvxpy_s, drs_s)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], \
                            normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_x = a2dr_result["x_vals"][0]
        a2dr_s = a2dr_result["x_vals"][1]
        a2dr_obj = phi*np.sum(a2dr_s[:m_free] ** 2) + psi*np.sum((a2dr_x - x0) ** 2)
        print("A2DR Objective:", a2dr_obj)
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_x, a2dr_x)
        self.assertItemsAlmostEqual(cvxpy_s, a2dr_s)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], \
                            normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_optimal_control(self):
        m = 2
        n = 5
        T = 20
        K = (T + 1)*(m + n)
        u_bnd = 1      # Upper bound on all |u_t|.
        w_eps = 0.01   # Lower bound on eigenvalues of R.
        A_eps = 0.05

        # Dynamic matrices.
        # TODO: Generate problem so max(|u_t|) hits the upper bound!
        A = np.eye(n) + A_eps*np.random.randn(n,n)
        A = A/np.max(np.abs(LA.eigvals(A)))   # Scale A so largest eigenvalue has magnitude of one.
        B = np.random.randn(n,m)
        c = np.zeros(n)
        x_init = np.random.randn(n)

        # Cost matrices.
        Q = np.random.randn(n,n)
        Q = Q.T.dot(Q)   # Q is positive semidefinite.
        R = np.random.randn(m,m)
        R = R.T.dot(R)
        w, v = LA.eig(R)
        w = np.maximum(w, w_eps)
        R = v.dot(np.diag(w)).dot(v.T)   # R is positive definite.
        QR_mats = (T+1)*[Q, R]
        QR_diag = sp.linalg.block_diag(*QR_mats)   # Quadratic form cost = z^T*QR_diag*z

        # Helper function for extracting (x_0,...,x_T) and (u_0,...,u_T) from z = (x_0, u_0, ..., x_T, u_T).
        def extract_xu(z):
            z_mat = np.reshape(z, (T+1, m+n), order='C')
            x = z_mat[:,:n].ravel(order='C')
            u = z_mat[:,n:].ravel(order='C')
            return x, u

        # Calculate objective directly from z = (x_0, u_0, ..., x_T, u_T).
        def calc_obj(z):
            x, u = extract_xu(z)
            u_inf = np.max(np.abs(u))
            return z.T.dot(QR_diag).dot(z) if u_inf <= u_bnd else np.inf

        # Define proximal operators.
        def prox_norm_inf_wrapper(z, rho):
            z_mat = np.reshape(z, (T+1, m+n), order='C')
            u = z_mat[:,n:].ravel(order='C')
            u_prox = prox_norm_inf(u_bnd)(u, rho)
            u_mat = np.reshape(u_prox, (T+1, m), order='C')
            z_mat[:,n:] = u_mat
            return z_mat.ravel(order='C')

        # Solve with CVXPY.
        x = Variable((T+1,n))
        u = Variable((T+1,m))
        obj = sum([quad_form(x[t], Q) + quad_form(u[t], R) for t in range(T+1)])
        constr = [x[0] == x_init, norm_inf(u) <= u_bnd]
        constr += [x[t+1] == A*x[t] + B*u[t] + c for t in range(T)]
        prob = Problem(Minimize(obj), constr)
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_x = x.value.ravel(order='C')
        cvxpy_u = u.value.ravel(order='C')
        print("CVXPY Objective:", cvxpy_obj)

        # Form problem matrices.
        D_row = np.hstack([-A, -B, np.eye(n), np.zeros((n,K-(2*n+m)))])
        D_rows = [np.hstack([np.eye(n), np.zeros((n,K-n))])]
        D_rows += [np.roll(D_row, t*(n+m), axis=1) for t in range(T)]
        D = np.vstack(D_rows)
        e_vec = np.concatenate([x_init] + T*[c])

        # Initialize problem.
        p_list = [prox_quad_form(QR_diag), prox_norm_inf_wrapper]
        v_init = 2*[np.random.randn(K)]
        A_list = [np.vstack([D, np.eye(K)]), np.vstack([np.zeros((D.shape[0],K)), -np.eye(K)])]
        b = np.concatenate([e_vec, np.zeros(K)])

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_rel=self.eps_rel, anderson=False)
        drs_z = drs_result["x_vals"][-1]
        drs_x, drs_u = extract_xu(drs_z)
        drs_obj = calc_obj(drs_z)
        print("DRS Objective:", drs_obj)
        self.assertAlmostEqual(cvxpy_obj, drs_obj)
        self.assertItemsAlmostEqual(cvxpy_x, drs_x)
        self.assertItemsAlmostEqual(cvxpy_u, drs_u)
        self.plot_residuals(drs_result["primal"], drs_result["dual"], normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_z = a2dr_result["x_vals"][-1]
        a2dr_x, a2dr_u = extract_xu(a2dr_z)
        a2dr_obj = calc_obj(a2dr_z)
        print("A2DR Objective:", a2dr_obj)
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_x, a2dr_x)
        self.assertItemsAlmostEqual(cvxpy_u, a2dr_u)
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_multi_optimal_control(self):
        S = 2    # Number of scenarios.
        m = 2    # Dimension of x_t^(s).
        n = 4    # Dimension of u_t^(s).
        T = 10   # Time periods (t = 0,...,T)
        K = (T + 1)*(m + n)
        w_eps = 0.01   # Lower bound on eigenvalues of R.
        A_eps = 0.05

        # Scenario matrices.
        weights = np.random.uniform(size=S)
        weights = weights / np.sum(weights)
        u_bnds = 1 + 0.01*np.random.randn(S)   # Upper bound on all |u_t|.

        # Dynamic matrices.
        A = np.eye(n) + A_eps * np.random.randn(n, n)
        A = A / np.max(np.abs(LA.eigvals(A)))  # Scale A so largest eigenvalue has magnitude of one.
        B = np.random.randn(n, m)
        c = np.zeros(n)
        x_init = np.random.randn(n)

        # Cost matrices.
        Q_list = []
        R_list = []
        for s in range(S):
            Q = np.random.randn(n,n)
            Q = Q.T.dot(Q)   # Q is positive semidefinite.
            Q_list.append(Q)

            R = np.random.randn(m,m)
            R = R.T.dot(R)
            w, v = LA.eig(R)
            w = np.maximum(w, w_eps)
            R = v.dot(np.diag(w)).dot(v.T)   # R is positive definite.
            R_list.append(R)

        # Helper function for extracting (x_0^(s),...,x_T^(s)) and (u_0^(s),...,u_T^(s)).
        def extract_xu(z):
            x = np.reshape(z[:(T+1)*n], (T+1,n), order='C')
            u = np.reshape(z[(T+1)*n:], (T+1,m), order='C')
            return x, u

        # Calculate objective function and check constraints are satisfied.
        def calc_obj(x_list, u_list):
            obj = 0
            for s in range(S):
                xQx = [x_list[s][t].dot(Q_list[s]).dot(x_list[s][t].T) for t in range(T+1)]
                uRu = [u_list[s][t].dot(R_list[s]).dot(u_list[s][t].T) for t in range(T+1)]
                u_inf = np.max(np.abs(u_list[s]))

                # Check x_0 initialized properly and u_t within bound.
                # if LA.norm(x_list[s][0] - x_init) > self.eps_abs + self.eps_rel*LA.norm(x_init) or \
                #       u_inf - u_bnds[s] > self.eps_abs + self.eps_rel*u_bnds[s]:
                #   return np.inf
                obj += weights[s]*np.sum(xQx + uRu)

            # Check u_0^(s) equal across scenarios.
            u0_mat = np.column_stack([u_list[s][0] for s in range(S)])
            u0_norms = LA.norm(u0_mat, 2, axis=0)
            u0_errs = LA.norm(np.diff(u0_mat, 1, axis=1), 2, axis=0)
            # if not np.all(u0_errs < self.eps_abs + self.eps_rel*u0_norms[:-1]):
            #    return np.inf
            return obj

        # Proximal operator using OSQP.
        def prox_control_osqp(Q, R, A, B, c, u_bnd, weight):
            x = Variable((T+1,n))
            u = Variable((T+1,m))
            rho_parm = Parameter(nonneg=True)
            x_parm = Parameter((T+1,n))
            u_parm = Parameter((T+1,m))

            obj = sum([weight*(quad_form(x[t], Q) + quad_form(u[t], R)) for t in range(T+1)])
            reg = (rho_parm/2)*(sum_squares(x - x_parm) + sum_squares(u - u_parm))
            constr = [x[0] == x_init, norm_inf(u) <= u_bnd]
            constr += [x[t+1] == A*x[t] + B*u[t] + c for t in range(T)]
            prob = Problem(Minimize(obj + reg), constr)

            def prox(v, rho):
                rho_parm.value = rho
                x_parm.value, u_parm.value = extract_xu(v)
                prob.solve(solver='OSQP', eps_abs=self.eps_abs)
                x_val = x.value.ravel(order='C')
                u_val = u.value.ravel(order='C')
                return np.concatenate([x_val, u_val])
            return prox

        # Solve with CVXPY.
        x_list = [Variable((T+1,n)) for s in range(S)]
        u_list = [Variable((T+1,m)) for s in range(S)]
        obj = 0
        constr = []
        for s in range(S):
            obj += sum([weights[s]*(quad_form(x_list[s][t],Q_list[s]) + quad_form(u_list[s][t],R_list[s])) for t in range(T+1)])
            constr += [x_list[s][0] == x_init, norm_inf(u_list[s]) <= u_bnds[s]]
            constr += [x_list[s][t+1] == A*x_list[s][t] + B*u_list[s][t] + c for t in range(T)]
        constr += [u_list[s+1][0] == u_list[s][0] for s in range(S-1)]
        prob = Problem(Minimize(obj), constr)
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_x_list = [x.value for x in x_list]
        cvxpy_u_list = [u.value for u in u_list]
        print("CVXPY Objective:", cvxpy_obj)

        # Initialize problem.
        p_list = [prox_control_osqp(Q,R,A,B,c,u_bnd,weight) for Q,R,u_bnd,weight in zip(Q_list,R_list,u_bnds,weights)]
        p_list += [lambda v, rho: v]
        v_init = S*[np.zeros(K)] + [np.zeros(m)]

        # Consensus constraint on u_0^(1),...,u_0^(S).
        E_mat = np.hstack([np.zeros((m,(T+1)*n)), np.eye(m), np.zeros((m,T*m))])
        E_mat = np.vstack([E_mat, np.zeros(((S-1)*m,K))])
        E_mats = [np.roll(E_mat, s*m, axis=0) for s in range(S)]
        I_mat = -np.vstack(S*[np.eye(m)])
        A_list = E_mats + [I_mat]
        b = np.zeros(S*m)

        # Solve transformed problem with CVXPY.
        # x_vecs = [Variable((T+1)*n) for s in range(S)]
        # u_vecs = [Variable((T+1)*m) for s in range(S)]
        # u0_all = Variable(m)
        # obj = Ax = 0
        # constr = []
        # for s in range(S):
        #     x_mat = reshape(x_vecs[s], (n,T+1)).T
        #     u_mat = reshape(u_vecs[s], (m,T+1)).T
        #     obj += weights[s]*sum([(quad_form(x_mat[t], Q_list[s]) + quad_form(u_mat[t], R_list[s])) for t in range(T+1)])
        #     constr += [x_mat[0] == x_init, norm_inf(u_vecs[s]) <= u_bnds[s]]
        #     constr += [x_mat[t+1] == A*x_mat[t] + B*u_mat[t] + c for t in range(T)]
        #     Ax += A_list[s]*hstack([x_vecs[s], u_vecs[s]])
        # Ax += A_list[S]*u0_all
        # prob = Problem(Minimize(obj), constr + [Ax == b])
        # prob.solve()
        # print("CVXPY Transformed Objective:", prob.value)
        # self.assertAlmostEqual(prob.value, cvxpy_obj)

        # Solve with DRS.
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                          eps_rel=self.eps_rel, anderson=False)
        drs_z_list = [extract_xu(z) for z in drs_result["x_vals"][:S]]
        drs_x_list, drs_u_list = map(list, zip(*drs_z_list))
        drs_obj = calc_obj(drs_x_list, drs_u_list)
        print("DRS Objective:", drs_obj)
        # self.assertAlmostEqual(cvxpy_obj, drs_obj)
        # for s in range(S):
        #    self.assertItemsAlmostEqual(cvxpy_x_list[s], drs_x_list[s])
        #    self.assertItemsAlmostEqual(cvxpy_u_list[s], drs_u_list[s])
        self.plot_residuals(drs_result["primal"], drs_result["dual"], \
                            normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs,
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_z_list = [extract_xu(z) for z in a2dr_result["x_vals"][:S]]
        a2dr_x_list, a2dr_u_list = map(list, zip(*a2dr_z_list))
        a2dr_obj = calc_obj(a2dr_x_list, a2dr_u_list)
        print("A2DR Objective:", a2dr_obj)
        # self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        # for s in range(S):
        #    self.assertItemsAlmostEqual(cvxpy_x_list[s], a2dr_x_list[s])
        #    self.assertItemsAlmostEqual(cvxpy_u_list[s], a2dr_u_list[s])
        self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], \
                            normalize=True, title="A2DR Residuals", semilogy=True)
        self.compare_primal_dual(drs_result, a2dr_result)