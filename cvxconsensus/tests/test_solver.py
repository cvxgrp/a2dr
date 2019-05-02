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
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from cvxconsensus.solver import a2dr
from cvxconsensus.tests.base_test import BaseTest

def prox_sum_squares(X, y, rcond = None):
    n = X.shape[1]
    def prox(v, rho):
        A = np.vstack((X, np.sqrt(rho/2)*np.eye(n)))
        b = np.concatenate((y, np.sqrt(rho/2)*v))
        return np.linalg.lstsq(A, b, rcond=rcond)[0]
    return prox

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
        drs_result = a2dr(p_list, v_init, max_iter=self.MAX_ITER, anderson=False)
        drs_beta = drs_result["x_vals"]
        drs_obj = np.sum([(yi - Xi.dot(beta))**2 for yi,Xi,beta in zip(y_split,X_split,drs_beta)])
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)

        # Compare results.
        self.assertAlmostEqual(np_obj, drs_obj)
        for i in range(N):
            self.assertItemsAlmostEqual(np_beta[i], drs_beta[i])

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
        drs_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, anderson=False)
        drs_beta = drs_result["x_vals"][-1]
        drs_obj = np.sum((y - X.dot(drs_beta))**2)
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)
        self.assertAlmostEqual(sp_obj, drs_obj)
        self.assertItemsAlmostEqual(sp_beta, drs_beta, places=3)
        # self.plot_residuals(drs_result["primal"], drs_result["dual"], \
        #                    normalize=True, title="DRS Residuals", semilogy=True)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, anderson=True)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)
        # self.assertAlmostEqual(sp_obj, a2dr_obj)
        # self.assertItemsAlmostEqual(sp_beta, a2dr_beta, places=3)
        # self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], \
        #                    normalize=True, title="A2DR Residuals", semilogy=True)