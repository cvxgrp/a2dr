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
from scipy.optimize import nnls
from cvxpy import *
from cvxconsensus.solver import a2dr
from cvxconsensus.tests.base_test import BaseTest

def prox_sum_squares(X, y, rcond = None):
    n = X.shape[1]
    def prox(u, rho):
        A = np.vstack((X, np.sqrt(rho/2)*np.eye(n)))
        b = np.concatenate((y, np.sqrt(rho/2)*u))
        return np.linalg.lstsq(A, b, rcond=rcond)[0]
    return prox

class TestSolver(BaseTest):
    """Unit tests for internal A2DR solver."""

    def setUp(self):
        np.random.seed(1)
        self.eps_stop = 1e-8
        self.eps_abs = 1e-16
        self.MAX_ITER = 2000

    def test_nnls(self):
        m = 100
        n = 10
        N = 4   # Number of splits.
        beta_true = np.array(np.arange(-n/2,n/2) + 1)
        X = np.random.randn(m,n)
        y = X.dot(beta_true) + np.random.randn(m)

        # Solve with CVXPY.
        beta = Variable(n)
        obj = sum_squares(X*beta - y)
        constr = [beta >= 0]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver = "OSQP")

        cvxpy_beta = beta.value
        cvxpy_obj = prob.value
        print("CVXPY Objective:", prob.value)
        print("CVXPY Solution:", beta.value)

        # Solve with SciPy.
        sp_result = nnls(X, y)
        print("Scipy Objective:", sp_result[1]**2)
        print("SciPy Solution:", sp_result[0])

        # Split and solve with A2DR.
        X_split = np.split(X,N)
        y_split = np.split(y,N)
        p_list = [prox_sum_squares(X_sub, y_sub) for X_sub, y_sub in zip(X_split, y_split)]
        p_list += [lambda u, rho: np.maximum(u, 0)]   # Projection onto non-negative orthant.
        v_init = (N+1)*[np.random.randn(n)]
        A_list = np.hsplit(np.eye(N*n),N) + [-np.vstack(N*(np.eye(n),))]
        b = np.zeros(N*n)
        result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, anderson=False)

        a2dr_beta = result["x_vals"][-1]
        a2dr_obj = np.sum((X.dot(a2dr_beta) - y)**2)
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)

        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_beta, a2dr_beta, places=3)