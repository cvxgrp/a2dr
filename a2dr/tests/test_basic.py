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

import numpy as np
import numpy.linalg as LA
from scipy import sparse

from a2dr import a2dr
from a2dr.proximal import prox_sum_squares_affine, prox_nonneg_constr
from a2dr.tests.base_test import BaseTest

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
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F=X, g=y, method="lstsq")]

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

        self.assertAlmostEqual(np_obj, drs_obj)
        self.assertAlmostEqual(np_obj, a2dr_obj)

    def test_ols(self):
        # minimize ||y - X\beta||_2^2.
        m = 100
        n = 10
        N = 4  # Number of splits. (split X row-wise)
        beta_true = np.array(np.arange(-n / 2, n / 2) + 1)
        X = np.random.randn(m, n)
        y = X.dot(beta_true) + np.random.randn(m)

        # Split problem.
        X_split = np.split(X, N)
        y_split = np.split(y, N)

        # Construct list of proximal operators.
        # Note: We must do it this way to avoid problems caused by late binding:
        # https://docs.python-guide.org/writing/gotchas/#late-binding-closures
        prox_list = [lambda v, t, i=i: prox_sum_squares_affine(v, t, F=X_split[i], g=y_split[i], method="lstsq") \
                        for i in range(N)]
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
        drs_result = a2dr(prox_list, v_init=v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                          eps_rel=self.eps_rel, anderson=False)
        drs_beta = drs_result["x_vals"]
        drs_obj = np.sum([(yi - Xi.dot(beta)) ** 2 for yi, Xi, beta in zip(y_split, X_split, drs_beta)])
        print("DRS Objective:", drs_obj)
        print("DRS Solution:", drs_beta)

        # Solve with A2DR (proximal point method with Anderson acceleration).
        a2dr_result = a2dr(prox_list, v_init=v_init, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
                           eps_rel=self.eps_rel, anderson=True)
        a2dr_beta = a2dr_result["x_vals"]
        a2dr_obj = np.sum([(yi - Xi.dot(beta)) ** 2 for yi, Xi, beta in zip(y_split, X_split, drs_beta)])
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)

        # Compare results.
        self.assertAlmostEqual(np_obj, drs_obj)
        self.assertAlmostEqual(np_obj, a2dr_obj)
        for i in range(N):
            self.assertItemsAlmostEqual(np_beta[i], drs_beta[i])
            self.assertItemsAlmostEqual(np_beta[i], a2dr_beta[i])

    def test_infeas(self):
        # a modified non-negative least squares example with infeasible linear constraints
        m, n = 150, 300
        density = 0.001
        X = sparse.random(m, n, density=density, data_rvs=np.random.randn)
        y = np.random.randn(m)

        # Convert problem to standard form.
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F=X, g=y), 
                     prox_nonneg_constr]
        Z = sparse.eye(n);
        e1 = sparse.lil_matrix((1,n))
        e1[0,0] = 1 
        A1 = sparse.bmat([[Z], [e1]])
        A2 = sparse.bmat([[-Z], [-e1]])
        A_list = [A1, A2]
        b = np.zeros(n+1)
        b[0] = 1
        b[-1] = -1

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False)
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True)
        self.assertTrue(drs_result == 
            {"x_vals": None, "primal": None, "dual": None, "num_iters": None, "solve_time": None} 
            and a2dr_result == 
            {"x_vals": None, "primal": None, "dual": None, "num_iters": None, "solve_time": None})
