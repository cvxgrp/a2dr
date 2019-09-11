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
import scipy as sp
from scipy import sparse

from a2dr import a2dr
from a2dr.proximal import prox_norm1, prox_sum_squares_affine
from a2dr.precondition import precondition
from a2dr.tests.base_test import BaseTest

class TestPrecondition(BaseTest):
    """Unit tests for preconditioning data before S-DRS"""

    def setUp(self):
        np.random.seed(1)
        self.MAX_ITERS = 1000

    def test_precond_l1_trend_filter(self):
        # Problem data.
        N = 2
        n0 = 2*10**4
        n = 2*n0-2
        m = n0-2
        y = np.random.randn(n)
        alpha = 0.1*np.linalg.norm(y, np.inf)

        # Form second difference matrix.
        D = sparse.lil_matrix(sparse.eye(n0))
        D.setdiag(-2, k = 1)
        D.setdiag(1, k = 2)
        D = D[:(n0-2),:]

        # Convert problem to standard form.
        # f_1(x_1) = (1/2)||y - x_1||_2^2, f_2(x_2) = \alpha*||x_2||_1.
        # A_1 = D, A_2 = -I_{n-2}, b = 0.
        prox_list = [lambda v, t: (t*y + v)/(t + 1.0), lambda v, t: prox_norm1(v, t = alpha*t)]
        A_list = [D, -sparse.eye(n0-2)]
        b = np.zeros(n0-2)

        b = np.random.randn(m)
        prox_list = [prox_norm1] * N
        A = sparse.csr_matrix(sparse.hstack(A_list))
        
        p_eq_list, A_eq_list, db, e = precondition(prox_list, A_list, b)
        A_eq = sparse.csr_matrix(sparse.hstack(A_eq_list))
        
        print(r'[Sanity Check]')
        print(r'\|A\|_2 = {}, \|DAE\|_2 = {}'.format(sparse.linalg.norm(A), sparse.linalg.norm(A_eq)))
        print(r'min(|A|) = {}, max(|A|) = {}, mean(|A|) = {}'.format(np.min(np.abs(A)), 
                                                                    np.max(np.abs(A)), sparse.csr_matrix.mean(np.abs(A))))
        print(r'min(|DAE|) = {}, max(|DAE|) = {}, mean(|DAE|) = {}'.format(np.min(np.abs(A_eq)), 
                                                                    np.max(np.abs(A_eq)), sparse.csr_matrix.mean(np.abs(A_eq))))

    def test_nnls(self):
        # Solve the non-negative least squares problem
        # Minimize (1/2)*||A*x - b||_2^2 subject to x >= 0.
        m = 100
        n = 10
        N = 1   # Number of nodes (split A row-wise)

        # Problem data.
        mu = 100
        sigma = 10
        X = mu + sigma*np.random.randn(m,n)
        y = mu + sigma*np.random.randn(m)

        # Solve with SciPy.
        sp_result = sp.optimize.nnls(X, y)
        sp_beta = sp_result[0]
        sp_obj = sp_result[1] ** 2  # SciPy objective is ||y - X\beta||_2.
        print("Scipy Objective:", sp_obj)
        print("SciPy Solution:", sp_beta)

        X_split = np.split(X, N)
        y_split = np.split(y, N)
        p_list = [lambda v, t: prox_sum_squares_affine(v, t, F=X_sub, g=y_sub, method="lsqr") \
                    for X_sub, y_sub in zip(X_split, y_split)]
        p_list += [lambda u, rho: np.maximum(u, 0)]   # Projection onto non-negative orthant.
        A_list = np.hsplit(np.eye(N*n), N) + [-np.vstack(N*(np.eye(n),))]
        b = np.zeros(N*n)

        # Solve with A2DR.
        a2dr_result = a2dr(p_list, A_list, b, anderson=True, precond=False, max_iter=self.MAX_ITERS)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
        print("A2DR Objective:", a2dr_obj)
        print("A2DR Solution:", a2dr_beta)
        self.assertAlmostEqual(sp_obj, a2dr_obj)
        self.assertItemsAlmostEqual(sp_beta, a2dr_beta, places=3)

        # Solve with preconditioned A2DR.
        cond_result = a2dr(p_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITERS)
        cond_beta = cond_result["x_vals"][-1]
        cond_obj = np.sum((y - X.dot(cond_beta))**2)
        print("Preconditioned A2DR Objective:", cond_obj)
        print("Preconditioned A2DR Solution:", cond_beta)
        self.assertAlmostEqual(sp_obj, cond_obj)
        self.assertItemsAlmostEqual(sp_beta, cond_beta, places=3)
