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
from cvxpy import *
from scipy import sparse

from a2dr import a2dr
from a2dr.proximal import *
from a2dr.tests.base_test import BaseTest

class TestOther(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8   # Specify these in all examples?
        self.eps_abs = 1e-6
        self.MAX_ITER = 2000

    def test_simple_logistic(self):
        # minimize \sum_j log(1 + exp(-y_j*Z_j)) subject to Z = X\beta with variables (Z,\beta).
        # Problem data.
        m = 100
        p = 80
        X = np.random.randn(m, p)
        beta_true = np.random.randn(p)
        Z_true = X.dot(beta_true)
        y = 2 * (Z_true > 0) - 1   # y_i = 1 or -1.

        # Solve with CVXPY.
        beta = Variable(p)
        obj = sum(logistic(multiply(-y, X*beta)))
        prob = Problem(Minimize(obj))
        prob.solve()
        cvxpy_beta = beta.value
        cvxpy_obj = obj.value
        print('CVXPY finished.')

        # Split problem by row.
        # minimize \sum_{i,j} log(1 + exp(-y_{ij}*Z_{ij})) subject to Z_i = X_i\beta with variables (Z_1,...,Z_K,\beta),
        # where y_i is the i-th (m/N) subvector and X_i is the i-th (m/N) x p submatrix for i = 1,...,K.
        K = 4                # Number of splits.
        m_split = int(m / K)   # Rows in each split.
        y_split = np.split(y, K)

        # Convert problem to standard form.
        # f_i(Z_i) = \sum_j log(1 + exp(-y_{ij}*Z_{ij})) for i = 1,...,K.
        # f_{K+1}(\beta) = 0.
        # A_1 = [I; 0; ...; 0], A_2 = [0; I; 0; ...; 0], ..., A_K = [0; ...; 0; I], A_{K+1} = -X, b = 0.
        prox_list = [lambda v, t, i=i: prox_logistic(v, t, y=y_split[i]) for i in range(K)] + \
                    [prox_constant]
        A_list = []
        for i in range(K):
            mat_top = sparse.csr_matrix((i*m_split, m_split))
            mat_bot = sparse.csr_matrix((m-(i+1)*m_split, m_split))
            A_list += [sparse.vstack([mat_top, sparse.eye(m_split), mat_bot])]
        A_list += [-X]
        b = np.zeros(K*m_split)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        drs_beta = drs_result["x_vals"][-1]
        drs_obj = np.sum(-np.log(sp.special.expit(np.multiply(y, X.dot(drs_beta)))))
        print('DRS finished.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum(-np.log(sp.special.expit(np.multiply(y, X.dot(a2dr_beta)))))
        print('A2DR finished.')

        # Compare results.
        self.compare_total(drs_result, a2dr_result)
        # self.assertItemsAlmostEqual(a2dr_beta, cvxpy_beta, places=3)
        # self.assertItemsAlmostEqual(a2dr_obj, cvxpy_obj, places=4)

if __name__ == '__main__':
    tests = TestOther()
    tests.setUp()
    tests.test_simple_logistic()
