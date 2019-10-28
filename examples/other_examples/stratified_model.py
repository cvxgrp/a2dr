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
import numpy.linalg as LA
import copy
import time
import scipy.sparse.linalg
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from cvxpy import *
from scipy import sparse
from scipy.optimize import nnls
from sklearn.datasets import make_sparse_spd_matrix

from a2dr import a2dr
from a2dr.proximal import *
from a2dr.tests.base_test import BaseTest

import networkx as nx

class TestOther(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8   # specify these in all examples?
        self.eps_abs = 1e-6
        self.MAX_ITER = 1000

    def test_strat_model(self):
        # http://web.stanford.edu/~boyd/papers/strat_models.html
        # minimize \sum_i \|A_i x_i - b_i\|_2^2 + lam \|x_i\|_2^2 +
        #          \sum_{i, j} W_{ij} \|x_i - x_j\|_2^2

        # Problem data.
        m, n, K = 30, 20, 5
        As = [np.random.randn(m, n) for _ in range(K)]
        bs = [np.random.randn(m) for _ in range(K)]
        G = nx.cycle_graph(K)
        L = nx.laplacian_matrix(G)
        lam = 1e-4

        # Convert problem to standard form.
        # f_1(x_1) = \sum_i \|A_i (x_1)_i - b_i\|_2^2
        # f_2(x_2) = \sum_i lam \|(x_2)_i\|_2^2
        # f_3(x_3) = \sum_{i, j} W_{ij} \|(x_3)_i - (x_3)_j\|_2^2
        # f_4(x_4) = 0
        # A_1 = [I 0 0]^T
        # A_2 = [0 I 0]^T
        # A_3 = [0 0 I]^T
        # A_4 = [-I -I -I]^T
        # b = 0

        def loss_prox(v, t):
            result = np.empty(0)
            for i in range(K):
                result = np.append(
                    result,
                    prox_sum_squares_affine(v[i*n:(i+1)*n], t, F=As[i], g=bs[i], method="lstsq")
                )                    
            return result

        def reg_prox(v, t):
            return prox_sum_squares(v, t, scale=lam)

        Q = sparse.kron(sparse.eye(n), L)
        Q = Q + 1e-12 * sparse.eye(n*K) # ensure positive-semi-definite-ness

        def laplacian_prox(v, t):
            return prox_quad_form(v, t, Q=Q, method="lsqr")
        
        def identity_prox(v, t):
            return v

        prox_list = [loss_prox, reg_prox, laplacian_prox, identity_prox]
        eye = sparse.eye(K*n)
        zero = sparse.csc_matrix((K*n, K*n))
        A_list = [
            sparse.vstack([eye, zero, zero]),
            sparse.vstack([zero, eye, zero]),
            sparse.vstack([zero, zero, eye]),
            sparse.vstack([-eye, -eye, -eye])
        ]
        b = np.zeros(3*K*n)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        print('A2DR finished.')

        self.compare_total(drs_result, a2dr_result)
        
if __name__ == '__main__':
    tests = TestOther()
    tests.setUp()
    tests.test_strat_model()
