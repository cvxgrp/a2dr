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

class TestPaper(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8   # specify these in all examples?
        self.eps_abs = 1e-6
        self.MAX_ITER = 1000

    def test_multi_task_logistic(self):
        # minimize \sum_{il} log(1 + exp(-Y_{il}*Z_{il})) + \alpha*||\theta||_{2,1} + \beta*||\theta||_*
        # subject to Z = W\theta, ||.||_{2,1} = group lasso, ||.||_* = nuclear norm.

        # Problem data.
        L = 3     # Number of tasks.
        s = 80    # Number of features.
        p = 100   # Number of samples.
        alpha = 0.1
        beta = 0.1

        W = np.random.randn(p,s)
        theta_true = np.random.randn(s,L)
        Z_true = W.dot(theta_true)
        Y = 2*(Z_true > 0) - 1   # Y_{ij} = 1 or -1.

        def calc_obj(theta):
            obj = np.sum(-np.log(sp.special.expit(np.multiply(Y, W.dot(theta)))))
            reg = alpha*np.sum([LA.norm(theta[:,l], 2) for l in range(L)])
            reg += beta*LA.norm(theta, ord='nuc')
            return obj + reg

        # Convert problem to standard form. 
        # f_1(Z) = \sum_{il} log(1 + exp(-Y_{il}*Z_{il})), 
        # f_2(\theta) = \alpha*||\theta||_{2,1}, 
        # f_3(\tilde \theta) = \beta*||\tilde \theta||_*.
        # A_1 = [I; 0], A_2 = [-W; I], A_3 = [0; -I], b = 0.
        prox_list = [lambda v, t: prox_logistic(v, t, y = Y.ravel(order='F')),   
                     # TODO: Calculate in parallel for l = 1,...L.
                     lambda v, t: prox_group_lasso(v.reshape((s,L), order='F'), t*alpha).ravel(order='F'),
                     lambda v, t: prox_norm_nuc(v.reshape((s,L), order='F'), t*beta).ravel(order='F')]
        A_list = [sparse.vstack([sparse.eye(p*L), sparse.csr_matrix((s*L,p*L))]),
                  sparse.vstack([-sparse.block_diag(L*[W]), sparse.eye(s*L)]),
                  sparse.vstack([sparse.csr_matrix((p*L,s*L)), -sparse.eye(s*L)])]
        b = np.zeros(p*L + s*L)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        a2dr_theta = a2dr_result["x_vals"][-1].reshape((s,L), order='F')
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result)

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_multi_task_logistic()

