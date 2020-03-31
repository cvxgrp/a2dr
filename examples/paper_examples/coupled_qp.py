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

    def test_coupled_qp(self):
        # Problem data.
        K = 4 # number of blocks
        p = 10 # number of coupling constraints
        nk = 30 # variable dimension of each subproblem QP
        mk = 50 # constraint dimension of each subproblem QP
        A_list = [np.random.randn(p, nk) for k in range(K)]
        F_list = [np.random.randn(mk, nk) for k in range(K)]
        q_list = [np.random.randn(nk) for k in range(K)]
        x_list = [np.random.randn(nk) for k in range(K)]
        g_list = [F_list[k].dot(x_list[k])+0.1 for k in range(K)]
        A = np.hstack(A_list)
        x = np.hstack(x_list)
        b = A.dot(x)
        P_list = [np.random.randn(nk,nk) for k in range(K)]
        Q_list = [P_list[k].T.dot(P_list[k]) for k in range(K)]
        
        # Convert problem to standard form.
        def tmp(k, Q_list, q_list, F_list, g_list):
            return lambda v, t: prox_qp(v, t, Q_list[k], q_list[k], F_list[k], g_list[k])
        # Use "map" method to avoid implicit overriding, which would make all the proximal operators the same
        prox_list = list(map(lambda k: tmp(k,Q_list,q_list,F_list,g_list), range(K)))
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result)
        
        # Check solution correctness.
        a2dr_x = a2dr_result['x_vals']
        a2dr_obj = np.sum([a2dr_x[k].dot(Q_list[k]).dot(a2dr_x[k]) 
                           + q_list[k].dot(a2dr_x[k]) for k in range(K)])
        a2dr_constr_vio = [np.linalg.norm(np.maximum(F_list[k].dot(a2dr_x[k])-g_list[k],0))**2 
                                  for k in range(K)]
        a2dr_constr_vio += [np.linalg.norm(A.dot(np.hstack(a2dr_x))-b)**2]
        a2dr_constr_vio_val = np.sqrt(np.sum(a2dr_constr_vio))
        print('objective value of A2DR = {}'.format(a2dr_obj))
        print('constraint violation of A2DR = {}'.format(a2dr_constr_vio_val))

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_coupled_qp()

