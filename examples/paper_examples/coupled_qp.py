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
        L = 4     # number of blocks
        s = 10    # number of coupling constraints
        ql = 30   # variable dimension of each subproblem QP
        pl = 50   # constraint dimension of each subproblem QP

        G_list = [np.random.randn(s,ql) for l in range(L)]
        F_list = [np.random.randn(pl,ql) for l in range(L)]
        c_list = [np.random.randn(ql) for l in range(L)]
        z_tld_list = [np.random.randn(ql) for l in range(L)]
        d_list = [F_list[l].dot(z_tld_list[l])+0.1 for l in range(L)]
        
        G = np.hstack(G_list)
        z_tld = np.hstack(z_tld_list)
        h = G.dot(z_tld)
        H_list = [np.random.randn(ql,ql) for l in range(L)]
        Q_list = [H_list[l].T.dot(H_list[l]) for l in range(L)]
        
        # Convert problem to standard form.
        def prox_qp_wrapper(l, Q_list, c_list, F_list, d_list):
            return lambda v, t: prox_qp(v, t, Q_list[l], c_list[l], F_list[l], d_list[l])
        # Use "map" method to avoid implicit overriding, which would make all the proximal operators the same
        prox_list = list(map(lambda l: prox_qp_wrapper(l, Q_list, c_list, F_list, d_list), range(L)))
        A_list = G_list
        b = h
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        #drs_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER, ada_reg=False)
        #drs_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER, ada_reg=False, lam_accel=0)
        #drs_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER, ada_reg=False, lam_accel=1e-12)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        #a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER, lam_accel=1e-12)
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result)
        
        # Check solution correctness.
        a2dr_z = a2dr_result['x_vals']
        a2dr_obj = np.sum([a2dr_z[l].dot(Q_list[l]).dot(a2dr_z[l]) 
                           + c_list[l].dot(a2dr_z[l]) for l in range(L)])
        a2dr_constr_vio = [np.linalg.norm(np.maximum(F_list[l].dot(a2dr_z[l])-d_list[l],0))**2 
                                  for l in range(L)]
        a2dr_constr_vio += [np.linalg.norm(G.dot(np.hstack(a2dr_z))-h)**2]
        a2dr_constr_vio_val = np.sqrt(np.sum(a2dr_constr_vio))
        print('objective value of A2DR = {}'.format(a2dr_obj))
        print('constraint violation of A2DR = {}'.format(a2dr_constr_vio_val))

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_coupled_qp()

