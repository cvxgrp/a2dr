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

    def test_optimal_control(self):
        # Problem data.
        p = 20
        q = 40
        L = 5

        F = np.random.randn(q,q)
        G = np.random.randn(q,p)
        h = np.random.randn(q)
        z_init = np.random.randn(q)
        F = F / np.max(np.abs(LA.eigvals(F)))
        
        z_hat = z_init
        for l in range(L-1):
            u_hat = np.random.randn(p)
            u_hat = u_hat / np.max(np.abs(u_hat))
            z_hat = F.dot(z_hat) + G.dot(u_hat) + h
        z_term = z_hat
        # no normalization of u_hat actually leads to more significant improvement of A2DR over DRS, and also happens to be feasible
        # z_term = 0 also happens to be feasible
        
        # Convert problem to standard form.
        def prox_sat(v, t, v_lo = -np.inf, v_hi = np.inf):
            return prox_box_constr(prox_sum_squares(v, t), t, v_lo, v_hi)
        prox_list = [prox_sum_squares, lambda v, t: prox_sat(v, t, -1, 1)]
        A1 = sparse.lil_matrix(((L+1)*q,L*q))
        A1[q:L*q,:(L-1)*q] = -sparse.block_diag((L-1)*[F])
        A1.setdiag(1)
        A1[L*q:,(L-1)*q:] = sparse.eye(q)
        A2 = sparse.lil_matrix(((L+1)*q,L*p))
        A2[q:L*q,:(L-1)*p] = -sparse.block_diag((L-1)*[G])
        A_list = [sparse.csr_matrix(A1), sparse.csr_matrix(A2)]
        b_list = [z_init]
        b_list.extend((L-1)*[h])
        b_list.extend([z_term])
        b = np.concatenate(b_list)
        
        # Solve with CVXPY
        z = Variable((L,q))
        u = Variable((L,p))
        obj = sum([sum_squares(z[l]) + sum_squares(u[l]) for l in range(L)])
        constr = [z[0] == z_init, norm_inf(u) <= 1]
        constr += [z[l+1] == F*z[l] + G*u[l] + h for l in range(L-1)]
        constr += [z[L-1] == z_term]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver='SCS', eps=self.eps_abs, verbose=True) 
        # OSQP fails for p=50, q=100, L=30, and also for p=100, q=200, L=30
        # SCS also fails to converge
        cvxpy_obj = prob.value
        cvxpy_z = z.value.ravel(order='C')
        cvxpy_u = u.value.ravel(order='C')
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result)
        print('Finished A2DR.')
        
        # check solution correctness
        a2dr_z = a2dr_result['x_vals'][0]
        a2dr_u = a2dr_result['x_vals'][1]
        a2dr_obj = np.sum(a2dr_z**2) + np.sum(a2dr_u**2)
        cvxpy_obj_raw = np.sum(cvxpy_z**2) + np.sum(cvxpy_u**2)
        cvxpy_Z = cvxpy_z.reshape([L,q], order='C')
        cvxpy_U = cvxpy_u.reshape([L,p], order='C')
        a2dr_Z = a2dr_z.reshape([L,q], order='C')
        a2dr_U = a2dr_u.reshape([L,p], order='C')
        cvxpy_constr_vio = [np.linalg.norm(cvxpy_Z[0]-z_init), np.linalg.norm(cvxpy_Z[L-1]-z_term)]
        a2dr_constr_vio = [np.linalg.norm(a2dr_Z[0]-z_init), np.linalg.norm(a2dr_Z[L-1]-z_term)]
        for l in range(L-1):
            cvxpy_constr_vio.append(np.linalg.norm(cvxpy_Z[l+1]-F.dot(cvxpy_Z[l])-G.dot(cvxpy_U[l])-h))
            a2dr_constr_vio.append(np.linalg.norm(a2dr_Z[l+1]-F.dot(a2dr_Z[l])-G.dot(a2dr_U[l])-h))    
        print('linear constr vio cvxpy = {}, linear constr vio a2dr = {}'.format(
            np.mean(cvxpy_constr_vio), np.mean(a2dr_constr_vio)))
        print('norm constr vio cvxpy = {}, norm constr vio a2dr = {}'.format(np.max(np.abs(cvxpy_u)), 
                                                                             np.max(np.abs(a2dr_u))))
        print('objective cvxpy = {}, objective cvxpy raw = {}, objective a2dr = {}'.format(cvxpy_obj, 
                                                                                           cvxpy_obj_raw,
                                                                                           a2dr_obj))

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_optimal_control()

