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
        # Problem data/
        m = 20
        n = 40
        K = 5
        A = np.random.randn(n,n)
        B = np.random.randn(n,m)
        c = np.random.randn(n)
        x_init = np.random.randn(n)
        A = A / np.max(np.abs(LA.eigvals(A)))
        xhat = x_init
        for k in range(K-1):
            uhat = np.random.randn(m)
            uhat = uhat / np.max(np.abs(uhat))
            xhat = A.dot(xhat) + B.dot(uhat) + c
        x_term = xhat
        # uhat no normalization actually leads to more significant improvement of A2DR over DRS, and also happens to be feasible
        # x_term = 0 also happens to be feasible
        
        # Convert problem to standard form.
        prox_list = [prox_sum_squares, prox_sat(1,1)]
        A1 = sparse.lil_matrix(((K+1)*n,K*n))
        A1[n:K*n,:(K-1)*n] = -sparse.block_diag((K-1)*[A])
        A1.setdiag(1)
        A1[K*n:,(K-1)*n:] = sparse.eye(n)
        A2 = sparse.lil_matrix(((K+1)*n,K*m))
        A2[n:K*n,:(K-1)*m] = -sparse.block_diag((K-1)*[B])
        A_list = [sparse.csr_matrix(A1), sparse.csr_matrix(A2)]
        b_list = [x_init]
        b_list.extend((K-1)*[c])
        b_list.extend([x_term])
        b = np.concatenate(b_list)
        
        # Solve with CVXPY
        x = Variable((K,n))
        u = Variable((K,m))
        obj = sum([sum_squares(x[k]) + sum_squares(u[k]) for k in range(K)])
        constr = [x[0] == x_init, norm_inf(u) <= 1]
        constr += [x[k+1] == A*x[k] + B*u[k] + c for k in range(K-1)]
        constr += [x[K-1] == x_term]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver='SCS', eps=self.eps_abs, verbose=True) 
        # OSQP fails for m=50, n=100, K=30, and also for m=100, n=200, K=30
        # SCS also fails to converge
        cvxpy_obj = prob.value
        cvxpy_x = x.value.ravel(order='C')
        cvxpy_u = u.value.ravel(order='C')
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result)
        print('Finished A2DR.')
        
        # check solution correctness
        a2dr_x = a2dr_result['x_vals'][0]
        a2dr_u = a2dr_result['x_vals'][1]
        a2dr_obj = np.sum(a2dr_x**2) + np.sum(a2dr_u**2)
        cvxpy_obj_raw = np.sum(cvxpy_x**2) + np.sum(cvxpy_u**2)
        cvxpy_X = cvxpy_x.reshape([K,n], order='C')
        cvxpy_U = cvxpy_u.reshape([K,m], order='C')
        a2dr_X = a2dr_x.reshape([K,n], order='C')
        a2dr_U = a2dr_u.reshape([K,m], order='C')
        cvxpy_constr_vio = [np.linalg.norm(cvxpy_X[0]-x_init), np.linalg.norm(cvxpy_X[K-1]-x_term)]
        a2dr_constr_vio = [np.linalg.norm(a2dr_X[0]-x_init), np.linalg.norm(a2dr_X[K-1]-x_term)]
        for k in range(K-1):
            cvxpy_constr_vio.append(np.linalg.norm(cvxpy_X[k+1]-A.dot(cvxpy_X[k])-B.dot(cvxpy_U[k])-c))
            a2dr_constr_vio.append(np.linalg.norm(a2dr_X[k+1]-A.dot(a2dr_X[k])-B.dot(a2dr_U[k])-c))    
        print('linear constr vio cvxpy = {}, linear constr_vio a2dr = {}'.format(
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

