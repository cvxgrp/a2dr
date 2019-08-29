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

    def test_nnls(self):
        # minimize ||Fx - g||_2^2 subject to x >= 0.

        # Problem data.
        m, n = 150, 300 
        density = 0.001
        F = sparse.random(m, n, density=density, data_rvs=np.random.randn)
        g = np.random.randn(m)

        # Convert problem to standard form.
        # f_1(x_1) = ||Fx_1 - g||_2^2, f_2(x_2) = I(x_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F, g), prox_nonneg_constr]
        A_list = [sparse.eye(n), -sparse.eye(n)]
        b = np.zeros(n)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finish DRS.')
    
        # Solve with A2DR.
        t0 = time.time()
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        t1 = time.time()
        a2dr_beta = a2dr_result["x_vals"][-1]
        print('nonzero entries proportion = {}'.format(np.sum(a2dr_beta > 0)*1.0/len(a2dr_beta)))
        print('Finish A2DR.')
        self.compare_total(drs_result, a2dr_result)
        
        # Check solution correctness.
        print('run time of A2DR = {}'.format(t1-t0))
        print('constraint violation of A2DR = {}'.format(np.min(a2dr_beta)))
        print('objective value of A2DR = {}'.format(np.linalg.norm(F.dot(a2dr_beta)-g)))

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_nnls()

