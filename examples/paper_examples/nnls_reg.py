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

    def test_nnls_reg(self):
        # minimize ||Fz - g||_2^2 subject to z >= 0.

        # Problem data.
        p, q = 300, 200
        density = 0.001
        F = sparse.random(p, q, density=density, data_rvs=np.random.randn)
        g = np.random.randn(p)

        # Convert problem to standard form.
        # f_1(x_1) = ||Fx - g||_2^2, f_2(x_2) = I(x_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F, g), prox_nonneg_constr]
        A_list = [sparse.eye(q), -sparse.eye(q)]
        b = np.zeros(q)

        # Solve with no regularization.
        a2dr_noreg_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, lam_accel=0, max_iter=self.MAX_ITER)
        print('Finish A2DR no regularization.')
        
        # Solve with constant regularization.
        a2dr_consreg_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, ada_reg=False, max_iter=self.MAX_ITER)
        print('Finish A2DR constant regularization.')
        
        # Solve with adaptive regularization.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, ada_reg=True, max_iter=self.MAX_ITER) 
        print('Finish A2DR adaptive regularization.')
        
        self.compare_total_all([a2dr_noreg_result, a2dr_consreg_result, a2dr_result], 
                               ['no-reg', 'constant-reg', 'ada-reg'])

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_nnls_reg()

