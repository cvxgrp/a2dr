"""
Copyright 2018 Anqi Fu

This file is part of CVXConsensus.

CVXConsensus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXConsensus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXConsensus. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from cvxpy.atoms import *
from cvxpy import Constant, Variable, Problem, Minimize
from cvxconsensus.proximal import prox_func, prox_func_mat
from cvxconsensus.tests.base_test import BaseTest

class TestProximal(BaseTest):
    """Unit tests for simple proximal operators"""

    def setUp(self):
        self.x = Variable(10)
        self.Y = Variable((5,5))

        self.rho = 5*np.abs(np.random.randn()) + 1e-6
        self.u = np.random.randn(*self.x.shape)
        self.A = np.random.randn(*self.Y.shape)

    def test_prox_func(self):
        f_val = Constant(0)
        u_prox = prox_func(f_val, self.u, self.rho)
        Problem(Minimize(f_val + 1/(2*self.rho) * sum_squares(self.x - self.u))).solve()
        self.assertItemsAlmostEqual(u_prox, self.x.value)

        f_val = norm1(self.x)
        u_prox = prox_func(f_val, self.u, self.rho)
        Problem(Minimize(f_val + 1/(2*self.rho) * sum_squares(self.x - self.u))).solve()
        self.assertItemsAlmostEqual(u_prox, self.x.value)

    def test_prox_func_mat(self):
        f_val = normNuc(self.Y)
        A_prox = prox_func_mat(f_val, self.A, self.rho)
        Problem(Minimize(f_val + 1/(2*self.rho) * sum_squares(self.Y - self.A))).solve()
        self.assertItemsAlmostEqual(A_prox, self.Y.value)

        f_val = -log_det(self.Y)
        A_prox = prox_func_mat(f_val, self.A, self.rho)
        Problem(Minimize(f_val + 1/(2*self.rho) * sum_squares(self.Y - self.A))).solve()
        self.assertItemsAlmostEqual(A_prox, self.Y.value)

        Y_symm = Variable(self.Y.shape, symmetric = True)
        constr = [Y_symm >> 0]
        A_symm = (self.A + self.A.T)/2
        A_prox = prox_func_mat(Y_symm, A_symm, self.rho, constr)
        Problem(Minimize(1/(2*self.rho) * sum_squares(Y_symm - A_symm)), constr).solve()
        self.assertItemsAlmostEqual(A_prox, Y_symm.value)
