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
from cvxpy.constraints.psd import PSD
from cvxpy import Constant, Variable, Problem, Minimize
from cvxconsensus.proximal import prox_func
from cvxconsensus.tests.base_test import BaseTest

class TestProximal(BaseTest):
    """Unit tests for simple proximal operators"""

    def setUp(self):
        np.random.seed(1)
        self.x = Variable(10)
        self.Y = Variable((5,5))

        self.rho = 5*np.abs(np.random.randn()) + 1e-6
        self.u = np.random.randn(*self.x.shape)
        self.A = np.random.randn(*self.Y.shape)

    def compare_prox(self, expr, x_var, u_val, rho = 1.0, constr = [], places = 4):
        u_prox = prox_func(expr, u_val, rho, constr)
        Problem(Minimize(expr + 1/(2 * rho) * sum_squares(x_var - u_val))).solve()
        self.assertItemsAlmostEqual(u_prox, x_var.value, places)

    def test_prox_scalar(self):
        self.compare_prox(Constant(0), self.x, self.u, self.rho)
        self.compare_prox(norm1(self.x), self.x, self.u, self.rho)
        self.compare_prox(pnorm(self.x,2), self.x, self.u, self.rho)

    def test_prox_matrix(self):
        self.compare_prox(normNuc(self.Y), self.Y, self.A, self.rho)
        self.compare_prox(norm(self.Y, "fro"), self.Y, self.A, self.rho)
        self.compare_prox(sum(abs(self.Y)), self.Y, self.A, self.rho, places = 4)
        self.compare_prox(trace(self.Y), self.Y, self.A, self.rho)
        # self.compare_prox(sigma_max(self.Y), self.Y, self.A, self.rho)
        # self.compare_prox(-log_det(self.Y), self.Y, self.A, self.rho, places = 3)

        Y_symm = Variable(self.Y.shape, symmetric = True)
        A_symm = (self.A + self.A.T)/2
        self.compare_prox(Constant(0), Y_symm, A_symm, self.rho, [PSD(Y_symm)])
