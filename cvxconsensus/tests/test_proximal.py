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
import cvxpy
from cvxpy.atoms import *
from cvxpy.constraints.psd import PSD
from cvxpy import Constant, Variable, Problem, Minimize
from cvxconsensus.proximal import ProxOperator, prox_func_vector, prox_func_matrix
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

    def compare_prox_func(self, expr, x_var, u_val, rho = 1.0, constr = [], places = 4):
        prox_func = prox_func_vector if np.isscalar(u_val) or len(u_val.shape) <= 1 else prox_func_matrix
        u_prox = prox_func(expr, constr)(u_val, rho)
        Problem(Minimize(expr + (rho / 2.0) * sum_squares(x_var - u_val))).solve()
        self.assertItemsAlmostEqual(u_prox, x_var.value, places)

    def compare_prox_operator(self, prob, rho_init = 1.0, places = 4):
        ProxOperator(prob, rho_init, use_cvxpy = True).solve()
        var_cvxpy = [var.value for var in prob.variables()]
        ProxOperator(prob, rho_init, use_cvxpy = False).solve()
        var_simple = [var.value for var in prob.variables()]
        for i in range(len(var_simple)):
            self.assertItemsAlmostEqual(var_simple[i], var_cvxpy[i], places)

    def test_prox_scalar(self):
        self.compare_prox_func(Constant(0), self.x, self.u, self.rho)
        self.compare_prox_func(norm1(self.x), self.x, self.u, self.rho)
        self.compare_prox_func(pnorm(self.x, 2), self.x, self.u, self.rho)
        self.compare_prox_func(max(self.x), self.x, self.u, self.rho)

    def test_prox_matrix(self):
        self.compare_prox_func(normNuc(self.Y), self.Y, self.A, self.rho)
        self.compare_prox_func(norm(self.Y, "fro"), self.Y, self.A, self.rho)
        self.compare_prox_func(sum(abs(self.Y)), self.Y, self.A, self.rho, places = 4)
        self.compare_prox_func(trace(self.Y), self.Y, self.A, self.rho)
        # self.compare_prox_func(sigma_max(self.Y), self.Y, self.A, self.rho)
        # self.compare_prox_func(-log_det(self.Y), self.Y, self.A, self.rho, places = 3)

        B = np.random.randn(self.Y.shape[1],self.Y.shape[1])
        self.compare_prox_func(trace(self.Y * B), self.Y, self.A, self.rho)

        Y_symm = Variable(self.Y.shape, symmetric = True)
        A_symm = (self.A + self.A.T)/2
        self.compare_prox_func(Constant(0), Y_symm, A_symm, self.rho, [PSD(Y_symm)])

    def test_prox_operator(self):
        # TODO: Debug for when parameter y is a random value.
        self.compare_prox_operator(Problem(Minimize(Constant(0)), {}))
        self.compare_prox_operator(Problem(Minimize(norm1(self.x))), {self.x.id: self.rho})

        # self.compare_prox_operator(Problem(Minimize(-log_det(self.Y))), {self.Y.id: self.rho})
        self.compare_prox_operator(Problem(Minimize(normNuc(self.Y))), {self.Y.id: self.rho})
        self.compare_prox_operator(Problem(Minimize(norm(self.Y, "fro"))), {self.Y.id: self.rho})
        self.compare_prox_operator(Problem(Minimize(sum(abs(self.Y)))), {self.Y.id: self.rho})
        self.compare_prox_operator(Problem(Minimize(trace(self.Y))), {self.Y.id: self.rho})

        B = np.random.randn(self.Y.shape[1], self.Y.shape[1])
        self.compare_prox_operator(Problem(Minimize(trace(self.Y * B))), {self.Y.id: self.rho})
