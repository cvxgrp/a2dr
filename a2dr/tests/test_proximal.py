"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

import scipy as sp
from cvxpy.atoms import *
from a2dr.proximal.prox_operators import *
from a2dr.tests.base_test import BaseTest

class TestProximal(BaseTest):
    """Unit tests for simple proximal operators"""

    def setUp(self):
        np.random.seed(1)
        self.x = Variable(10)
        self.Y = Variable((5,5))

        self.rho = 5*np.abs(np.random.randn()) + 1e-6
        self.u = np.random.randn(100)
        self.v = np.random.randn(*self.x.shape)

        self.A = np.random.randn(self.u.size, self.x.size)
        self.B = np.random.randn(*self.Y.shape)

    def compare_prox_func(self, expr, x_var, u_val, rho = 1.0, constr = [], places = 4):
        prox_func = prox_func_vector if np.isscalar(u_val) or len(u_val.shape) <= 1 else prox_func_matrix
        u_prox = prox_func(expr, constr)(u_val, rho)
        Problem(Minimize(expr + (rho / 2.0) * sum_squares(x_var - u_val)), constr).solve()
        self.assertItemsAlmostEqual(u_prox, x_var.value, places)

    def compare_prox_oper(self, prob, y_vals, rho_vals = {}, places = 4):
        ProxOperator(prob, y_vals, rho_vals, use_cvxpy = True).solve()
        var_cvxpy = [var.value for var in prob.variables()]
        ProxOperator(prob, y_vals, rho_vals, use_cvxpy = False).solve()
        var_simple = [var.value for var in prob.variables()]
        for i in range(len(var_simple)):
            self.assertItemsAlmostEqual(var_simple[i], var_cvxpy[i], places)

    def test_is_simple_prox(self):
        self.assertTrue(is_simple_prox(norm1(self.x), [], [self.x], True))
        self.assertFalse(is_simple_prox(norm1(self.A*self.x + self.u), [], [self.x], True))
        self.assertTrue(is_simple_prox(sum(abs(self.x)), [], [self.x], False))
        self.assertFalse(is_simple_prox(sum(abs(self.A*self.x)), [], [self.x], False))
        # TODO: Add tests to ensure deeper expression trees are rejected.

    def test_prox_scalar(self):
        self.compare_prox_func(Constant(0), self.x, self.v, self.rho)
        self.compare_prox_func(norm1(self.x), self.x, self.v, self.rho)
        self.compare_prox_func(pnorm(self.x, 2), self.x, self.v, self.rho)
        self.compare_prox_func(max(self.x), self.x, self.v, self.rho)

#     def test_prox_matrix(self):
#         self.compare_prox_func(normNuc(self.Y), self.Y, self.B, self.rho, places = 3)
#         self.compare_prox_func(norm(self.Y, "fro"), self.Y, self.B, self.rho)
#         self.compare_prox_func(sum(abs(self.Y)), self.Y, self.B, self.rho, places = 4)
#         self.compare_prox_func(trace(self.Y), self.Y, self.B, self.rho)
#         # self.compare_prox_func(sigma_max(self.Y), self.Y, self.A, self.rho)

#         B = np.random.randn(self.Y.shape[1],self.Y.shape[1])
#         self.compare_prox_func(trace(self.Y * B), self.Y, self.B, self.rho)

#         A_symm = (self.B + self.B.T) / 2.0
#         self.compare_prox_func(Constant(0), self.Y, A_symm, self.rho, [self.Y >> 0, self.Y == self.Y.T], places = 3)

#         A_spd = self.B.dot(self.B.T)
#         self.compare_prox_func(-log_det(self.Y), self.Y, A_spd, self.rho, places = 3)

    def test_prox_operator(self):
        x_to_u = {self.x.id: self.v}
        x_to_rho = {self.x.id: self.rho}
        self.compare_prox_oper(Problem(Minimize(Constant(0))), {}, {})
        self.compare_prox_oper(Problem(Minimize(norm1(self.x))), x_to_u, x_to_rho)
        self.compare_prox_oper(Problem(Minimize(2*(norm1(self.x)))), x_to_u, x_to_rho)
        self.compare_prox_oper(Problem(Minimize(2*(norm1(self.x)) + norm_inf(self.x))), x_to_u, x_to_rho)

        Y_to_A = {self.Y.id: self.B}
        Y_to_rho = {self.Y.id: self.rho}
        self.compare_prox_oper(Problem(Minimize(normNuc(self.Y))), Y_to_A, Y_to_rho, places = 3)
        self.compare_prox_oper(Problem(Minimize(norm(self.Y, "fro"))), Y_to_A, Y_to_rho)
        self.compare_prox_oper(Problem(Minimize(sum(abs(self.Y)))), Y_to_A, Y_to_rho)
        self.compare_prox_oper(Problem(Minimize(trace(self.Y))), Y_to_A, Y_to_rho)
        # self.compare_prox_oper(Problem(Minimize(-log_det(self.Y))), Y_to_A, Y_to_rho)

        B = np.random.randn(self.Y.shape[1], self.Y.shape[1])
        self.compare_prox_oper(Problem(Minimize(trace(self.Y * B))), Y_to_A, Y_to_rho)
        self.compare_prox_oper(Problem(Minimize(0), [self.Y >> 0, self.Y == self.Y.T]), Y_to_A, Y_to_rho, places = 3)

        self.compare_prox_oper(Problem(Minimize(2*norm(self.Y, "fro"))), Y_to_A, Y_to_rho)
        self.compare_prox_oper(Problem(Minimize(2*sum(abs(self.Y)))), Y_to_A, Y_to_rho)

    def test_prox_logistic(self):
        # Scalar logistic.
        y = -1
        rho = 1.0
        z = np.random.randn()

        x = Variable()
        obj = logistic(-y*x) + (rho/2)*sum_squares(x - z)
        prob = Problem(Minimize(obj))
        prob.solve()
        cvxpy_var = x.value

        x0 = np.random.randn()
        scipy_var = prox_logistic(z, rho, x0, y)
        self.assertAlmostEqual(cvxpy_var, scipy_var)

        # Sum of logistic functions.
        m = 100
        Y = np.random.randint(0,2,size=m)
        Y = 2*Y - 1
        Z = np.random.randn(m)

        X = Variable(m)
        obj = sum(logistic(-multiply(Y,X))) + (rho/2)*sum_squares(X - Z)
        prob = Problem(Minimize(obj))
        prob.solve()
        cvxpy_var = X.value

        X0 = np.random.randn(m)
        scipy_var = prox_logistic(Z, rho, X0, Y)
        self.assertItemsAlmostEqual(cvxpy_var, scipy_var)
