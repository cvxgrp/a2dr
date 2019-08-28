import numpy as np
from cvxpy import *
from a2dr.proximal import *
from a2dr.tests.base_test import BaseTest

class TestProximal(BaseTest):

    def setUp(self):
        np.random.seed(1)
        self.t = 5*np.abs(np.random.randn()) + 1e-6
        self.v = np.random.randn(100)
        self.B = np.random.randn(10,10)

    def prox_cvxpy(self, fun, v, t = 1, scale = 1, offset = 0, lin_term = 0, quad_term = 0):
        x_var = Variable(v.shape)
        expr = t * fun(scale * x_var - offset) + sum(lin_term * x_var) + quad_term * sum_squares(x_var)
        Problem(Minimize(expr + 0.5 * sum_squares(x_var - v))).solve()
        return x_var.value

    def test_sum_squares(self):
        x_a2dr = prox_sum_squares(self.v)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        x_a2dr = prox_sum_squares(self.v, self.t)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v, self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        x_a2dr = prox_sum_squares(self.v, scale = -1)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v, scale = -1)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        x_a2dr = prox_sum_squares(self.v, scale = 2, offset = 0.5)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v, scale = 2, offset = 0.5)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        x_a2dr = prox_sum_squares(self.v, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        offset = np.random.randn(*self.v.shape)
        lin_term = np.random.randn(*self.v.shape)
        x_a2dr = prox_sum_squares(self.v, self.t, scale = 2, offset = offset, lin_term = lin_term, quad_term = 2.5)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v, self.t, scale = 2, offset = offset, lin_term = lin_term, quad_term = 2.5)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_huber(self):
        x_a2dr = prox_huber(self.v, M = 2)
        x_cvxpy = self.prox_cvxpy(lambda x: sum(huber(x, M = 2)), self.v)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        x_a2dr = prox_huber(self.v, self.t, M = 2, scale = 0.5)
        x_cvxpy = self.prox_cvxpy(lambda x: sum(huber(x, M = 2)), self.v, self.t, scale = 0.5)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_norm_nuc(self):
        B_a2dr = prox_norm_nuc(self.B)
        B_cvxpy = self.prox_cvxpy(normNuc, self.B)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 4)

        B_a2dr = prox_norm_nuc(self.B, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        B_cvxpy = self.prox_cvxpy(normNuc, self.B, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 4)