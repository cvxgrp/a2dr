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
        self.B_symm = (self.B + self.B.T) / 2.0

    def prox_cvxpy(self, fun, v, t = 1, scale = 1, offset = 0, lin_term = 0, quad_term = 0, *args, **kwargs):
        x_var = Variable() if np.isscalar(v) else Variable(v.shape)
        expr = t * fun(scale * x_var - offset) + sum(lin_term * x_var) + quad_term * sum_squares(x_var)
        Problem(Minimize(expr + 0.5 * sum_squares(x_var - v))).solve(*args, **kwargs)
        return x_var.value

    def composition_tests(self, prox, fun, v_init, places = 4, *args, **kwargs):
        x_a2dr = prox(v_init)
        x_cvxpy = self.prox_cvxpy(fun, v_init, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, t = self.t)
        x_cvxpy = self.prox_cvxpy(fun, v_init, self.t, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, scale = -1)
        x_cvxpy = self.prox_cvxpy(fun, v_init, scale = -1, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, scale = 2, offset = 0.5)
        x_cvxpy = self.prox_cvxpy(fun, v_init, scale = 2, offset = 0.5, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5)
        x_cvxpy = self.prox_cvxpy(fun, v_init, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5,
                                  *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        if np.isscalar(v_init):
            offset = np.random.randn()
            lin_term = np.random.randn()
        else:
            offset = np.random.randn(*v_init.shape)
            lin_term = np.random.randn(*v_init.shape)
        x_a2dr = prox(v_init, t = self.t, scale = 0.5, offset = offset, lin_term = lin_term, quad_term = 2.5)
        x_cvxpy = self.prox_cvxpy(fun, v_init, t = self.t, scale = 0.5, offset = offset, lin_term = lin_term,
                                  quad_term = 2.5, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

    def test_sum_squares(self):
        # General composition tests.
        self.composition_tests(prox_sum_squares, sum_squares, self.v, places = 4)

        # f(x) = (1/2)*||x - offset||_2^2
        offset = np.random.randn(*self.v.shape)
        x_a2dr = prox_sum_squares(self.v, t = 0.5*self.t, offset = offset)
        x_cvxpy = self.prox_cvxpy(sum_squares, self.v, t = 0.5*self.t, offset = offset)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_huber(self):
        for M in [0, 0.5, 1, 2]:
            # Scalar input.
            self.composition_tests(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: huber(x, M = M), np.random.randn(), places = 4)
            # Vector input.
            self.composition_tests(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: sum(huber(x, M = M)), self.v, places = 4)
            # TODO: Matrix input.

    def test_norm1(self):
        # General composition tests.
        self.composition_tests(prox_norm1, norm1, np.random.randn(), places=4)
        self.composition_tests(prox_norm1, norm1, self.v, places=4)
        # self.composition_tests(prox_norm1, norm1, self.B, places=4)

        # f(x) = (1/2)*||x||_1
        x_a2dr = prox_norm1(self.v, t=0.5 * self.t)
        x_cvxpy = self.prox_cvxpy(norm1, self.v, t=0.5 * self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places=4)

    def test_norm2(self):
        # General composition tests.
        self.composition_tests(prox_norm2, norm2, np.random.randn(), places=4)
        self.composition_tests(prox_norm2, norm2, self.v, places=4, solver="SCS")
        # self.composition_tests(prox_norm2, norm2, self.B, places=4)

        # f(x) = (1/2)*||x||_2
        x_a2dr = prox_norm2(self.v, t=0.5 * self.t)
        x_cvxpy = self.prox_cvxpy(norm2, self.v, t=0.5 * self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places=4)

    def test_norm_nuc(self):
        B_a2dr = prox_norm_nuc(self.B)
        B_cvxpy = self.prox_cvxpy(normNuc, self.B)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        # f(B) = (1/2)*||B||_*
        B_a2dr = prox_norm_nuc(self.B, t = 0.5*self.t)
        B_cvxpy = self.prox_cvxpy(normNuc, self.B, t = 0.5*self.t)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_norm_nuc(self.B, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        B_cvxpy = self.prox_cvxpy(normNuc, self.B, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

    def test_neg_log_det(self):
        # f(B) = -log det(B)
        B_a2dr = prox_neg_log_det(self.B_symm)
        B_cvxpy = self.prox_cvxpy(lambda X: -log_det(X), self.B_symm)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_neg_log_det(self.B_symm, self.t)
        B_cvxpy = self.prox_cvxpy(lambda X: -log_det(X), self.B_symm, self.t)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_neg_log_det(self.B_symm, scale = 2, offset = 0.5)
        B_cvxpy = self.prox_cvxpy(lambda X: -log_det(X), self.B_symm, scale = 2, offset = 0.5)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_neg_log_det(self.B_symm, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        B_cvxpy = self.prox_cvxpy(lambda X: -log_det(X), self.B_symm, self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)
