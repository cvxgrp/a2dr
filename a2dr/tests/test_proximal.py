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
        self.B_psd = self.B.T.dot(self.B)

    def prox_cvxpy(self, v, fun, constr_fun = None, t = 1, scale = 1, offset = 0, lin_term = 0, quad_term = 0, *args, **kwargs):
        x_var = Variable() if np.isscalar(v) else Variable(v.shape)
        expr = t * fun(scale * x_var - offset) + sum(multiply(lin_term, x_var)) + quad_term * sum_squares(x_var)
        constrs = [] if constr_fun is None else constr_fun(scale * x_var - offset)
        Problem(Minimize(expr + 0.5 * sum_squares(x_var - v)), constrs).solve(*args, **kwargs)
        return x_var.value

    def composition_check(self, prox, fun, v_init, places = 4, *args, **kwargs):
        x_a2dr = prox(v_init)
        x_cvxpy = self.prox_cvxpy(v_init, fun, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, t = self.t)
        x_cvxpy = self.prox_cvxpy(v_init, fun, t = self.t, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, scale = -1)
        x_cvxpy = self.prox_cvxpy(v_init, fun, scale = -1, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, scale = 2, offset = 0.5)
        x_cvxpy = self.prox_cvxpy(v_init, fun, scale = 2, offset = 0.5, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        x_a2dr = prox(v_init, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5)
        x_cvxpy = self.prox_cvxpy(v_init, fun, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5,
                                  *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

        if np.isscalar(v_init):
            offset = np.random.randn()
            lin_term = np.random.randn()
        else:
            offset = np.random.randn(*v_init.shape)
            lin_term = np.random.randn(*v_init.shape)
        x_a2dr = prox(v_init, t = self.t, scale = 0.5, offset = offset, lin_term = lin_term, quad_term = 2.5)
        x_cvxpy = self.prox_cvxpy(v_init, fun, t = self.t, scale = 0.5, offset = offset, lin_term = lin_term,
                                  quad_term = 2.5, *args, **kwargs)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = places)

    def test_box_constr(self):
        # Projection onto nonnegative/nonpositive orthant.
        self.composition_check(prox_nonneg_constr, lambda x: 0, self.v, constr_fun = lambda x: [x >= 0])
        self.composition_check(prox_nonneg_constr, lambda x: 0, self.B, constr_fun = lambda x: [x >= 0], places = 3)

        self.composition_check(prox_nonpos_constr, lambda x: 0, self.v, constr_fun = lambda x: [x <= 0])
        self.composition_check(prox_nonpos_constr, lambda x: 0, self.B, constr_fun = lambda x: [x <= 0], places = 3)

        # Projection onto a box interval.
        bounds = [(0, 0), (-1, 1), (0, np.inf), (-np.inf, 0)]
        for bound in bounds:
            lo, hi = bound
            self.composition_check(lambda v, *args, **kwargs: prox_box_constr(v, v_lo = lo, v_hi = hi, *args, **kwargs),
                                   lambda x: 0, self.v, constr_fun = lambda x: [lo <= x, x <= hi])
            self.composition_check(lambda v, *args, **kwargs: prox_box_constr(v, v_lo = lo, v_hi = hi, *args, **kwargs),
                                   lambda x: 0, self.B, constr_fun = lambda x: [lo <= x, x <= hi])

    def test_huber(self):
        for M in [0, 0.5, 1, 2]:
            # Scalar input.
            self.composition_check(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: huber(x, M = M), np.random.randn(), places = 4)
            # Vector input.
            self.composition_check(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: sum(huber(x, M = M)), self.v, places = 4)
            # Matrix input.
            self.composition_check(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: sum(huber(x, M = M)), self.B, places = 4)

    def test_logistic(self):
        # TODO: Add more tests with different y.
        self.composition_check(prox_logistic, lambda x: sum(logistic(x)), self.v, places = 4)

    def test_norm1(self):
        # General composition tests.
        self.composition_check(prox_norm1, norm1, np.random.randn(), places = 4)
        self.composition_check(prox_norm1, norm1, self.v, places = 4)
        self.composition_check(prox_norm1, norm1, self.B, places = 4)

        # l1 trend filtering: f(x) = \alpha*||x||_1
        alpha = 0.5 + np.abs(np.random.randn())
        x_a2dr = prox_norm1(self.v, t = alpha*self.t)
        x_cvxpy = self.prox_cvxpy(self.v, norm1, t = alpha*self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        # Sparse inverse covariance estimation: f(B) = \alpha*||B||_1
        B_symm_a2dr = prox_norm1(self.B_symm, t = alpha*self.t)
        B_symm_cvxpy = self.prox_cvxpy(self.B_symm, norm1, t = alpha*self.t)
        self.assertItemsAlmostEqual(B_symm_a2dr, B_symm_cvxpy, places = 4)

    def test_norm2(self):
        # General composition tests.
        self.composition_check(prox_norm2, norm2, np.random.randn(), places = 4)
        self.composition_check(prox_norm2, norm2, self.v, places = 4, solver = "SCS")
        self.composition_check(prox_norm2, lambda B: cvxpy.norm(B, 'fro'), self.B, places = 4, solver = "SCS")

        # f(x) = \alpha*||x||_2
        alpha = 0.5 + np.abs(np.random.randn())
        x_a2dr = prox_norm2(self.v, t = alpha*self.t)
        x_cvxpy = self.prox_cvxpy(self.v, norm2, t = alpha*self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    # def test_norm_inf(self):
    #     # TODO: Numbers are wrong here.
    #     # General composition tests.
    #     self.composition_check(prox_norm_inf, norm_inf, np.random.randn(), places = 4)
    #     self.composition_check(prox_norm_inf, norm_inf, self.v, places = 4)
    #     # self.composition_check(prox_norm_inf, norm_inf, self.B, places = 4)

    #     # f(x) = \alpha*||x||_{\infty}
    #     alpha = 0.5 + np.abs(np.random.randn())
    #     x_a2dr = prox_norm_inf(self.v, t = alpha*self.t)
    #     x_cvxpy = self.prox_cvxpy(self.v, norm_inf, t = alpha*self.t)
    #     self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_norm_nuc(self):
        B_a2dr = prox_norm_nuc(self.B)
        B_cvxpy = self.prox_cvxpy(self.B, normNuc)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        # f(B) = (1/2)*||B||_*
        B_a2dr = prox_norm_nuc(self.B, t = 0.5*self.t)
        B_cvxpy = self.prox_cvxpy(self.B, normNuc, t = 0.5*self.t)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_norm_nuc(self.B, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        B_cvxpy = self.prox_cvxpy(self.B, normNuc, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

    def test_neg_log_det(self):
        # f(B) = -log det(B)
        B_a2dr = prox_neg_log_det(self.B_symm)
        B_cvxpy = self.prox_cvxpy(self.B_symm, lambda X: -log_det(X))
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_neg_log_det(self.B_symm, self.t)
        B_cvxpy = self.prox_cvxpy(self.B_symm, lambda X: -log_det(X), t = self.t)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_neg_log_det(self.B_symm, scale = 2, offset = 0.5)
        B_cvxpy = self.prox_cvxpy(self.B_symm, lambda X: -log_det(X), scale = 2, offset = 0.5)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        B_a2dr = prox_neg_log_det(self.B_symm, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        B_cvxpy = self.prox_cvxpy(self.B_symm, lambda X: -log_det(X), t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 1.75)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

    def test_sum_squares(self):
        # General composition tests.
        self.composition_check(prox_sum_squares, sum_squares, self.v, places = 4)
        self.composition_check(prox_sum_squares, sum_squares, self.B, places = 4)

        # f(x) = (1/2)*||x - offset||_2^2
        offset = np.random.randn(*self.v.shape)
        x_a2dr = prox_sum_squares(self.v, t = 0.5*self.t, offset = offset)
        x_cvxpy = self.prox_cvxpy(self.v, sum_squares, t = 0.5*self.t, offset = offset)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_sum_squares_affine(self):
        # Scalar terms.
        F = np.random.randn()
        g = np.random.randn()
        v = np.random.randn()

        self.composition_check(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method = "lsqr",
                                    *args, **kwargs), lambda x: sum_squares(F*x - g), v)
        self.composition_check(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method = "lstsq",
                                    *args, **kwargs), lambda x: sum_squares(F*x - g), v)

        # Simple sum of squares.
        n = 100
        F = np.eye(n)
        g = np.zeros(n)
        v = np.random.randn(n)

        self.composition_check(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method = "lsqr",
                                    *args, **kwargs), lambda x: sum_squares(F*x - g), v)
        self.composition_check(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method = "lstsq",
                                    *args, **kwargs), lambda x: sum_squares(F*x - g), v)

        # General composition tests.
        m = 1000
        n = 100
        F = 10 + 5*np.random.randn(m,n)
        x = 2*np.random.randn(n)
        g = F.dot(x) + 0.01*np.random.randn(m)
        v = np.random.randn(n)

        self.composition_check(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method = "lsqr",
                                    *args, **kwargs), lambda x: sum_squares(F*x - g), v)
        self.composition_check(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method = "lstsq",
                                    *args, **kwargs), lambda x: sum_squares(F*x - g), v)

    def test_quad_form(self):
        # Simple quadratic.
        v = np.random.randn(1)
        Q = np.array([[5]])
        self.composition_check(lambda v, *args, **kwargs: prox_quad_form(v, Q = Q, *args, **kwargs),
                               lambda x: quad_form(x, P = Q), v)

        # General composition tests.
        n = 10
        v = np.random.randn(n)
        Q = np.random.randn(n,n)
        Q = Q.T.dot(Q) + 0.5*np.eye(n)
        self.composition_check(lambda v, *args, **kwargs: prox_quad_form(v, Q = Q, *args, **kwargs),
                               lambda x: quad_form(x, P = Q), v)
