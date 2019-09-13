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
from scipy import sparse
from cvxpy import *
from a2dr.proximal import *
from a2dr.tests.base_test import BaseTest

class TestProximal(BaseTest):
    """Unit tests for proximal operators"""

    def setUp(self):
        np.random.seed(1)
        self.TOLERANCE = 1e-6
        self.t = 5*np.abs(np.random.randn()) + self.TOLERANCE
        self.c = np.random.randn()
        self.v = np.random.randn(100)
        self.v_small = np.random.randn(10)

        self.B = np.random.randn(50,10)
        self.B_small = np.random.randn(10,5)
        self.B_square = np.random.randn(10,10)

        self.B_symm = np.random.randn(10,10)
        self.B_symm = (self.B_symm + self.B_symm.T) / 2.0
        self.B_psd = np.random.randn(10,10)
        self.B_psd = self.B_psd.T.dot(self.B_psd)

        self.u_sparse = sparse.random(100,1)
        self.u_dense = self.u_sparse.todense()
        self.C_sparse = sparse.random(50,10)
        self.C_dense = self.C_sparse.todense()
        self.C_square_sparse = sparse.random(50,50)
        self.C_square_dense = self.C_square_sparse.todense()

    def prox_cvxpy(self, v, fun, constr_fun = None, t = 1, scale = 1, offset = 0, lin_term = 0, quad_term = 0, *args, **kwargs):
        x_var = Variable() if np.isscalar(v) else Variable(v.shape)
        expr = t * fun(scale * x_var - offset) + sum(multiply(lin_term, x_var)) + quad_term * sum_squares(x_var)
        constrs = [] if constr_fun is None else constr_fun(scale * x_var - offset)
        prob = Problem(Minimize(expr + 0.5 * sum_squares(x_var - v)), constrs)
        prob.solve(*args, **kwargs)
        return x_var.value

    def check_composition(self, prox, fun, v_init, places = 3, *args, **kwargs):
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

    def check_elementwise(self, prox, places = 4):
        # Vector input.
        x_vec1 = prox(self.v_small)
        x_vec2 = np.array([prox(self.v_small[i]) for i in range(self.v_small.shape[0])])
        self.assertItemsAlmostEqual(x_vec1, x_vec2, places = places)

        x_vec1 = prox(self.v_small, t = self.t)
        x_vec2 = np.array([prox(self.v_small[i], t = self.t) for i in range(self.v_small.shape[0])])
        self.assertItemsAlmostEqual(x_vec1, x_vec2, places = places)

        offset = np.random.randn(*self.v_small.shape)
        lin_term = np.random.randn(*self.v_small.shape)
        x_vec1 = prox(self.v_small, t = self.t, scale = 0.5, offset = offset, lin_term = lin_term, quad_term = 2.5)
        x_vec2 = np.array([prox(self.v_small[i], t = self.t, scale = 0.5, offset = offset[i], lin_term = lin_term[i], \
                                quad_term = 2.5) for i in range(self.v_small.shape[0])])
        self.assertItemsAlmostEqual(x_vec1, x_vec2, places = places)

        # Matrix input.
        x_mat1 = prox(self.B_small)
        x_mat2 = [[prox(self.B_small[i,j]) for j in range(self.B_small.shape[1])] for i in range(self.B_small.shape[0])]
        x_mat2 = np.array(x_mat2)
        self.assertItemsAlmostEqual(x_mat1, x_mat2, places = places)

        x_mat1 = prox(self.B_small, t = self.t)
        x_mat2 = [[prox(self.B_small[i,j], t = self.t) for j in range(self.B_small.shape[1])] \
                    for i in range(self.B_small.shape[0])]
        x_mat2 = np.array(x_mat2)
        self.assertItemsAlmostEqual(x_mat1, x_mat2, places = places)

        offset = np.random.randn(*self.B_small.shape)
        lin_term = np.random.randn(*self.B_small.shape)
        x_mat1 = prox(self.B_small, t = self.t, scale = 0.5, offset = offset, lin_term = lin_term, quad_term = 2.5)
        x_mat2 = [[prox(self.B_small[i,j], t = self.t, scale = 0.5, offset = offset[i,j], lin_term = lin_term[i,j], \
                        quad_term = 2.5) for j in range(self.B_small.shape[1])] for i in range(self.B_small.shape[0])]
        x_mat2 = np.array(x_mat2)
        self.assertItemsAlmostEqual(x_mat1, x_mat2, places = places)

    def check_sparsity(self, prox, places = 4, check_vector = True, check_matrix = True, matrix_type = "general"):
        if check_vector:
            # Vector input.
            x_vec1 = prox(self.u_sparse)
            x_vec2 = prox(self.u_dense)
            self.assertTrue(sparse.issparse(x_vec1))
            self.assertItemsAlmostEqual(x_vec1.todense(), x_vec2, places = places)

            x_vec1 = prox(self.u_sparse, t=self.t)
            x_vec2 = prox(self.u_dense, t=self.t)
            self.assertTrue(sparse.issparse(x_vec1))
            self.assertItemsAlmostEqual(x_vec1.todense(), x_vec2, places = places)

            offset = sparse.random(*self.u_sparse.shape)
            lin_term = sparse.random(*self.u_sparse.shape)
            x_vec1 = prox(self.u_sparse, t=self.t, scale=0.5, offset=offset, lin_term=lin_term, quad_term=2.5)
            x_vec2 = prox(self.u_dense, t=self.t, scale=0.5, offset=offset, lin_term=lin_term, quad_term=2.5)
            self.assertTrue(sparse.issparse(x_vec1))
            self.assertItemsAlmostEqual(x_vec1.todense(), x_vec2, places = places)

        if check_matrix:
            if matrix_type == "general":
                C_sparse = self.C_sparse
                C_dense = self.C_dense
            elif matrix_type == "square":
                C_sparse = self.C_square_sparse
                C_dense = self.C_square_dense
            else:
                raise ValueError("matrix_type must be 'general' or 'square'")

            # Matrix input.
            x_mat1 = prox(C_sparse)
            x_mat2 = prox(C_dense)
            self.assertTrue(sparse.issparse(x_mat1))
            self.assertItemsAlmostEqual(x_mat1.todense(), x_mat2, places = places)

            x_mat1 = prox(C_sparse, t=self.t)
            x_mat2 = prox(C_dense, t=self.t)
            self.assertTrue(sparse.issparse(x_mat1))
            self.assertItemsAlmostEqual(x_mat1.todense(), x_mat2, places = places)

            offset = sparse.random(*C_sparse.shape)
            lin_term = sparse.random(*C_sparse.shape)
            x_mat1 = prox(C_sparse, t=self.t, scale=0.5, offset=offset, lin_term=lin_term, quad_term=2.5)
            x_mat2 = prox(C_dense, t=self.t, scale=0.5, offset=offset, lin_term=lin_term, quad_term=2.5)
            self.assertTrue(sparse.issparse(x_mat1))
            self.assertItemsAlmostEqual(x_mat1.todense(), x_mat2, places = places)

    def test_box_constr(self):
        # Projection onto a random interval.
        lo = np.random.randn()
        hi = lo + 5*np.abs(np.random.randn())

        x_a2dr = prox_box_constr(self.v, self.t, v_lo = lo, v_hi = hi)
        self.assertTrue(np.all(lo - self.TOLERANCE <= x_a2dr) and np.all(x_a2dr <= hi + self.TOLERANCE))

        # Projection onto a random interval with affine composition.
        scale = 2 * np.abs(np.random.randn()) + self.TOLERANCE
        if np.random.rand() < 0.5:
            scale = -scale
        offset = np.random.randn(*self.v.shape)
        lin_term = np.random.randn(*self.v.shape)
        quad_term = np.abs(np.random.randn())
        x_a2dr = prox_box_constr(self.v, self.t, v_lo = lo, v_hi = hi, scale = scale, offset = offset, \
                                    lin_term = lin_term, quad_term = quad_term)
        x_scaled = scale*x_a2dr - offset
        self.assertTrue(np.all(lo - self.TOLERANCE <= x_scaled) and np.all(x_scaled <= hi + self.TOLERANCE))

        # Common box intervals.
        bounds = [(0, 0), (-1, 1), (0, np.inf), (-np.inf, 0)]
        for bound in bounds:
            lo, hi = bound
            # Elementwise consistency tests.
            self.check_elementwise(lambda v, *args, **kwargs: prox_box_constr(v, v_lo = lo, v_hi = hi, *args, **kwargs))

            # Sparsity consistency tests.
            self.check_sparsity(lambda v, *args, **kwargs: prox_box_constr(v, v_lo = lo, v_hi = hi, *args, **kwargs))

            # General composition tests.
            self.check_composition(lambda v, *args, **kwargs: prox_box_constr(v, v_lo = lo, v_hi = hi, *args, **kwargs),
                                   lambda x: 0, self.v, constr_fun = lambda x: [lo <= x, x <= hi])
            self.check_composition(lambda v, *args, **kwargs: prox_box_constr(v, v_lo = lo, v_hi = hi, *args, **kwargs),
                                   lambda x: 0, self.B, constr_fun = lambda x: [lo <= x, x <= hi])

        # Optimal control term: f(x) = I(||x||_{\infty} <= 1) = I(-1 <= x <= 1).
        x_a2dr = prox_box_constr(self.v, self.t, v_lo = -1, v_hi = 1)
        x_cvxpy = self.prox_cvxpy(self.v, lambda x: 0, constr_fun = lambda x: [norm_inf(x) <= 1], t = self.t)
        self.assertTrue(np.all(-1 - self.TOLERANCE <= x_a2dr) and np.all(x_a2dr <= 1 + self.TOLERANCE))
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 3)

    def test_nonneg_constr(self):
        x_a2dr = prox_nonneg_constr(self.v, self.t, scale = -2, offset = 0)
        self.assertTrue(np.all(-2*x_a2dr >= -self.TOLERANCE))

        scale = 2*np.abs(np.random.randn()) + self.TOLERANCE
        if np.random.rand() < 0.5:
            scale = -scale
        offset = np.random.randn(*self.v.shape)
        lin_term = np.random.randn(*self.v.shape)
        quad_term = np.abs(np.random.randn())

        x_a2dr = prox_nonneg_constr(self.v, self.t, scale = scale, offset = offset, lin_term = lin_term, \
                                    quad_term = quad_term)
        self.assertTrue(np.all(scale*x_a2dr - offset) >= -self.TOLERANCE)

        # Elementwise consistency tests.
        self.check_elementwise(prox_nonneg_constr)

        # Sparsity consistency tests.
        self.check_sparsity(prox_nonneg_constr)

        # General composition tests.
        self.check_composition(prox_nonneg_constr, lambda x: 0, self.v, constr_fun = lambda x: [x >= 0])
        self.check_composition(prox_nonneg_constr, lambda x: 0, self.B, constr_fun = lambda x: [x >= 0])

        # Non-negative least squares term: f(x) = I(x >= 0).
        x_a2dr = prox_nonneg_constr(self.v, self.t)
        x_cvxpy = self.prox_cvxpy(self.v, lambda x: 0, constr_fun = lambda x: [x >= 0], t = self.t)
        self.assertTrue(np.all(x_a2dr >= -self.TOLERANCE))
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 3)

    def test_nonpos_constr(self):
        x_a2dr = prox_nonpos_constr(self.v, self.t)
        self.assertTrue(np.all(x_a2dr <= self.TOLERANCE))

        x_a2dr = prox_nonpos_constr(self.v, self.t, scale=-2, offset=0)
        self.assertTrue(np.all(-2*x_a2dr <= self.TOLERANCE))

        scale = 2 * np.abs(np.random.randn()) + self.TOLERANCE
        if np.random.rand() < 0.5:
            scale = -scale
        offset = np.random.randn(*self.v.shape)
        lin_term = np.random.randn(*self.v.shape)
        quad_term = np.abs(np.random.randn())

        x_a2dr = prox_nonpos_constr(self.v, self.t, scale=scale, offset=offset, lin_term=lin_term, \
                                    quad_term=quad_term)
        self.assertTrue(np.all(scale * x_a2dr - offset) <= self.TOLERANCE)

        # Elementwise consistency tests.
        self.check_elementwise(prox_nonpos_constr)

        # Sparsity consistency tests.
        self.check_sparsity(prox_nonpos_constr)

        # General composition tests.
        self.check_composition(prox_nonpos_constr, lambda x: 0, self.v, constr_fun=lambda x: [x <= 0])
        self.check_composition(prox_nonpos_constr, lambda x: 0, self.B, constr_fun=lambda x: [x <= 0])

    def test_psd_cone(self):
        # Projection onto the PSD cone.
        B_a2dr = prox_psd_cone(self.B_symm, self.t)
        self.assertTrue(np.all(np.linalg.eigvals(B_a2dr) >= -self.TOLERANCE))

        # Projection onto the PSD cone with affine composition.
        scale = 2 * np.abs(np.random.randn()) + self.TOLERANCE
        if np.random.rand() < 0.5:
            scale = -scale
        offset = np.random.randn(*self.B_symm.shape)
        lin_term = np.random.randn(*self.B_symm.shape)
        quad_term = np.abs(np.random.randn())
        B_a2dr = prox_psd_cone(self.B_symm, self.t, scale = scale, offset = offset, lin_term = lin_term, \
                                  quad_term = quad_term)
        B_scaled = scale*B_a2dr - offset
        self.assertTrue(np.all(np.linalg.eigvals(B_scaled) >= -self.TOLERANCE))

        # Simple composition.
        B_a2dr = prox_psd_cone(self.B_symm, t = self.t, scale = 2, offset = 0.5, lin_term = 1.5, quad_term = 2.5)
        B_cvxpy = self.prox_cvxpy(self.B_symm, lambda X: 0, constr_fun = lambda X: [X >> 0], t = self.t, scale = 2, \
                                  offset = 0.5, lin_term = 1.5, quad_term = 2.5)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy)

    def test_soc(self):
        # Projection onto the SOC.
        x_a2dr = prox_soc(self.v, self.t)
        self.assertTrue(np.linalg.norm(x_a2dr[:-1],2) <= x_a2dr[-1] + self.TOLERANCE)

        # Projection onto the SOC with affine composition.
        x_a2dr = prox_soc(self.v, self.t, scale=2, offset=0.5)
        x_scaled = 2*x_a2dr - 0.5
        self.assertTrue(np.linalg.norm(x_scaled[:-1],2) <= x_scaled[-1] + self.TOLERANCE)

        scale = 2 * np.abs(np.random.randn()) + self.TOLERANCE
        if np.random.rand() < 0.5:
            scale = -scale
        offset = np.random.randn(*self.v.shape)
        lin_term = np.random.randn(*self.v.shape)
        quad_term = np.abs(np.random.randn())

        x_a2dr = prox_soc(self.v, self.t, scale=scale, offset=offset, lin_term=lin_term, quad_term=quad_term)
        x_scaled = scale*x_a2dr - offset
        self.assertTrue(np.linalg.norm(x_scaled[:-1], 2) <= x_scaled[-1] + self.TOLERANCE)

        # Sparsity consistency tests.
        self.check_sparsity(prox_soc, check_matrix = False)

        # General composition tests.
        self.check_composition(prox_soc, lambda x: 0, self.v, constr_fun = lambda x: [SOC(x[-1], x[:-1])], \
                               solver = "SCS")

    def test_abs(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_abs)

        # Elementwise consistency tests.
        self.check_elementwise(prox_abs)

        # General composition tests.
        self.check_composition(prox_abs, cvxpy.abs, self.c)
        self.check_composition(prox_abs, lambda x: sum(abs(x)), self.v)
        self.check_composition(prox_abs, lambda x: sum(abs(x)), self.B)

    def test_constant(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_constant)

        # Sparsity consistency tests.
        self.check_sparsity(prox_constant)

        # General composition tests.
        self.check_composition(prox_constant, lambda x: 0, self.c)
        self.check_composition(prox_constant, lambda x: 0, self.v)
        self.check_composition(prox_constant, lambda x: 0, self.B)

    def test_exp(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_exp)

        # General composition tests.
        self.check_composition(prox_exp, cvxpy.exp, self.c)
        self.check_composition(prox_exp, lambda x: sum(exp(x)), self.v, solver = "SCS")
        self.check_composition(prox_exp, lambda x: sum(exp(x)), self.B, places=2, solver = "SCS")

    def test_huber(self):
        for M in [0, 0.5, 1, 2]:
            # Elementwise consistency tests.
            self.check_elementwise(lambda v, *args, **kwargs: prox_huber(v, *args, **kwargs, M = M))

            # Sparsity consistency tests.
            self.check_sparsity(lambda v, *args, **kwargs: prox_huber(v, *args, **kwargs, M = M))

            # Scalar input.
            self.check_composition(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: huber(x, M = M), self.c)
            # Vector input.
            self.check_composition(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: sum(huber(x, M = M)), self.v)
            # Matrix input.
            self.check_composition(lambda v, *args, **kwargs: prox_huber(v, M = M, *args, **kwargs),
                                   lambda x: sum(huber(x, M = M)), self.B)

    def test_identity(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_identity)

        # General composition tests.
        self.check_composition(prox_identity, lambda x: x, self.c)
        self.check_composition(prox_identity, lambda x: sum(x), self.v)
        self.check_composition(prox_identity, lambda x: sum(x), self.B)

    def test_logistic(self):
        # General composition tests.
        self.check_composition(prox_logistic, lambda x: logistic(x), self.c, solver='ECOS')
        self.check_composition(prox_logistic, lambda x: sum(logistic(x)), self.v, solver = 'SCS')
        self.check_composition(prox_logistic, lambda x: sum(logistic(x)), self.B, solver = "SCS")

        # Simple logistic function: f(x) = \sum_i log(1 + exp(-y_i*x_i)).
        y = np.random.randn(*self.v.shape)
        self.check_composition(lambda v, *args, **kwargs: prox_logistic(v, y = y, *args, **kwargs),
                               lambda x: sum(logistic(-multiply(y,x))), self.v, places = 2, solver = "SCS")

        # Multi-task logistic regression term: f(B) = \sum_i log(1 + exp(-Y_{ij}*B_{ij}).
        Y_mat = np.random.randn(*self.B.shape)
        B_a2dr = prox_logistic(self.B, t = self.t, y = Y_mat)
        B_cvxpy = self.prox_cvxpy(self.B, lambda B: sum(logistic(-multiply(Y_mat,B))), t = self.t, solver = "SCS")
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 2)

        self.check_composition(lambda v, *args, **kwargs: prox_logistic(v, y = Y_mat, *args, **kwargs),
                               lambda B: sum(logistic(-multiply(Y_mat,B))), self.B, places = 2, solver = "SCS")

    def test_pos(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_pos)

        # Sparsity consistency tests.
        self.check_sparsity(prox_pos)

        # General composition tests.
        self.check_composition(prox_pos, cvxpy.pos, self.c)
        self.check_composition(prox_pos, lambda x: sum(pos(x)), self.v)
        self.check_composition(prox_pos, lambda x: sum(pos(x)), self.B)

    def test_neg(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_neg)

        # Sparsity consistency tests.
        self.check_sparsity(prox_neg)

        # General composition tests.
        self.check_composition(prox_neg, cvxpy.neg, self.c)
        self.check_composition(prox_neg, lambda x: sum(neg(x)), self.v)
        self.check_composition(prox_neg, lambda x: sum(neg(x)), self.B)

    def test_neg_entr(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_neg_entr)

        # General composition tests.
        self.check_composition(prox_neg_entr, lambda x: -entr(x), self.c)
        self.check_composition(prox_neg_entr, lambda x: sum(-entr(x)), self.v)
        self.check_composition(prox_neg_entr, lambda x: sum(-entr(x)), self.B, places=2, solver="SCS")

    def test_neg_log(self):
        # Elementwise consistency tests.
        self.check_elementwise(prox_neg_log)

        # General composition tests.
        self.check_composition(prox_neg_log, lambda x: -log(x), self.c, solver='SCS')
        self.check_composition(prox_neg_log, lambda x: sum(-log(x)), self.v, solver='SCS')
        self.check_composition(prox_neg_log, lambda x: sum(-log(x)), self.B, places=2, solver="SCS")

    def test_neg_log_det(self):
        # TODO: Poor accuracy in compositions.
        # General composition tests.
        # self.check_composition(prox_neg_log_det, lambda X: -log_det(X), self.B_symm, places=2, solver="SCS")
        # self.check_composition(prox_neg_log_det, lambda X: -log_det(X), self.B_psd, places=2, solver="SCS")

        # Sparse inverse covariance estimation term: f(B) = -log(det(B)) for symmetric positive definite B.
        B_spd = self.B_psd + np.eye(self.B_psd.shape[0])
        B_a2dr = prox_neg_log_det(B_spd, self.t)
        B_cvxpy = self.prox_cvxpy(B_spd, lambda X: -log_det(X), t=self.t, solver="SCS")
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places=2)

        # Sparse inverse covariance estimation term: f(B) = -log(det(B)) + tr(BQ) for symmetric positive definite B
        # and given matrix Q.
        # Q = np.random.randn(*B_spd.shape)
        # B_a2dr = prox_neg_log_det(B_spd, self.t, lin_term = Q.T)   # tr(BQ) = \sum_{ij} Q_{ji}B_{ij}
        # B_cvxpy = self.prox_cvxpy(B_spd, lambda X: -log_det(X) + trace(X*Q), t=self.t, solver="SCS")
        # self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places=2)

    def test_max(self):
        # General composition tests.
        self.check_composition(prox_max, cvxpy.max, self.c)
        self.check_composition(prox_max, cvxpy.max, self.v)
        self.check_composition(prox_max, cvxpy.max, self.B, solver = "SCS")

    def test_norm1(self):
        # Sparsity consistency tests.
        self.check_sparsity(prox_norm1)

        # General composition tests.
        self.check_composition(prox_norm1, norm1, self.c)
        self.check_composition(prox_norm1, norm1, self.v)
        self.check_composition(prox_norm1, norm1, self.B)

        # l1 trend filtering term: f(x) = \alpha*||x||_1.
        alpha = 0.5 + np.abs(np.random.randn())
        x_a2dr = prox_norm1(self.v, t = alpha*self.t)
        x_cvxpy = self.prox_cvxpy(self.v, norm1, t = alpha*self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        # Sparse inverse covariance estimation term: f(B) = \alpha*||B||_1.
        B_symm_a2dr = prox_norm1(self.B_symm, t = alpha*self.t)
        B_symm_cvxpy = self.prox_cvxpy(self.B_symm, norm1, t = alpha*self.t)
        self.assertItemsAlmostEqual(B_symm_a2dr, B_symm_cvxpy, places = 4)

    def test_norm2(self):
        # Sparsity consistency tests.
        self.check_sparsity(prox_norm2)

        # General composition tests.
        self.check_composition(prox_norm2, norm2, np.random.randn())
        self.check_composition(prox_norm2, norm2, self.v, solver ="SCS")
        self.check_composition(prox_norm2, lambda B: cvxpy.norm(B, 'fro'), self.B, solver ="SCS")

        # f(x) = \alpha*||x||_2
        alpha = 0.5 + np.abs(np.random.randn())
        x_a2dr = prox_norm2(self.v, t = alpha*self.t)
        x_cvxpy = self.prox_cvxpy(self.v, norm2, t = alpha*self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_norm_inf(self):
        # General composition tests.
        self.check_composition(prox_norm_inf, norm_inf, self.c)
        self.check_composition(prox_norm_inf, norm_inf, self.v)
        self.check_composition(prox_norm_inf, norm_inf, self.B, solver='ECOS')

        # f(x) = \alpha*||x||_{\infty}
        alpha = 0.5 + np.abs(np.random.randn())
        x_a2dr = prox_norm_inf(self.v, t = alpha*self.t)
        x_cvxpy = self.prox_cvxpy(self.v, norm_inf, t = alpha*self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_norm_fro(self):
        # General composition tests.
        self.check_composition(prox_norm_fro, lambda X: cvxpy.norm(X,'fro'), self.B, solver='SCS')

    def test_norm_nuc(self):
        # General composition tests.
        self.check_composition(prox_norm_nuc, normNuc, self.B, solver='SCS')

        # Multi-task logistic regression term: f(B) = \beta*||B||_*.
        beta = 1.5 + np.abs(np.random.randn())
        B_a2dr = prox_norm_nuc(self.B, t = beta*self.t)
        B_cvxpy = self.prox_cvxpy(self.B, normNuc, t = beta*self.t, solver='SCS')
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

    def test_group_lasso(self):
        # Sparsity consistency tests.
        self.check_sparsity(prox_group_lasso)

        # General composition tests.
        groupLasso = lambda B: sum([norm2(B[:,j]) for j in range(B.shape[1])])
        self.check_composition(prox_group_lasso, groupLasso, self.B, solver="SCS")

        # Multi-task logistic regression term: f(B) = \alpha*||B||_{2,1}.
        alpha = 1.5 + np.abs(np.random.randn())
        B_a2dr = prox_group_lasso(self.B, t = alpha*self.t)
        B_cvxpy = self.prox_cvxpy(self.B, groupLasso, t = alpha*self.t)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 3)

        # Compare with taking l2-norm separately on each column.
        B_norm2 = [prox_norm2(self.B[:,j], t = alpha*self.t) for j in range(self.B.shape[1])]
        B_norm2 = np.vstack(B_norm2)
        self.assertItemsAlmostEqual(B_a2dr, B_norm2, places = 3)

    def test_sigma_max(self):
        # General composition tests.
        self.check_composition(prox_sigma_max, sigma_max, self.B, solver='SCS')

    def test_sum_squares(self):
        # Sparsity consistency tests.
        self.check_sparsity(prox_sum_squares)

        # General composition tests.
        self.check_composition(prox_sum_squares, sum_squares, self.v)
        self.check_composition(prox_sum_squares, sum_squares, self.B)

        # Optimal control term: f(x) = ||x||_2^2.
        x_a2dr = prox_sum_squares(self.v, t = self.t)
        x_cvxpy = self.prox_cvxpy(self.v, sum_squares, t = self.t)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        # l1 trend filtering term: f(x) = (1/2)*||x - y||_2^2 for given y.
        y = np.random.randn(*self.v.shape)
        x_a2dr = prox_sum_squares(self.v, t = 0.5*self.t, offset = y)
        x_cvxpy = self.prox_cvxpy(self.v, sum_squares, t = 0.5*self.t, offset = y)
        self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

    def test_sum_squares_affine(self):
        # Scalar terms.
        F = np.random.randn()
        g = np.random.randn()
        v = np.random.randn()

        self.check_composition(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method ="lsqr",
                                                                                  *args, **kwargs), lambda x: sum_squares(F*x - g), v)
        self.check_composition(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method ="lstsq",
                                                                                  *args, **kwargs), lambda x: sum_squares(F*x - g), v)

        # Simple sum of squares: f(x) = ||x||_2^2.
        n = 100
        F = np.eye(n)
        g = np.zeros(n)
        v = np.random.randn(n)
        F_sparse = sparse.eye(n)
        g_sparse = sparse.csr_matrix((n,1))

        for method in ["lsqr", "lstsq"]:
            self.check_composition(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, \
                                        method = method, *args, **kwargs), lambda x: sum_squares(F*x - g), v)
            self.check_composition(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F_sparse, g = g_sparse, \
                                        method = method, *args, **kwargs), lambda x: sum_squares(F*x - g), v)

        # Non-negative least squares term: f(x) = ||Fx - g||_2^2.
        m = 1000
        n = 100
        F = 10 + 5*np.random.randn(m,n)
        x = 2*np.random.randn(n)
        g = F.dot(x) + 0.01*np.random.randn(m)
        v = np.random.randn(n)

        for method in ["lsqr", "lstsq"]:
            x_a2dr = prox_sum_squares_affine(self.v, t = self.t, F = F, g = g, method = method)
            x_cvxpy = self.prox_cvxpy(self.v, lambda x: sum_squares(F*x - g), t = self.t)
            self.assertItemsAlmostEqual(x_a2dr, x_cvxpy, places = 4)

        # General composition tests.
        self.check_composition(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method ="lsqr",
                                                                                  *args, **kwargs), lambda x: sum_squares(F*x - g), v)
        self.check_composition(lambda v, *args, **kwargs: prox_sum_squares_affine(v, F = F, g = g, method ="lstsq",
                                                                                  *args, **kwargs), lambda x: sum_squares(F*x - g), v)

    def test_quad_form(self):
        # Simple quadratic.
        v = np.random.randn(1)
        Q = np.array([[5]])
        self.check_composition(lambda v, *args, **kwargs: prox_quad_form(v, Q = Q, method = "lsqr", *args, **kwargs),
                               lambda x: quad_form(x, P = Q), v)
        self.check_composition(lambda v, *args, **kwargs: prox_quad_form(v, Q = Q, method = "lstsq", *args, **kwargs),
                               lambda x: quad_form(x, P = Q), v)

        # General composition tests.
        n = 10
        v = np.random.randn(n)
        Q = np.random.randn(n,n)
        Q = Q.T.dot(Q) + 0.5*np.eye(n)
        self.check_composition(lambda v, *args, **kwargs: prox_quad_form(v, Q = Q, method = "lsqr", *args, **kwargs),
                               lambda x: quad_form(x, P = Q), v)
        self.check_composition(lambda v, *args, **kwargs: prox_quad_form(v, Q = Q, method = "lstsq", *args, **kwargs),
                               lambda x: quad_form(x, P = Q), v)

    def test_trace(self):
        # Sparsity consistency tests.
        C = sparse.random(*self.C_square_sparse.shape)
        self.check_sparsity(prox_trace, check_vector = False, matrix_type = "square")
        self.check_sparsity(lambda B, *args, **kwargs: prox_trace(B, C = C, *args, **kwargs), check_vector = False, \
                            matrix_type = "square")

        # General composition tests.
        C = np.random.randn(*self.B.shape)
        self.check_composition(prox_trace, cvxpy.trace, self.B_square)
        self.check_composition(lambda B, *args, **kwargs: prox_trace(B, C = C, *args, **kwargs),
                               lambda X: cvxpy.trace(C.T * X), self.B)

        # Sparse inverse covariance estimation term: f(B) = tr(BQ) for given symmetric positive semidefinite Q.
        Q = np.random.randn(*self.B_square.shape)
        Q = Q.T.dot(Q)
        B_a2dr = prox_trace(self.B_square, t = self.t, C = Q.T)   # tr(BQ) = tr(QB) = tr((Q^T)^TB).
        B_cvxpy = self.prox_cvxpy(self.B_square, lambda X: cvxpy.trace(X * Q), t = self.t)
        self.assertItemsAlmostEqual(B_a2dr, B_cvxpy, places = 4)
