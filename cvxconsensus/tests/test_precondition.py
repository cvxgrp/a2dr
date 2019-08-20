"""
Copyright 2018 Anqi Fu

This file is part of CVXConsensus.

CVXConsensus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXConsensus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXConsensus. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cvxconsensus import a2dr
from cvxconsensus.precondition import mat_equil
from cvxconsensus.tests.base_test import BaseTest

def prox_sum_squares(X, y, type = "lsqr"):
    n = X.shape[1]
    if type == "lsqr":
        X = sp.sparse.csc_matrix(X)
        def prox(v, rho):
            A = sp.sparse.vstack((X, np.sqrt(rho/2)*sp.sparse.eye(n)))
            b = np.concatenate((y, np.sqrt(rho/2)*v))
            return sp.sparse.linalg.lsqr(A, b, atol=1e-16, btol=1e-16)[0]
    elif type == "lstsq":
        def prox(v, rho):
           A = np.vstack((X, np.sqrt(rho/2)*np.eye(n)))
           b = np.concatenate((y, np.sqrt(rho/2)*v))
           return np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError("Algorithm type not supported:", type)
    return prox

class TestPrecondition(BaseTest):
	"""Unit tests for preconditioning data before S-DRS"""
	
	def setUp(self):
		np.random.seed(1)
		self.eps_rel = 1e-8
		self.eps_abs = 1e-6
		self.MAX_ITER = 2000

	def test_mat_equil(self):
		m = 3000
		N = 500
		max_size = 20  # maximum size of each block
		n_list = [np.random.randint(max_size + 1) for i in range(N)]  # list of variable block sizes n_i
		n = np.sum(n_list)  # total variable dimension = n_1 + ... + n_N
		A = np.random.randn(m, n)
		tol = 1e-3  # tolerance for terminating the equilibration
		max_iter = 10000  # maximum number of iterations for terminating the equilibration

		d, e, B, k = mat_equil(A, n_list, tol, max_iter)

		print('[Sanity Check]')
		print('len(d) = {}, len(e) = {}, iter number = {}'.format(len(d), len(e), k))
		print('mean(d) = {}, mean(e) = {}'.format(np.mean(d), np.mean(e)))
		print('\|A\|_2 = {}, \|DAE\|_2 = {}'.format(np.linalg.norm(A), np.linalg.norm(B)))
		print('min(|A|) = {}, max(|A|) = {}, mean(|A|) = {}'.format(np.min(np.abs(A)),
																	np.max(np.abs(A)), np.average(np.abs(A))))
		print('min(|B|) = {}, max(|B|) = {}, mean(|B|) = {}'.format(np.min(np.abs(B)),
																	np.max(np.abs(B)), np.average(np.abs(B))))

		# Row norms.
		A_norms_r = np.sqrt((A ** 2).dot(np.ones(n)))
		B_norms_r = np.sqrt((B ** 2).dot(np.ones(n)))
		# scale_r = np.mean(A_norms_r) / np.mean(B_norms_r)
		# A_norms_r = A_norms_r / scale_r

		# Column norms.
		A_norms_c = np.sqrt(np.ones(m).dot(A ** 2))
		B_norms_c = np.sqrt(np.ones(m).dot(B ** 2))
		# scale_c = np.mean(A_norms_c) / np.mean(B_norms_c)
		# A_norms_c = A_norms_c / scale_c

		# Visualization of row norms.
		plt.plot(A_norms_r)
		plt.plot(B_norms_r)
		plt.title('Row Norms Before and After Equilibration')
		plt.legend(['Before', 'After'])
		plt.show()

		# Visualization of column norms.
		plt.plot(A_norms_c)
		plt.plot(B_norms_c)
		plt.title('Column Norms Before and After Equilibration')
		plt.legend(['Before', 'After'])
		plt.show()

		# Visualization of left scaling d.
		plt.plot(d)
		plt.title('d: min = {:.3}, max = {:.3}, mean = {:.3}'.format(np.min(d), np.max(d), np.average(d)))
		plt.show()

		# Visualization of right scaling e.
		plt.plot(e)
		plt.title('e: min = {:.3}, max = {:.3}, mean = {:.3}'.format(np.min(e), np.max(e), np.average(e)))
		plt.show()

	def test_nnls(self):
		# Solve the non-negative least squares problem
		# Minimize (1/2)*||A*x - b||_2^2 subject to x >= 0.
		m = 100
		n = 10
		N = 4   # Number of nodes.

		# Problem data.
		mu = 100
		sigma = 10
		X = mu + sigma*np.random.randn(m,n)
		y = mu + sigma*np.random.randn(m)

		# Solve with SciPy.
		sp_result = sp.optimize.nnls(X, y)
		sp_beta = sp_result[0]
		sp_obj = sp_result[1] ** 2  # SciPy objective is ||y - X\beta||_2.
		print("Scipy Objective:", sp_obj)
		print("SciPy Solution:", sp_beta)

		X_split = np.split(X, N)
		y_split = np.split(y, N)
		p_list = [prox_sum_squares(X_sub, y_sub) for X_sub, y_sub in zip(X_split, y_split)]
		p_list += [lambda u, rho: np.maximum(u, 0)]   # Projection onto non-negative orthant.
		v_init = (N + 1) * [np.random.randn(n)]
		A_list = np.hsplit(np.eye(N*n), N) + [-np.vstack(N*(np.eye(n),))]
		b = np.zeros(N*n)

		# Solve with A2DR.
		a2dr_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
						   eps_rel=self.eps_rel, anderson=True, precond=False)
		a2dr_beta = a2dr_result["x_vals"][-1]
		a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
		print("A2DR Objective:", a2dr_obj)
		print("A2DR Solution:", a2dr_beta)
		# self.assertAlmostEqual(sp_obj, a2dr_obj)
		# self.assertItemsAlmostEqual(sp_beta, a2dr_beta, places=3)
		self.plot_residuals(a2dr_result["primal"], a2dr_result["dual"], normalize=True, title="A2DR Residuals", \
							semilogy=True)

		# Solve with preconditioned A2DR.
		cond_result = a2dr(p_list, v_init, A_list, b, max_iter=self.MAX_ITER, eps_abs=self.eps_abs, \
						   eps_rel=self.eps_rel, anderson=True, precond=True)
		cond_beta = cond_result["x_vals"][-1]
		cond_obj = np.sum((y - X.dot(cond_beta))**2)
		print("Preconditioned A2DR Objective:", cond_obj)
		print("Preconditioned A2DR Solution:", cond_beta)
		# self.assertAlmostEqual(sp_obj, cond_obj)
		# self.assertItemsAlmostEqual(sp_beta, cond_beta, places=3)
		self.plot_residuals(cond_result["primal"], cond_result["dual"], normalize=True, \
							title="Preconditioned A2DR Residuals", semilogy=True)
