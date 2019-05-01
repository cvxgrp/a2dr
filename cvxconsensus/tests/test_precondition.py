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
import matplotlib.pyplot as plt
from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import *
from cvxconsensus import Problems
from cvxconsensus.precondition import mat_equil
from cvxconsensus.tests.base_test import BaseTest

def standardize(x, center = True, scale = True, axis = 0):
	mu = np.mean(x, axis = axis)
	sigma = np.std(x, axis = axis)
	if center:
		x = x - mu
	if scale:
		x = x/sigma
	return x

def probs_nnls(A_split, b_split, center = True, scale = True, axis = 0):
	# Minimize \sum_i f_i(x) subject to x >= 0
	# where f_i(x) = ||A_ix - b_i||_2^2 for subproblem i = 1,...,N.
	n = A_split[0].shape[1]
	x = Variable(n)
	constr = [x >= 0]
	
	p_list = []
	for A_sub, b_sub in zip(A_split, b_split):
		# Standardize data locally.
		A_sub = standardize(A_sub, center, scale, axis)
		b_sub = standardize(b_sub, center, scale, axis)
		
		obj = sum_squares(A_sub*x - b_sub)
		# p_list += [Problem(Minimize(obj), constr)]
		p_list += [Problem(Minimize(obj))]
	p_list += [Problem(Minimize(0), constr)]
	probs = Problems(p_list)
	# probs.pretty_vars()
	return probs

class TestPrecondition(BaseTest):
	"""Unit tests for preconditioning data before S-DRS"""
	
	def setUp(self):
		np.random.seed(1)
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
		# Minimize (1/2) ||Ax - b||_2^2 subject to x >= 0.
		m = 50
		n = 100
		N = 5   # Number of nodes.
		rho = 10   # Step size.
		
		# Problem data.
		mu = 100
		sigma = 10
		A = mu + sigma*np.random.randn(m,n)
		b = mu + sigma*np.random.randn(m)
		A_split = np.split(A, N)
		b_split = np.split(b, N)
		
		# Solve without standardization.
		probs = probs_nnls(A_split, b_split, center = False, scale = False)
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER)
		res = probs.residuals
		
		# Solve with centering only.
		probs = probs_nnls(A_split, b_split, center = False, scale = True)
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER)
		res_cnt = probs.residuals
		
		# Solve with centering and scaling.
		probs = probs_nnls(A_split, b_split, center = True, scale = True)
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER)
		res_std = probs.residuals
		
		# Plot and compare residuals.
		plt.semilogy(range(res.shape[0]), res, label = "Original")
		plt.semilogy(range(res_cnt.shape[0]), res_cnt, label = "Scaled")
		plt.semilogy(range(res_std.shape[0]), res_std, label = "Centered and Scaled")
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Residual")
		plt.show()
