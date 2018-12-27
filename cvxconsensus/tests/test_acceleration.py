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
from scipy.linalg import toeplitz
from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import *
import cvxconsensus
from cvxconsensus import Problems
from cvxconsensus.tests.base_test import BaseTest

def compare_residuals(res_sdrs, res_aa2, m_vals):
	if not isinstance(res_aa2, list):
		res_aa2 = [res_aa2]
	if not isinstance(m_vals, list):
		m_vals = [m_vals]
	if len(m_vals) != len(res_aa2):
		raise ValueError("Must have same number of AA-II residuals as memory parameter values")
		
	plt.semilogy(range(res_sdrs.shape[0]), res_sdrs, label = "S-DRS")
	for i in range(len(m_vals)):
		label = "AA-II S-DRS (m = {})".format(m_vals[i])
		plt.semilogy(range(res_aa2[i].shape[0]), res_aa2[i], linestyle = "--", label = label)
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Residual")
	plt.show()
	
class TestAcceleration(BaseTest):
	"""Unit tests for Anderson acceleration of consensus ADMM"""
	
	def setUp(self):
		np.random.seed(1)
		self.MAX_ITER = 2000
	
	def test_lasso(self):
		m = 100
		n = 10
		DENSITY = 0.75
		m_accel = 5
		
		# Problem data.
		A = np.random.randn(m,n)
		xtrue = np.random.randn(n)
		idxs = np.random.choice(range(n), int((1-DENSITY)*n), replace = False)
		for idx in idxs:
			xtrue[idx] = 0
		b = A.dot(xtrue) + np.random.randn(m)
		
		# List of all problems with objective f_i.
		x = Variable(n)
		p_list = [Problem(Minimize(sum_squares(A*x-b))),
				  Problem(Minimize(norm(x,1)))]
		probs = Problems(p_list)
		N = len(p_list)
		
		# Solve with consensus ADMM.
		obj_sdrs = probs.solve(method = "consensus", rho_init = 1.0, max_iter = self.MAX_ITER)
		res_sdrs = probs.residuals
		
		# Solve with consensus ADMM using Anderson acceleration.
		obj_aa2 = probs.solve(method = "consensus", rho_init = 1.0, \
							   max_iter = self.MAX_ITER, anderson = True, m_accel = m_accel)
		res_aa2 = probs.residuals
		x_aa2 = [x.value for x in probs.variables()]
		compare_residuals(res_sdrs, [res_aa2], [m_accel])
		
		# Solve combined problem.
		obj_comb = probs.solve(method = "combined")
		x_comb = [x.value for x in probs.variables()]
		
		# Compare results.
		N = len(probs.variables())
		self.assertAlmostEqual(obj_aa2, obj_comb)
		for i in range(N):
			self.assertItemsAlmostEqual(x_aa2[i], x_comb[i])
	
	def test_nnls(self):
		# Solve the non-negative least squares problem
		# Minimize (1/2) ||Ax - b||_2^2 subject to x >= 0.
		m = 50
		n = 100
		N = 5   # Number of nodes.
		m_accel = [2, 4, 8, 10]
		
		# Problem data.
		A = np.random.randn(m,n)
		b = np.random.randn(m)
		A_split = np.split(A, N)
		b_split = np.split(b, N)
		
		# Step size.
		AA = A.T.dot(A)
		# alpha = 1.8/np.linalg.norm(AA, ord = 2)
		alpha = 100
		
		# Minimize \sum_i f_i(x) subject to x >= 0
		# where f_i(x) = ||A_ix - b_i||_2^2 for subproblem i = 1,...,N.
		x = Variable(n)
		constr = [x >= 0]
		
		p_list = []
		for A_sub, b_sub in zip(A_split, b_split):
			obj = 0.5*sum_squares(A_sub*x - b_sub)
			p_list += [Problem(Minimize(obj), constr)]
		probs = Problems(p_list)
		probs.pretty_vars()
		
		# Solve with consensus ADMM.
		obj_sdrs = probs.solve(method = "consensus", rho_init = alpha, max_iter = self.MAX_ITER)
		res_sdrs = probs.residuals
		
		# Solve with consensus ADMM using Anderson acceleration.
		res_aa2 = []
		for i in range(len(m_accel)):
			obj_aa2 = probs.solve(method = "consensus", rho_init = alpha, \
								max_iter = self.MAX_ITER, anderson = True, m_accel = m_accel[i])
			res_aa2.append(probs.residuals)
		x_aa2 = [x.value for x in probs.variables()]
		compare_residuals(res_sdrs, res_aa2, m_accel)
		
		# Solve combined problem.
		obj_comb = probs.solve(method = "combined")
		x_comb = [x.value for x in probs.variables()]
		
		# Compare results.
		self.assertAlmostEqual(obj_aa2, obj_comb)
		for i in range(N):
			self.assertItemsAlmostEqual(x_aa2[i], x_comb[i])
	
	def test_toeplitz(self):
		m = 50 
		n = 100
		N = 5
		rho = 1000   # Step size.
		tol = 1e-8   # Stopping tolerance.
		m_accel = 10   # Memory size for Anderson acceleration.

		# Problem data.
		A = np.hstack((toeplitz(np.arange(0,m)+1), np.eye(m,n-m)))
		b = (np.arange(0,m)+2)/100
		A_split = np.split(A, N)
		b_split = np.split(b, N)

		# Minimize \sum_i f_i(x) subject to x >= 0
		# where f_i(x) = ||A_ix - b_i||_2^2 for subproblem i = 1,...,N.
		x = Variable(n)
		constr = [x >= 0]
		
		p_list = []
		for A_sub, b_sub in zip(A_split, b_split):
			obj = sum_squares(A_sub*x - b_sub)
			# p_list += [Problem(Minimize(obj), constr)]
			p_list += [Problem(Minimize(obj))]
		p_list += [Problem(Minimize(0), constr)]
		probs = Problems(p_list)
		probs.pretty_vars()
		
		# Solve with consensus ADMM.
		obj_sdrs = probs.solve(method = "consensus", rho_init = rho, eps_stop = tol, max_iter = self.MAX_ITER)
		res_sdrs = probs.residuals
		
		# Solve combined problem.
		obj_comb = probs.solve(method = "combined")
		x_comb = [x.value for x in probs.variables()]
		
		obj_aa2 = probs.solve(method = "consensus", rho_init = rho, eps_stop = tol, \
								max_iter = self.MAX_ITER, anderson = True, m_accel = m_accel)
		res_aa2 = probs.residuals
		x_aa2 = [x.value for x in probs.variables()]
		compare_residuals(res_sdrs, res_aa2, m_accel)
		
		# Compare results.
		self.assertAlmostEqual(obj_aa2, obj_comb)
		for i in range(N):
			self.assertItemsAlmostEqual(x_aa2[i], x_comb[i])
