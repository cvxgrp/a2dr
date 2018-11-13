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
from cvxpy import Variable, Parameter, Problem, Minimize
from cvxpy.atoms import *
import cvxconsensus
from cvxconsensus import Problems
from cvxconsensus.tests.base_test import BaseTest

def compare_residuals(res_sdrs, res_aa2):
	plt.plot(range(res_sdrs.shape[0]), res_sdrs, "b-", label = "S-DRS")
	plt.plot(range(res_aa2.shape[0]), res_aa2, "b--", label = "S-DRS with AA-II")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Residual")
	plt.show()
	
class TestAcceleration(BaseTest):
	"""Unit tests for Anderson acceleration of consensus ADMM"""
	
	def setUp(self):
		np.random.seed(1)
		self.MAX_ITER = 100
	
	def test_lasso(self):	
		m = 100
		n = 10
		DENSITY = 0.75
		x = Variable(n)
		
		# Problem data.
		A = np.random.randn(m*n).reshape(m,n)
		xtrue = np.random.randn(n)
		idxs = np.random.choice(range(n), int((1-DENSITY)*n), replace = False)
		for idx in idxs:
			xtrue[idx] = 0
		b = A.dot(xtrue) + np.random.randn(m)
		
		# List of all problems with objective f_i.
		p_list = [Problem(Minimize(sum_squares(A*x-b))),
				  Problem(Minimize(norm(x,1)))]
		probs = Problems(p_list)
		N = len(p_list)
		
		# Solve with consensus ADMM.
		obj_sdrs = probs.solve(method = "consensus", rho_init = 1.0, max_iter = self.MAX_ITER)
		res_sdrs = probs.residuals
		
		# Solve with consensus ADMM using Anderson acceleration.
		obj_aa2 = probs.solve(method = "consensus", rho_init = 1.0, \
							   max_iter = self.MAX_ITER, anderson = True, m_accel = 5)
		res_aa2 = probs.residuals
		x_aa2 = [x.value for x in probs.variables()]
		compare_residuals(res_sdrs, res_aa2)
		
		# Solve combined problem.
		obj_comb = probs.solve(method = "combined")
		x_comb = [x.value for x in probs.variables()]
		
		# Compare results.
		N = len(probs.variables())
		self.assertAlmostEqual(obj_aa2, obj_comb)
		for i in range(N):
			self.assertItemsAlmostEqual(x_aa2[i], x_comb[i])
