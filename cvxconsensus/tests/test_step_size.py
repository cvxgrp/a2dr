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
from cvxpy import Variable, Parameter, Problem, Minimize
from cvxpy.atoms import *
import cvxconsensus
from cvxconsensus import Problems
from cvxconsensus.tests.base_test import BaseTest

class TestStepSize(BaseTest):
	"""Unit tests for various step size parameters"""
	
	def setUp(self):
		np.random.seed(1)
		self.MAX_ITER = 100
	
	def test_error(self):
		n = 10
		x = Variable(n)
		p_list = [Problem(Minimize(norm(x,1)))]
		probs = Problems(p_list)
		
		probs.solve(method = "consensus")
		probs.solve(method = "consensus", rho_init = {x.id: 0.5})
		with self.assertRaises(ValueError) as cm:
			probs.solve(method = "consensus", rho_init = {(x.id + 1): 0.5})
	
	def test_multiple(self):
		m = 100
		n = 10
		N = 5
		
		betas = []
		p_list = []
		rho_init = {}
		for i in range(N):
			x = Variable(n)
			A = np.random.randn(m,n)
			b = np.random.randn(m)
			prob = Problem(Minimize(sum_squares(A*x - b)))
			p_list.append(prob)
			rho_init[x.id] = (i+1)*0.1
		
		probs = Problems(p_list)
		probs.pretty_vars()
		
		# Solve with consensus ADMM.
		obj_admm = probs.solve(method = "consensus", rho_init = rho_init, \
								max_iter = self.MAX_ITER, spectral = False)
		x_admm = [x.value for x in probs.variables()]
		# probs.plot_residuals()
		
		# Solve combined problem.
		obj_comb = probs.solve(method = "combined")
		x_comb = [x.value for x in probs.variables()]
		
		# Compare results.
		self.compare_results(probs, obj_admm, obj_comb, x_admm, x_comb)
