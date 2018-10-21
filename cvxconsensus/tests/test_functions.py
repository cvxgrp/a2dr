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
from cvxconsensus.utilities import assign_rho, partition_vars
from cvxconsensus.tests.base_test import BaseTest

class TestFunctions(BaseTest):
	""" Unit tests for internal functions"""
	
	def setUp(self):
		np.random.seed(1)
		self.n = 10
		self.x = Variable(10)
		self.y = Variable(20)
		self.z = Variable(5)
	
	def test_rho_init(self):
		p_list = [Problem(Minimize(norm(self.x)))]
		rho_list = assign_rho(p_list)
		self.assertDictEqual(rho_list[0], {self.x.id: 1.0})
		
		rho_list = assign_rho(p_list, default = 0.5)
		self.assertDictEqual(rho_list[0], {self.x.id: 0.5})
		
		rho_list = assign_rho(p_list, rho_init = {self.x.id: 1.5})
		self.assertDictEqual(rho_list[0], {self.x.id: 1.5})
		
		p_list.append(Problem(Minimize(0.5*sum_squares(self.y)), [norm(self.x) <= 10]))
		rho_list = assign_rho(p_list)
		self.assertDictEqual(rho_list[0], {self.x.id: 1.0})
		self.assertDictEqual(rho_list[1], {self.x.id: 1.0, self.y.id: 1.0})
		
		rho_list = assign_rho(p_list, rho_init = {self.x.id: 2.0}, default = 0.5)
		self.assertDictEqual(rho_list[0], {self.x.id: 2.0})
		self.assertDictEqual(rho_list[1], {self.x.id: 2.0, self.y.id: 0.5})
	
	def test_var_partition(self):
		p_list = [Problem(Minimize(norm(self.x)))]
		var_list = partition_vars(p_list)
		self.assertSetEqual(var_list[0]["private"], {self.x.id})
		self.assertSetEqual(var_list[0]["public"], set())
		
		p_list = [Problem(Minimize(norm(self.x) + norm(self.y))),
				  Problem(Minimize(norm(self.y)))]
		var_list = partition_vars(p_list)
		self.assertSetEqual(var_list[0]["private"], {self.x.id})
		self.assertSetEqual(var_list[0]["public"], {self.y.id})
		self.assertSetEqual(var_list[1]["private"], set())
		self.assertSetEqual(var_list[1]["public"], {self.y.id})
