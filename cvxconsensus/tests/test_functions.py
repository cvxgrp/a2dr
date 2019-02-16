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
from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import *
import cvxpy.settings as s
from cvxconsensus.consensus import prox_step, w_project, w_project_gen
from cvxconsensus.utilities import assign_rho, partition_vars
from cvxconsensus.tests.base_test import BaseTest

class TestFunctions(BaseTest):
	""" Unit tests for internal functions"""
	
	def setUp(self):
		np.random.seed(1)
		self.m = 50
		self.n = 10
		self.x = Variable(10)
		self.y = Variable(20)
		self.z = Variable(5)
	
	def test_rho_init(self):
		p_list = [Problem(Minimize(norm(self.x)))]
		rho_list, rho_all = assign_rho(p_list)
		self.assertDictEqual(rho_list[0], {self.x.id: 1.0})
		self.assertDictEqual(rho_all, {self.x.id: 1.0})
		
		rho_list, rho_all = assign_rho(p_list, default = 0.5)
		self.assertDictEqual(rho_list[0], {self.x.id: 0.5})
		self.assertDictEqual(rho_all, {self.x.id: 0.5})
		
		rho_list, rho_all = assign_rho(p_list, rho_init = {self.x.id: 1.5})
		self.assertDictEqual(rho_list[0], {self.x.id: 1.5})
		self.assertDictEqual(rho_all, {self.x.id: 1.5})
		
		p_list.append(Problem(Minimize(0.5*sum_squares(self.y)), [norm(self.x) <= 10]))
		rho_list, rho_all = assign_rho(p_list)
		self.assertDictEqual(rho_list[0], {self.x.id: 1.0})
		self.assertDictEqual(rho_list[1], {self.x.id: 1.0, self.y.id: 1.0})
		self.assertDictEqual(rho_all, {self.x.id: 1.0, self.y.id: 1.0})
		
		rho_list, rho_all = assign_rho(p_list, rho_init = {self.x.id: 2.0}, default = 0.5)
		self.assertDictEqual(rho_list[0], {self.x.id: 2.0})
		self.assertDictEqual(rho_list[1], {self.x.id: 2.0, self.y.id: 0.5})
		self.assertDictEqual(rho_all, {self.x.id: 2.0, self.y.id: 0.5})

		p_list.append(Problem(Minimize(norm(self.z))))
		rho_list, rho_all = assign_rho(p_list, rho_init = {self.x.id: 2.0, self.z.id: 3.0}, default = 0.5)
		self.assertDictEqual(rho_list[0], {self.x.id: 2.0})
		self.assertDictEqual(rho_list[1], {self.x.id: 2.0, self.y.id: 0.5})
		self.assertDictEqual(rho_list[2], {self.z.id: 3.0})
		self.assertDictEqual(rho_all, {self.x.id: 2.0, self.y.id: 0.5, self.z.id: 3.0})
	
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

	def test_prox_step(self):
		rho = {self.x.id: 1.0}
		obj = norm(self.x)
		y_val = np.random.randn(*self.x.shape)

		prox, v = prox_step(Problem(Minimize(obj)), rho)
		v[self.x.id]["y"].value = y_val
		prox.solve()
		x_prox = self.x.value

		p = Problem(Minimize(obj + (rho[self.x.id]/2.0)*sum_squares(self.x - y_val)))
		p.solve()
		self.assertItemsAlmostEqual(x_prox, self.x.value)

		obj = 0
		constr = [self.x >= 0]
		prox, v = prox_step(Problem(Minimize(obj), constr), rho)
		v[self.x.id]["y"].value = y_val
		prox.solve()
		x_prox = self.x.value

		p = Problem(Minimize(obj + (rho[self.x.id]/2.0)*sum_squares(self.x - y_val)), constr)
		p.solve()
		self.assertItemsAlmostEqual(x_prox, self.x.value)

	def test_projection(self):
		xid = self.x.id
		y_half = {xid: np.random.randn(*self.x.shape)}
		s_half = {xid: np.random.randn(*self.x.shape)}
		mat_term, z_new = w_project([(s.OPTIMAL, y_half)], s_half)
		self.assertItemsAlmostEqual(mat_term[xid], (y_half[xid] - s_half[xid])/2)
		self.assertItemsAlmostEqual(z_new[xid], (y_half[xid] + s_half[xid])/2)

		N = 5
		prox_res = []
		for i in range(N):
			y_half = {xid: np.random.randn(*self.x.shape)}
			prox_res.append((s.OPTIMAL, y_half))
		s_half = {xid: np.random.randn(*self.x.shape)}
		mat_term, z_new = w_project(prox_res, s_half)

		y_halves = [y_half[xid] for status, y_half in prox_res]
		self.assertItemsAlmostEqual(mat_term[xid], (sum(y_halves) - N*s_half[xid])/(N + 1))
		self.assertItemsAlmostEqual(z_new[xid], (sum(y_halves) + s_half[xid])/(N + 1))

		y_half1 = {self.x.id: np.random.randn(*self.x.shape), self.y.id: np.random.randn(*self.y.shape)}
		y_half2 = {self.y.id: np.random.randn(*self.y.shape), self.z.id: np.random.randn(*self.z.shape)}
		s_half = {self.x.id: np.random.randn(*self.x.shape), self.y.id: np.random.randn(*self.y.shape), \
				  self.z.id: np.random.randn(*self.z.shape)}
		prox_res = [(s.OPTIMAL, y_half1), (s.OPTIMAL_INACCURATE, y_half2)]
		mat_term, z_new = w_project(prox_res, s_half)

		y_halves = [y_half[self.y.id] for status, y_half in prox_res]
		self.assertItemsAlmostEqual(mat_term[self.y.id], (sum(y_halves) - 2*s_half[self.y.id])/3)
		self.assertItemsAlmostEqual(mat_term[self.x.id], (y_half1[self.x.id] - s_half[self.x.id])/2)
		self.assertItemsAlmostEqual(mat_term[self.z.id], (y_half2[self.z.id] - s_half[self.z.id])/2)
		self.assertItemsAlmostEqual(z_new[self.y.id], -s_half[self.y.id] + sum(y_halves) - 2*(sum(y_halves) - 2*s_half[self.y.id])/3)
		self.assertItemsAlmostEqual(z_new[self.x.id], y_half1[self.x.id] - (y_half1[self.x.id] - s_half[self.x.id])/2)
		self.assertItemsAlmostEqual(z_new[self.z.id], y_half2[self.z.id] - (y_half2[self.z.id] - s_half[self.z.id])/2)

	def test_projection_gen(self):
		xid = self.x.id
		rho_all = {xid: 1.0}
		y_half = {xid: np.random.randn(*self.x.shape)}
		s_half = {xid: np.random.randn(*self.x.shape)}
		x_new, z_new = w_project_gen([(s.OPTIMAL, y_half)], s_half, rho_all)
		self.assertItemsAlmostEqual(x_new[0][xid], (y_half[xid] + s_half[xid])/2)
		self.assertItemsAlmostEqual(z_new[xid], (y_half[xid] + s_half[xid])/2)

		N = 5
		rho_all = {xid: 0.5}
		prox_res = []
		for i in range(N):
			y_half = {xid: np.random.randn(*self.x.shape)}
			prox_res.append((s.OPTIMAL, y_half))
		s_half = {xid: np.random.randn(*self.x.shape)}
		x_new, z_new = w_project_gen(prox_res, s_half, rho_all)

		y_halves = [y_half[xid] for status, y_half in prox_res]
		xz_res = (sum(y_halves) + s_half[xid])/(N + 1)
		for i in range(N):
			self.assertItemsAlmostEqual(x_new[i][xid], xz_res)
		self.assertItemsAlmostEqual(z_new[xid], xz_res)

		rho_all = {self.x.id: 1.0, self.y.id: 0.5, self.z.id: 2.0}
		y_half1 = {self.x.id: np.random.randn(*self.x.shape), self.y.id: np.random.randn(*self.y.shape)}
		y_half2 = {self.y.id: np.random.randn(*self.y.shape), self.z.id: np.random.randn(*self.z.shape)}
		s_half = {self.x.id: np.random.randn(*self.x.shape), self.y.id: np.random.randn(*self.y.shape), \
				  self.z.id: np.random.randn(*self.z.shape)}
		prox_res = [(s.OPTIMAL, y_half1), (s.OPTIMAL_INACCURATE, y_half2)]
		x_new, z_new = w_project_gen(prox_res, s_half, rho_all)

		y_halves = [y_half[self.y.id] for status, y_half in prox_res]
		self.assertItemsAlmostEqual(x_new[0][self.x.id], s_half[self.x.id] + (y_half1[self.x.id] - s_half[self.x.id])/2)
		self.assertItemsAlmostEqual(x_new[0][self.y.id], s_half[self.y.id] + (sum(y_halves) - 2*s_half[self.y.id])/3)
		self.assertItemsAlmostEqual(x_new[1][self.y.id], s_half[self.y.id] + (sum(y_halves) - 2*s_half[self.y.id])/3)
		self.assertItemsAlmostEqual(x_new[1][self.z.id], s_half[self.z.id] + (y_half2[self.z.id] - s_half[self.z.id])/2)

		self.assertItemsAlmostEqual(z_new[self.y.id], -s_half[self.y.id] + sum(y_halves) - 2*(sum(y_halves) - 2*s_half[self.y.id])/3)
		self.assertItemsAlmostEqual(z_new[self.x.id], y_half1[self.x.id] - (y_half1[self.x.id] - s_half[self.x.id])/2)
		self.assertItemsAlmostEqual(z_new[self.z.id], y_half2[self.z.id] - (y_half2[self.z.id] - s_half[self.z.id])/2)
