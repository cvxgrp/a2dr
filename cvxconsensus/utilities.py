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
from cvxpy.problems.problem import Problem, Minimize

def flip_obj(prob):
	"""Helper function to flip sign of objective function.
	"""
	if isinstance(prob.objective, Minimize):
		return prob.objective
	else:
		return -prob.objective

def assign_rho(p_list, rho_init = dict(), default = 1.0):
	"""Construct dictionaries that map variable id to initial step
	   size value for each problem.
	"""
	return [{var.id: rho_init.get(var.id, default) for var in \
				prob.variables()} for prob in p_list]

def separate_vars(p_list):
	all_vars = {var.id for var in prob.variables() for prob in p_list}
	var_list = []
	for prob in p_list:
		var_dict = {"public": [], "private": []}
		for var in prob.variables():
			if var in all_vars:
				var_dict["public"].append(var)
			else:
				var_dict["private"].append(var)
		var_list.append(var_dict)
	return var_list
