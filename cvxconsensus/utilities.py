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
from cvxpy.problems.problem import Minimize
from collections import Counter

def flip_obj(prob):
	"""Helper function to flip sign of objective function.
	"""
	if isinstance(prob.objective, Minimize):
		return prob.objective
	else:
		return -prob.objective

def assign_rho(p_list, rho_init = dict(), default = 1.0):
	"""For each problem, construct a dictionary that maps variable id to 
	   initial step size value.
	"""
	vids = {var.id for prob in p_list for var in prob.variables()}
	for key in rho_init.keys():
		if not key in vids:
			raise ValueError("{} is not a valid variable id".format(key))
	rho_list = [{var.id: rho_init.get(var.id, default) for var in prob.variables()} for prob in p_list]
	rho_all = {var.id: rho_init.get(var.id, default) for prob in p_list for var in prob.variables()}
	return rho_list, rho_all

def partition_vars(p_list):
	"""For each problem, partition variables into public (shared with at 
	   least one other problem) and private (only exists in that problem).
	"""
	var_merge = [var.id for prob in p_list for var in prob.variables()]
	var_count = Counter(var_merge)
	var_list = []
	for prob in p_list:
		var_dict = {"public": set(), "private": set()}
		for var in prob.variables():
			if var_count[var.id] == 1:
				var_dict["private"].add(var.id)
			else:
				var_dict["public"].add(var.id)
		var_list.append(var_dict)
	return var_list
