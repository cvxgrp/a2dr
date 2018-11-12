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
from time import time
from collections import defaultdict
from multiprocessing import Process, Pipe
import cvxpy.settings as s
from cvxpy.problems.problem import Problem, Minimize
from cvxpy.expressions.constants import Parameter
from cvxpy.atoms import sum_squares
from cvxconsensus.acceleration import dual_update
from cvxconsensus.utilities import flip_obj, assign_rho, partition_vars

def prox_step(prob, rho_init):
	"""Formulates the proximal operator for a given objective, constraints, and step size.
	Parikh, Boyd. "Proximal Algorithms."
	
	Parameters
    ----------
    prob : Problem
        The objective and constraints associated with the proximal operator.
        The sign of the objective function is flipped if `prob` is a maximization problem.
    rho_init : dict
        The initial step size indexed by unique variable id.
    
    Returns
    ----------
    prox : Problem
        The proximal step problem.
    vmap : dict
        A map of each proximal variable id to a dictionary containing that variable `x`,
        the consensus variable parameter `z`, the associated dual parameter `y`, and the
        step size parameter `rho`.
	"""
	vmap = {}   # Store consensus variables.
	f = flip_obj(prob).args[0]
	
	# Add penalty for each variable.
	for xvar in prob.variables():
		xid = xvar.id
		shape = xvar.shape
		vmap[xid] = {"x": xvar, "z": Parameter(shape, value = np.zeros(shape)),
		 		     "y": Parameter(shape, value = np.zeros(shape)),
					 "rho": Parameter(value = rho_init[xid], nonneg = True)}
		f += (vmap[xid]["rho"]/2.0)*sum_squares(xvar - vmap[xid]["y"])
	
	prox = Problem(Minimize(f), prob.constraints)
	return prox, vmap

def w_project(prox_res, s_half):
	"""Projection step update of w^(k+1) = (x^(k+1), z^(k+1)) in the
	   consensus scaled Douglas-Rachford algorithm.
	"""
	ys_diff = defaultdict(list)
	var_cnt = defaultdict(float)
	
	for status, y_half in prox_res:
		# Check if proximal step converged.
		if status in s.INF_OR_UNB:
			raise RuntimeError("Proximal problem is infeasible or unbounded")
		
		# Store difference between y_i and consensus term s for each variable ID.
		for key in y_half.keys():
			ys_diff[key].append(y_half[key] - s_half[key])
			var_cnt[key] += 1.0
	
	if set(s_half.keys()) != set(ys_diff.keys()):
		raise RuntimeError("Mismatch between variable IDs of consensus and individual node terms")
	
	# Compute common matrix term and z update.
	mat_term = {}
	z_new = {}
	for key in z.keys()
		ys_sum[key] = np.sum(np.array(ys_diff[key]), axis = 0)
		mat_term[key] = ys_sum[key]/(1.0 + var_cnt[key])
		z_new[key] = s_half[key] + ys_sum[key] - var_cnt[key] * mat_term[key]
	
	return mat_term, z_new

def run_worker(pipe, p, rho_init, anderson, m_accel, *args, **kwargs):
	# Initialize proximal problem.
	prox, v = prox_step(p, rho_init)
	
	# Initialize AA-II parameters.
	if anderson:
		y_diff = []   # History of y^(k) - F(y^(k)) fixed point mappings.
	
	# Consensus S-DRS loop.
	while True:	
		# Proximal step for x^(k+1/2).
		prox.solve(*args, **kwargs)
		x_half = {key: v[key]["x"].value for key in v.keys()}
		
		# Calculate y^(k+1/2) = 2*x^(k+1/2) - y^(k).
		y_half = {key: 2*x_half[key] - v[key]["y"].value for key in v.keys()}
		
		# Project to obtain w^(k+1) = (x^(k+1), z^(k+1)).
		pipe.send((prox.status, y_half))
		mat_term, z_new, k = pipe.recv()
		
		if anderson:
			m_k = min(m_accel, k)   # Keep iterations (k - m_k) through k.
			
			diff = {}
			for key in v.keys():
				# Update corresponding w^(k+1) parameters.
				s_half = v[key]["z"].value
				v[key]["x"].value = s_half + mat_term[key]
				v[key]["z"].value = z_new[key]
				
				# Save history of y^(k) - F(y^(k)) = x^(k+1/2) - x^(k+1),
				# where F(.) is the consensus S-DRS mapping of v^(k+1) = F(v^(k)).
				diff[key] = x_half[key] - v[key]["x"].value
			y_diff.append(diff)
			if len(y_diff) > m_k + 1:
				y_diff.pop(0)
			
			# Receive s^(k+1) and AA-II weights for y^(k+1).
			pipe.send(y_diff)
			s_new, y_weight = pipe.recv()
			
			for key in v.keys():
				# Set s^(k+1) value.
				v[key]["s"].value = s_new[key]
				
				# Weighted update of y^(k+1).
				y_val = np.zeros(y_diff[0][key].shape)
				for j in range(m_k + 1):
					y_val += y_weight[j] * y_diff[j][key]
				v[key]["y"].value = y_val
		else:
			for key in v.keys():
				# Update corresponding w^(k+1) parameters.
				s_half = v[key]["z"].value
				v[key]["x"].value = s_half + mat_term[key]
				v[key]["z"].value = z_new[key]
				
				# Update y^(k+1) = y^(k) + x^(k+1) - x^(k+1/2).
				v[key]["y"].value += v[key]["x"].value - x_half[key]
				# v[key]["s"].value += v[key]["z"].value - s_half
			
			# Receive and set s^(k+1) value.
			s_new = pipe.recv()
			for key in v.keys():
				v[key]["s"].value = s_new[key]

def consensus(p_list, *args, **kwargs):
	N = len(p_list)   # Number of problems.
	max_iter = kwargs.pop("max_iter", 100)
	rho_init = kwargs.pop("rho_init", dict())   # Step sizes.
	
	# AA-II parameters.
	anderson = kwargs.pop("anderson", False)
	m_accel = int(kwargs.pop("m_accel", 5))   # Maximum past iterations to keep (>= 0).
	if m_accel < 0:
		raise ValueError("m_accel must be a non-negative integer.")
	
	# Construct dictionary of step sizes for each node.
	var_all = {var.id: var for prob in p_list for var in prob.variables()}
	if np.isscalar(rho_init):
		rho_list = assign_rho(p_list, default = rho_init)
	else:
		rho_list = assign_rho(p_list, rho_init = rho_init)
	# var_list = partition_vars(p_list)   # Public/private variable partition.
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = run_worker, args = (remote, p_list[i], rho_list[i], \
							anderson, m_accel) + args, kwargs = kwargs)]
		procs[-1].start()

	# Initialize consensus variables.
	s = defaultdict(float)
	z = {key: np.zeros(var.shape) for key, var in var_all.items()}
	
	# Initialize AA-II parameters.
	if anderson:
		s_diff = []

	# Consensus S-DRS loop.
	start = time()
	for k in range(max_iter):
		# Gather y_i^(k+1/2) from nodes.
		prox_res = [pipe.recv() for pipe in pipes]
		
		# Projection step for w^(k+1).
		mat_term, z_new = w_project(prox_res, z)
	
		# Scatter z^(k+1) and common matrix term.
		for pipe in pipes:
			pipe.send((mat_term, z_new, k))
		
		if anderson:
			m_k = min(m_accel, k)   # Keep iterations (k - m_k) through k.
			
			# Receive history of y_i differences.
			y_diffs = [pipe.recv() for pipe in pipes]
			
			# Save history of s^(k) - F(s^(k)) = z^(k+1/2) - z^(k+1),
			# where F(.) is the consensus S-DRS mapping of v^(k+1) = F(v^(k)).
			diff = {}
			for key in var_all.keys():
				diff[key] = z[key] - z_new[key]
			s_diff.append(diff)
			if len(s_diffs) > m_k + 1:
				s_diff.pop(0)
			
			# Compute AA-II weights.
			y_weights, s_weight = aa_weights(y_diffs + [s_diff])
			
			# Weighted update of s^(k+1).
			for key in var_all.keys():
				s_val = np.zeros(s_diff[0][key].shape)
				for j in range(m_k + 1):
					s_val += s_weight[j] * s_diff[j][key]
				s[key] = s_val
			
			# Scatter s^(k+1) and AA-II weights for y^(k+1).
			for pipe, y_weight in zip(pipes, y_weights):
				pipe.send((s, y_weight))
			z = z_new
		else:
			# Update s^(k+1) = s^(k) + z^(k+1) - z^(k+1/2).
			for key in var_all.keys():
				s[key] += z_new[key] - z[key]
			z = z_new
	end = time()
	
	[p.terminate() for p in procs]
	return {"xvals": z, "num_iters": k + 1, "solve_time": (end - start)}
