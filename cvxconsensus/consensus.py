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
from cvxconsensus.acceleration import aa_weights
from cvxconsensus.utilities import flip_obj, assign_rho
from cvxconsensus.proximal import ProxOperator

def prox_step(prob, rho_init):
	"""Formulates the proximal operator for a given objective, constraints, and step size.
	   Reference: N. Parikh and S. Boyd (2013). "Proximal Algorithms."
	
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
		vmap[xid] = {"x": xvar, "y": Parameter(shape, value = np.zeros(shape)),
					 "rho": Parameter(value = rho_init[xid], nonneg = True)}
		f += (vmap[xid]["rho"]/2.0)*sum_squares(xvar - vmap[xid]["y"])
	
	prox = Problem(Minimize(f), prob.constraints)
	return prox, vmap

def w_project(prox_res, s_half):
	"""Projection step update of w^(k+1) = (x^(k+1), z^(k+1)) in the consensus scaled
	   Douglas-Rachford algorithm.
	"""
	y_part = defaultdict(list)
	var_cnt = defaultdict(float)
	
	for status, y_half in prox_res:
		# Check if proximal step converged.
		if status in s.INF_OR_UNB:
			raise RuntimeError("Proximal problem is infeasible or unbounded")
		
		# Partition y_i by variable ID and count instances.
		for key in y_half.keys():
			y_part[key].append(y_half[key])
			var_cnt[key] += 1.0
	
	if set(s_half.keys()) != set(y_part.keys()):
		raise RuntimeError("Mismatch between variable IDs of consensus and individual node terms")
	
	# Compute common matrix term and z update.
	mat_term = {}
	z_new = {}
	for key in s_half.keys():
		ys_sum = np.sum(np.array(y_part[key]), axis = 0) - var_cnt[key]*s_half[key]
		mat_term[key] = ys_sum/(1.0 + var_cnt[key])
		z_new[key] = s_half[key] + ys_sum - var_cnt[key] * mat_term[key]
	
	return mat_term, z_new

# def w_project_gen(prox_res, s_half, rho_list, rho_all):
#   Gamma_diag = np.array([rho_val for rhos in rho_list for rho_val in rhos.values()])
#   D_diag = np.array(list(rho_all.values()))
#   H_diag = np.concatenate((Gamma_diag, D_diag))
#   TODO: Define ED_mat according to mapping.
#	M = np.hstack((np.diag(1.0/np.sqrt(Gamma_diag)), -ED_mat))
#
#   TODO: Define v_half = (y_half, s_half) stacked.
#	w_half = np.diag(np.sqrt(H_diag)).dot(v_half)
#
#	Mw_sol = np.linalg.lstsq(M.T, w_half, rcond = None)
#	M_annl = np.eye(len(w_half)) - M.T.dot(Mw_sol)
#	w_proj = M_annl.dot(w_half)
#   w_new = np.diag(1.0/np.sqrt(H_diag)).dot(w_proj)
#
#   TODO: Partition back into x_new, z_new
#   return x_new, z_new

def run_worker(pipe, p, rho_init, anderson, m_accel, use_cvxpy, *args, **kwargs):
	# Initialize proximal problem.
	# prox, v = prox_step(p, rho_init)
	prox = ProxOperator(p, rho_vals = rho_init, use_cvxpy = use_cvxpy)
	v = prox.var_map
	
	# Initialize AA-II parameters.
	if anderson:
		y_hist = []   # History of y^(k).
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
		mat_term, s_half, k = pipe.recv()
		
		if anderson:
			m_k = min(m_accel, k)   # Keep iterations (k - m_k) through k.
			
			# Save history of y^(k).
			y = {key: v[key]["y"].value for key in v.keys()}
			y_hist.append(y)
			if len(y_hist) > m_k + 1:
				y_hist.pop(0)
			
			diff = {}
			y_res = []
			for key in v.keys():
				# Update corresponding w^(k+1) parameters.
				v[key]["x"].value = s_half[key] + mat_term[key]
				
				# Save history of y^(k) - F(y^(k)) = x^(k+1/2) - x^(k+1),
				# where F(.) is the consensus S-DRS mapping of v^(k+1) = F(v^(k)).
				diff[key] = x_half[key] - v[key]["x"].value
				y_res.append(diff[key].flatten(order = "C"))
			y_res = np.empty(0) if len(y_res) == 0 else np.concatenate(y_res)
			
			y_diff.append(diff)
			if len(y_diff) > m_k + 1:
				y_diff.pop(0)
			
			# Receive AA-II weights for y^(k+1).
			pipe.send((y_diff, y_res))
			alpha = pipe.recv()
			
			for key in v.keys():
				# Weighted update of y^(k+1).
				y_val = np.zeros(y_diff[0][key].shape)
				for j in range(m_k + 1):
					y_val += alpha[j] * (y_hist[j][key] - y_diff[j][key])
				v[key]["y"].value = y_val
		else:
			y_res = []
			for key in v.keys():
				# Update corresponding w^(k+1) parameters.
				v[key]["x"].value = s_half[key] + mat_term[key]
				
				# Update y^(k+1) = y^(k) + x^(k+1) - x^(k+1/2).
				v[key]["y"].value += v[key]["x"].value - x_half[key]
				
				# Send residual y^(k) - y^(k+1) for stopping criteria.
				res = x_half[key] - v[key]["x"].value
				y_res.append(res.flatten(order = "C"))
			y_res = np.empty(0) if len(y_res) == 0 else np.concatenate(y_res)
			pipe.send(y_res)

def consensus(p_list, *args, **kwargs):
	N = len(p_list)   # Number of problems.
	max_iter = kwargs.pop("max_iter", 100)
	rho_init = kwargs.pop("rho_init", dict())   # Step sizes.
	eps_stop = kwargs.pop("eps_stop", 1e-6)     # Stopping tolerance.
	use_cvxpy = kwargs.pop("use_cvxpy", False)	# Use CVXPY for proximal step?
	
	# AA-II parameters.
	anderson = kwargs.pop("anderson", False)
	m_accel = int(kwargs.pop("m_accel", 5))   # Maximum past iterations to keep (>= 0).
	if m_accel < 0:
		raise ValueError("m_accel must be a non-negative integer.")
	
	# Construct dictionary of step sizes for each node.
	var_all = {var.id: var for prob in p_list for var in prob.variables()}
	if np.isscalar(rho_init):
		rho_list, rho_all = assign_rho(p_list, default = rho_init)
	else:
		rho_list, rho_all = assign_rho(p_list, rho_init = rho_init)
	# var_list = partition_vars(p_list)   # Public/private variable partition.
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = run_worker, args = (remote, p_list[i], rho_list[i], \
							anderson, m_accel, use_cvxpy) + args, kwargs = kwargs)]
		procs[-1].start()

	# Initialize consensus variables.
	z = {key: np.zeros(var.shape) for key, var in var_all.items()}
	s = {key: np.zeros(var.shape) for key, var in var_all.items()}
	resid = np.zeros(max_iter)
	
	# Initialize AA-II parameters.
	if anderson:
		s_hist = []   # History of s^(k).
		s_diff = []   # History of s^(k) - F(s^(k)) fixed point mappings.

	# Consensus S-DRS loop.
	k = 0
	finished = False
	start = time()
	while not finished:
		# Gather y_i^(k+1/2) from nodes.
		prox_res = [pipe.recv() for pipe in pipes]
		
		# Projection step for w^(k+1).
		mat_term, z_new = w_project(prox_res, s)
	
		# Scatter s^(k+1/2) and common matrix term.
		for pipe in pipes:
			pipe.send((mat_term, s, k))
		
		if anderson:
			m_k = min(m_accel, k)   # Keep iterations (k - m_k) through k.
			
			# Save history of s^(k).
			s_hist.append(s.copy())
			if len(s_hist) > m_k + 1:
				s_hist.pop(0)
			
			# Receive history of y_i differences.
			y_update = [pipe.recv() for pipe in pipes]
			y_diffs, v_res = map(list, zip(*y_update))
			
			# Save history of s^(k) - F(s^(k)) = z^(k+1/2) - z^(k+1),
			# where F(.) is the consensus S-DRS mapping of v^(k+1) = F(v^(k)).
			diff = {}
			s_res = []   # Save current residual for stopping criteria.
			for key in var_all.keys():
				diff[key] = s[key] - z_new[key]
				s_res.append(diff[key].flatten(order = "C"))
			s_res = np.empty(0) if len(s_res) == 0 else np.concatenate(s_res)
			s_diff.append(diff)
			if len(s_diff) > m_k + 1:
				s_diff.pop(0)
			
			# Compute l2-norm of residual G(v^(k)) = v^(k) - v^(k+1) 
			# where v^(k) = (y^(k), s^(k)).
			v_res.append(s_res)
			v_res = np.concatenate(v_res, axis = 0)
			resid[k] = np.linalg.norm(v_res, ord = 2)
			
			# Compute and scatter AA-II weights.
			# alpha = aa_weights(y_diffs + [s_diff], type = "inexact", solver = "OSQP", eps_abs = 1e-16)
			alpha = aa_weights(y_diffs + [s_diff])
			for pipe in pipes:
				pipe.send(alpha)
			
			# Weighted update of s^(k+1).
			for key in var_all.keys():
				s_val = np.zeros(s_diff[0][key].shape)
				for j in range(m_k + 1):
					s_val += alpha[j] * (s_hist[j][key] - s_diff[j][key])
				s[key] = s_val
		else:
			s_res = []
			for key in var_all.keys():
				# Save residual s^(k) - s^(k+1) for stopping criteria.
				res = s[key] - z_new[key]
				s_res.append(res.flatten(order = "C"))

				# Update s^(k+1) = s^(k) + z^(k+1) - z^(k+1/2).
				s[key] = z_new[key]
			s_res = np.empty(0) if len(s_res) == 0 else np.concatenate(s_res)
			
			# Compute l2-norm of residual G(v^(k)) = v^(k) - v^(k+1) 
			# where v^(k) = (y^(k), s^(k)).
			v_res = [pipe.recv() for pipe in pipes]
			v_res.append(s_res)
			v_res = np.concatenate(v_res, axis = 0)
			resid[k] = np.linalg.norm(v_res, ord = 2)
			# rnorm = resid[k]/np.max(resid[k]) if anderson else resid[k]/resid[0]
			
		# Stop if G(v^(k))/G(v^(0)) falls below tolerance.
		z = z_new
		k = k + 1
		finished = k >= max_iter or resid[k-1] <= eps_stop*resid[0]
	end = time()
	
	[p.terminate() for p in procs]
	return {"zvals": z, "residuals": np.array(resid[:k]), "num_iters": k, "solve_time": (end - start)}
