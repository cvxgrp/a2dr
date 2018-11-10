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
    rho_init : float
        The initial step size.
    
    Returns
    ----------
    prox : Problem
        The proximal step problem.
    vmap : dict
        A map of each proximal variable id to a dictionary containing that variable `x`,
        the mean variable parameter `xbar`, the associated dual parameter `y`, and the
        step size parameter `rho`. If `spectral = True`, the estimated dual parameter 
        `yhat` is also included.
	"""
	vmap = {}   # Store consensus variables
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

def res_stop(res_ssq, eps = 1e-4):
	"""Calculate the sum of squared primal/dual residuals.
	   Determine whether the stopping criterion is satisfied:
	   ||r^(k)||^2 <= eps*max(\sum_i ||x_i^(k)||^2, \sum_i ||x_bar^(k)||^2) and
	   ||d^(k)||^2 <= eps*\sum_i ||y_i^(k)||^2
	"""
	primal = np.sum([r["primal"] for r in res_ssq])
	dual = np.sum([r["dual"] for r in res_ssq])
	
	x_ssq = np.sum([r["x"] for r in res_ssq])
	xbar_ssq = np.sum([r["xbar"] for r in res_ssq])
	u_ssq = np.sum([r["y"] for r  in res_ssq])
	
	eps_fl = np.finfo(float).eps   # Machine precision.
	stopped = (primal <= eps*max(x_ssq, xbar_ssq) + eps_fl) and \
			  (dual <= eps*u_ssq + eps_fl)
	return primal, dual, stopped

def run_worker(pipe, p, *args, **kwargs):
	# Anderson acceleration parameters.
	anderson = kwargs.pop("anderson", False)
	m_accel = kwargs.pop("m_accel", 5)   # Number of past iterations to keep (>= 1)
	
	# Initiate proximal problem.
	prox, v = prox_step(p, rho_init)
	
	# ADMM loop.
	while True:	
		# Proximal step for x^(k+1/2).
		prox.solve(*args, **kwargs)
		x_half = {key: v[key]["x"].value for key in v.keys()}
		
		# Calculate y^(k+1/2) = 2*x^(k+1/2) - y^(k).
		y_half = {key: 2*x_half[key] - v[key]["y"].value for key in v.keys()}
		
		# Project to obtain w^(k+1) = (x^(k+1), z^(k+1)).
		pipe.send((prox.status, y_half))
		mat_term, z_new, i = pipe.recv()
		for key in v.keys():
			# Update corresponding w^(k+1) parameters.
			s_half = v[key]["z"].value
			v[key]["x"].value = s_half + mat_term[key]
			v[key]["z"].value = z_new[key]
		
			# Update v^(k+1) = v^(k) + w^(k+1) - w^(k+1/2) where v^(k) = (y^(k), s^(k)).
			v[key]["y"].value += v[key]["x"].value - x_half[key]
			
			# TODO: Should we have central node update consensus term s 
			# to avoid nodes diverging due to, e.g., floating point error?
			v[key]["s"].value += v[key]["z"].value - s_half

def consensus(p_list, *args, **kwargs):
	N = len(p_list)   # Number of problems.
	max_iter = kwargs.pop("max_iter", 100)
	rho_x = kwargs.pop("rho_x", dict())   # Step sizes.
	rho_z = kwargs.pop("rho_z", 1.0)
	resid = np.zeros((max_iter, 2))
	
	var_all = {var.id: var for prob in p_list for var in prob.variables()}
	if np.isscalar(rho_x):
		rho_x = {key: rho_x for key in var_all.keys()}
	else:
		rho_x = {key: rho_x.get(key, 1.0) for key in var_all.keys()}
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = run_worker, args = (remote, p_list[i]) + args, kwargs = kwargs)]
		procs[-1].start()

	# ADMM loop.
	start = time()
	for i in range(max_iter):
		# Gather and average x_i.
		prox_res = [pipe.recv() for pipe in pipes]
		mat_term, z_new = w_project(prox_res, z, rho_x, rho_z)
	
		# Scatter x_bar.
		for pipe in pipes:
			pipe.send((mat_term, z_new, i))
			
		# TODO: Update v^(k+1) = v^(k) + w^(k+1) - w^(k+1/2).
	end = time()
	
	[p.terminate() for p in procs]
	return {"xbars": z_new, "num_iters": i+1, "solve_time": (end - start)}
