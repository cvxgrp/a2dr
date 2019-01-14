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

import cvxpy
import numpy as np
from cvxpy import Variable, Problem, Minimize
from scipy.linalg import qr_insert, qr_delete, solve_triangular

def dicts_to_arr(dicts):
	d_cols = []
	d_info = []
	
	for d in dicts:
		col = []
		info = {}
		offset = 0
		for key, val in d.items():		
			# Save shape and offset for later reconstruction.
			info[key] = {"shape": val.shape, "offset": offset}
			offset += val.size
			
			# Flatten into column-order array.
			col.append(val.flatten(order = "C"))
		
		# Stack all dict values into single column.
		col = np.concatenate(col)
		d_cols.append(col)
		d_info.append(info)
	
	arr = np.array(d_cols).T
	return arr, d_info

def arr_to_dicts(arr, d_info):
	dicts = []
	ncol = arr.shape[1]
	
	for j in range(ncol):
		d = {}
		for key, info in d_info[j].items():
			# Select corresponding rows.
			offset = info["offset"]
			size = np.prod(info["shape"])
			val = arr[offset:(offset + size),j]
			
			# Reshape vector and add to dict.
			val = np.reshape(val, newshape = info["shape"])
			d[key] = val
		dicts.append(d)
	return dicts

def aa_weights(residuals, lam = None, type = "exact", *args, **kwargs):
	""" Solve the constrained least-squares problem
	   Minimize sum_squares(\sum_{j=0}^m w[j]*r^(k+1-m+j)
	      subject to \sum_{j=0}^m w[j] = 1
	
	This can be transformed via a change of variables
	   w[0] = g[0], w[j] = g[j] - g[j-1] for j = 1,...,m-1, and w[m] = 1-g[m-1]
	into the unconstrained problem
	   Minimize sum_squares(r^(k+1) - \sum_{j=0}^{m-1} g[j]*(r^(k+2-m+j) - r^(k+1-m+j)))
	
	Parameters
	----------
	primals : array
	     A numpy array containing the primal residuals for the last `m+1`
	     iterations, including the current iteration. If our current iteration 
	     is `k+1`, then column `j` is the residual from iteration `k+1-m+j`
	     for `j = 0,...,m`, i.e., new residuals are appended (as the rightmost 
	     column) to the array.
	
	Returns
    ----------
    An array of length `m` containing the solution to the least-squares problem.
	"""
	# Form matrix of residuals G(v^(k) - m_k + j).
	# G_blocks = []
	# for res in residuals:
	#	arr, info = dicts_to_arr(res)
	#	G_blocks.append(arr)
	G_blocks = [dicts_to_arr(res)[0] for res in residuals]
	G = np.vstack(tuple(G_blocks))

	if type == "exact":
		# Solve for AA-II weights using unconstrained LS in numpy.
		e = np.ones((G.shape[1],))
		reg = lam*np.eye(G.shape[1]) if lam else 0   # Stabilization with l2 penalty.
		gamma = np.linalg.lstsq(G.T.dot(G) + reg, e, rcond = None)
		alpha = gamma[0]/np.sum(gamma[0])
		return alpha
	elif type == "inexact":
		# Solve for AA-II weights using constrained LS in CVXPY.
		alpha = Variable(G.shape[1])
		obj = cvxpy.sum_squares(G * alpha)
		reg = lam * cvxpy.sum_squares(alpha) if lam else 0  # Stabilization with l2 penalty.
		constr = [cvxpy.sum(alpha) == 1]
		prob = Problem(Minimize(obj + reg), constr)
		prob.solve(*args, **kwargs)

		if prob.status in cvxpy.settings.INF_OR_UNB:
			raise RuntimeError("AA-II weights subproblem is infeasible or unbounded")
		return alpha.value
	else:
		raise ValueError("type must be either 'exact' or 'inexact'")
