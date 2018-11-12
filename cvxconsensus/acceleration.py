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

def aa_weights(residuals, lam = None):
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
	
	w = Variable(residuals.shape[1])
	obj = cvxpy.sum_squares(residuals * w)
	reg = lam * sum_squares(w) if lam else 0
	constr = [cvxpy.sum(w) == 1]
	prob = Problem(Minimize(obj + reg), constr)
	prob.solve()
	return w.value

def aa_update(vals, residuals, rho, lam = None):
	""" Dual variable update for the accelerated ADMM consensus problem.
	
	Parameters
	----------
	vals : array
	     A numpy array containing the dual variable value for the last `m+1`
	     iterations, including the current iteration. If our current iteration
	     is `k`, then column `j` is the dual from iteration `k-m+j` for
	     `j = 0,...,m`, i.e., new values are appended (as the rightmost column)
	     to the array.
	primals : array
	     A numpy array containing the primal residuals for the last `m+1`
	     iterations, ordered in the same manner as `yvals`.
	rho : float
	     The positive mixing parameter.
	"""
	weights = aa_weights(residuals, lam)
	prhos = np.multiply(residuals, rho)
	return (vals + prhos).dot(weights)
	
	# yRnew = yvals[:,-1] + rho[-1]*primals[:,-1]
	# if primals.shape[1] == 1:
	#	return yRnew
	#
	# weights = dual_weights(primals)
	# ydiff = np.diff(yvals, axis = 1)
	# prhos = np.multiply(primals, rho)
	# Rdiff = np.diff(prhos, axis = 1)
	# yRoff = (ydiff + Rdiff).dot(weights)
	# return yRnew - yRoff[:,0]
