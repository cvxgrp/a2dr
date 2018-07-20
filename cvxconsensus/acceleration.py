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
from scipy.linalg import qr_insert, qr_delete, solve_triangular
from cvxpy import Variable, Problem, Minimize, sum_squares

def solve_ls(b, Q, R, Qt, Rt):
	""" Solve the least-squares problem
	   Minimize sum_squares(Ax - b)
	with variable x, given the QR decomposition of `A` and `A.T`. Both
	are necessary because we do not know a priori if `A` is a fat or 
	skinny matrix.
	
	https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
	
	Parameters
	----------
	b : array
	     The data for the problem.
	Q, R : array
	     The QR decomposition of `A`.
	Qt, Rt : array
	     The QR decomposition of `A.T`.
	
	Returns
    ----------
    An array containing the solution.
	"""
	m = Q.shape[0]
	n = R.shape[1]
	if m < n:
		Rt = Rt[0:m,:]
		Rib = solve_triangular(Rt.T, b, lower = True)
		Rib = np.insert(Rib, m, np.zeros((n-m,1)), axis = 0)
		return Qt.dot(Rib)
	else:
		Q = Q[:,0:n]
		R = R[0:n,:]
		Qb = Q.T.dot(b)
		return solve_triangular(R, Qb, lower = False)

def trim_cond(Q, R, Qt, Rt, rcond = np.inf):
	""" Update the QR decomposition of a matrix `A` after its columns have been
	trimmed (from left to right) so its condition number falls below a threshold.
	
	Parameters
	----------
	Q, R : array
	     The QR decomposition of `A`.
	Qt, Rt : array
	     The QR decomposition of `A.T`.
	rcond : int
	     Upper threshold on the condition number.
	
	Returns
    ----------
    The QR decomposition of the trimmed matrix and its transpose.
	"""
	cond = np.linalg.cond(Q.dot(R))
	while (cond > rcond):
		Q, R = qr_delete(Q, R, 0, 1, which = 'col')
		Qt, Rt = qr_delete(Qt, Rt, 0, 1, which = 'row')
		cond = np.linalg.cond(Q.dot(R))
	return (Q, R, Qt, Rt)

def dual_weights(primals):
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
	
	w = Variable(primals.shape[1])
	obj = sum_squares(primals*w)
	constr = [sum(w) == 1]
	prob = Problem(Minimize(obj), constr)
	prob.solve()
	return w.value
	
	# if primals.shape[1] == 1:
	#	return np.zeros((primals.shape[0],1))
	#
	# rnew = np.array([primals[:,-1]]).T
	# Rdiff = np.diff(primals, axis = 1)
	# g = Variable((Rdiff.shape[1], 1))
	# obj = sum_squares(rnew - Rdiff*g)
	# prob = Problem(Minimize(obj))
	# prob.solve()
	# return g.value

def dual_update(yvals, primals, rho):
	""" Dual variable update for the accelerated ADMM consensus problem.
	
	Parameters
	----------
	yvals : array
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
	weights = dual_weights(primals)
	return (yvals + rho*primals).dot(weights)
	
	# yRnew = yvals[:,-1] + rho*primals[:,-1]
	# if primals.shape[1] == 1:
	#	return yRnew
	# weights = dual_weights(primals)
	# ydiff = np.diff(yvals, axis = 1)
	# Rdiff = np.diff(primals, axis = 1)
	# return yRnew - (ydiff + rho*Rdiff).dot(weights)
	
