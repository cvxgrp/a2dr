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
import scipy.sparse as sp
from cvxpy import Variable, Problem, Minimize
from cvxconsensus.utilities import dicts_to_arr

def aa_weights_alt(residuals, lam=None, type="exact", *args, **kwargs):
	""" Solve the constrained least-squares problem
	   Minimize sum_squares(\sum_{j=0}^m w[j]*r^(k+1-m+j))
	      subject to \sum_{j=0}^m w[j] = 1

	This can be transformed via a change of variables
	   w[0] = g[0], w[j] = g[j] - g[j-1] for j = 1,...,m-1, and w[m] = 1-g[m-1]
	into the unconstrained problem
	   Minimize sum_squares(r^(k+1) - \sum_{j=0}^{m-1} g[j]*(r^(k+2-m+j) - r^(k+1-m+j)))

	Parameters
	----------
	residuals : array
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
		reg = lam * np.eye(G.shape[1]) if lam else 0  # Stabilization with l2 penalty.
		gamma = np.linalg.lstsq(G.T.dot(G) + reg, e, rcond=None)[0]
		alpha = gamma / np.sum(gamma)
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

def aa_weights(Y, g, reg = 0, type = "lstsq", *args, **kwargs):
	""" Solve the constrained least-squares problem
		Minimize sum_squares(\sum_{j=0}^m w_j * G^(k-m+j))
			subject to \sum_{j=0}^m w_j = 1.
		with respect to w \in \reals^{m+1}.

	This can be transformed via a change of variables
		w_0 = c_0, w_j = c_j - c_{j-1} for j = 1,...,m-1, and w_m = 1 - c_{m-1}
	into the unconstrained problem
		Minimize sum_squares(g - Y*c)
	with respect to c \in \reals^m, where g_i = G^(i) and Y_k = [y_{k-m},...,y_{k-1}]
	for y_i = g_{i+1} - g_i.

	We add a regularization term for stability, so the final problem we solve is
		Minimize sum_squares(g - Y*c) + \lambda*sum_squares(c)
	and return w as defined above.
	"""
	if type == "lstsq":
		if reg != 0:
			m = Y.shape[1]
			Y = np.vstack([Y, np.sqrt(reg)*np.eye(m)])
			g = np.concatenate([g, np.zeros(m)])
		gamma = np.linalg.lstsq(Y, g, *args, **kwargs)[0]
	elif type == "lsqr":
		if reg != 0:
			m = Y.shape[1]
			Y = sp.csc_matrix(Y)
			Y = sp.vstack([Y, np.sqrt(reg)*sp.eye(m)])
			g = np.concatenate([g, np.zeros(m)])
		gamma = sp.linalg.lsqr(Y, g, *args, **kwargs)[0]
	else:
		raise ValueError("Algorithm type not supported:", type)
	gamma_diff = np.diff(gamma, n=1)
	alpha = np.concatenate(([gamma[0]], gamma_diff, [1-gamma[-1]]))
	return alpha
