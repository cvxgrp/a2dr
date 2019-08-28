"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.sparse as sp

def aa_weights(Y, g, reg = 0, type = "lstsq", *args, **kwargs):
	""" Solve the constrained least-squares problem
		Minimize sum_squares(\\sum_{j=0}^m w_j * G^(k-m+j))
			subject to \\sum_{j=0}^m w_j = 1.
		with respect to w \\in \\reals^{m+1}.

	This can be transformed via a change of variables
		w_0 = c_0, w_j = c_j - c_{j-1} for j = 1,...,m-1, and w_m = 1 - c_{m-1}
	into the unconstrained problem
		Minimize sum_squares(g - Y*c)
	with respect to c \\in \\reals^m, where g_i = G^(i) and Y_k = [y_{k-m},...,y_{k-1}]
	for y_i = g_{i+1} - g_i.

	We add a regularization term for stability, so the final problem we solve is
		Minimize sum_squares(g - Y*c) + \\lambda*sum_squares(c)
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
