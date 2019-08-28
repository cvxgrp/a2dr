import numpy as np
import numpy.linalg as LA
from scipy import sparse
from a2dr.proximal.composition import prox_scale

def prox_box_constr(v, t = 1, v_lo = -np.inf, v_hi = np.inf, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`\\underline x \\leq x \\leq \\overline x`. The scalar t > 0, and the optional arguments are a = scale,
	b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d > 0. By default, t = 1,
	a = 1, b = 0, c = 0, and d = 0.
	"""
	return prox_scale(prox_box_constr_base, v_lo, v_hi, *args, **kwargs)(v, t)

def prox_box_constr_base(v, t, v_lo, v_hi):
	"""Proximal operator of the set indicator that :math:`\\underline x \\leq x \\leq \\overline x`.
	"""
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
		min_elemwise = lambda x, y: x.minimum(y)
	else:
		max_elemwise = np.maximum
		min_elemwise = np.minimum
	return min_elemwise(max_elemwise(v, v_lo), v_hi)

def prox_psd_cone(B, t):
	B_symm = (B + B.T)/2.0
	if not np.allclose(B, B_symm):
		raise Exception("Proximal operator for positive semidefinite cone only operates on symmetric matrices.")
	s, u = LA.eigh(B_symm)
	s_new = np.maximum(s, 0)
	return u.dot(np.diag(s_new)).dot(u.T)
