import numpy as np
from scipy import sparse
from a2dr.proximal.interface import NUMPY_FUNS, SPARSE_FUNS, apply_to_nonzeros
from a2dr.proximal.composition import prox_scale

def prox_cardinality(v, t = 1, k = 10, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`card(x) \\leq k` for an integer k >= 0. The scalar t > 0, and the optional arguments are a = scale,
	b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default,
	k = 10, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	if not (int(k) == k and k >= 0):
		raise ValueError("k must be a non-negative integer.")
	return prox_scale(prox_cardinality_base, k, *args, **kwargs)(v, t)

def prox_rank(B, t = 1, k = 10, *args, **kwargs):
	"""Proximal operator of :math:`tf(aB-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`rank(B) \\leq k` for an integer k >= 0. The scalar t > 0, and the optional arguments are a = scale,
	b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default,
	k = 10, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	if not (int(k) == k and k >= 0):
		raise ValueError("k must be a non-negative integer.")
	return prox_scale(prox_rank_base, k, *args, **kwargs)(B, t)

def prox_boolean(v, t = 1, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`x \\in {0,1}^n`, i.e., all elements of :math:`x` are booleans. The scalar t > 0, and the optional
	arguments are a = scale, b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and
	d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	return prox_scale(prox_boolean_base, *args, **kwargs)(v, t)

def prox_integer(v, t = 1, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`x \\in Z^n`, i.e., all elements of :math:`x` are integers. The scalar t > 0, and the optional
	arguments are a = scale, b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and
	d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	return prox_scale(prox_integer_base, *args, **kwargs)(v, t)

def prox_cardinality_base(v,t,k):
	"""Proximal operator of the set indicator that :math:`card(x) \\leq k`, where k is a non-negative integer.
	"""
	if k == 0:
		FUNS = SPARSE_FUNS if sparse.issparse(v) else NUMPY_FUNS
		return FUNS["zeros"](v.shape)

	if sparse.issparse(v):
		if v.data.size <= k:
			return v
		v_new = v.copy()
		idx_zero = np.argpartition(abs(v.data), -k)[:-k]
		v_new.data[idx_zero] = 0
		v_new.eliminate_zeros()   # This may be computationally intensive.
	else:
		v_new = np.zeros(v.shape)
		idx_keep = np.argpartition(abs(v), -k, axis=None)[-k:]
		vals = np.take_along_axis(v, idx_keep, axis=None)
		np.put_along_axis(v_new, idx_keep, vals, axis=None)
	return v_new

def prox_rank_base(B,t,k):
	"""Proximal operator of the set indicator that :math:`rank(B) \\leq k`, where k is a non-negative integer.
	"""
	if k == 0:
		return np.zeros(B.shape)

	U, s, Vt = np.linalg.svd(B, full_matrices=False)
	U_new, s_new, Vt_new = U[:,:k], s[:k], Vt[:k,:]
	return U_new.dot(np.diag(s_new)).dot(Vt_new)

def prox_boolean_base(v,t):
	"""Proximal operator of the set indicator that :math:`x \\in {0,1}^n`, where :math:`x` is a vector of length n.
	"""
	return apply_to_nonzeros(lambda y: np.where(y >= 0.5, 1, 0), v)

def prox_integer_base(v,t):
	"""Proximal operator of the set indicator that :math:`x \\in Z^n`, where :math:`x` is a vector of length n and
	:math:`Z` denotes the space of integers.
	"""
	return apply_to_nonzeros(np.rint, v)
