import numpy as np
from scipy import sparse
from a2dr.proximal.interface import NUMPY_FUNS, SPARSE_FUNS
from a2dr.proximal.composition import prox_scale

def prox_box_constr(v, t = 1, v_lo = -np.inf, v_hi = np.inf, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`\\underline x \\leq x \\leq \\overline x`. Here the lower/upper bounds are (v_lo, v_hi), which default
	to (-Inf, Inf). The scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and 
	d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	return prox_scale(prox_box_constr_base, v_lo, v_hi, *args, **kwargs)(v, t)

def prox_nonneg_constr(v, t = 1, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`x \\geq 0`. The scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
	d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	return prox_scale(prox_nonneg_constr_base, *args, **kwargs)(v, t)

def prox_nonpos_constr(v, t = 1, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`x \\leq 0`. The scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
	d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	return prox_scale(prox_nonpos_constr_base, *args, **kwargs)(v, t)

def prox_psd_cone(B, t = 1, *args, **kwargs):
	"""Proximal operator of :math:`tf(aB-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`B \\succeq 0` for :math:`B` a symmetric matrix. The scalar t > 0, and the optional arguments are
	a = scale, b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default,
	t = 1, a = 1, b = 0, c = 0, and d = 0.
	"""
	if np.isscalar(B):
		B = np.array([[B]])
	if B.shape[0] != B.shape[1]:
		raise ValueError("B must be a square matrix.")
	B_symm = (B + B.T) / 2.0
	if not np.allclose(B, B_symm):
		raise ValueError("B must be a symmetric matrix.")
	return prox_scale(prox_psd_cone_base, *args, **kwargs)(B_symm, t)

def prox_soc(v, t = 1, *args, **kwargs):
	"""Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f` is the set indicator that
	:math:`\\|x_{1:n}\\|_2 \\leq x_{n+1}`. The scalar t > 0, and the optional arguments are a = scale, b = offset,
	c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, 	t = 1, a = 1, b = 0,
	c = 0, and d = 0.
	"""
	if np.isscalar(v) or not (len(v.shape) == 1 or (len(v.shape) == 2 and v.shape[1] == 1)):
		raise ValueError("v must be a vector")
	if v.shape[0] < 2:
		raise ValueError("v must have at least 2 elements.")
	return prox_scale(prox_soc_base, *args, **kwargs)(v, t)

def prox_box_constr_base(v, t, v_lo, v_hi):
	"""Proximal operator of the set indicator that :math:`\\underline x \\leq x \\leq \\overline x`.
	"""
	FUNS = SPARSE_FUNS if sparse.issparse(v) else NUMPY_FUNS
	max_elemwise, min_elemwise = FUNS["max_elemwise"], FUNS["min_elemwise"]
	return min_elemwise(max_elemwise(v, v_lo), v_hi)

def prox_nonneg_constr_base(v, t):
	"""Proximal operator of the set indicator that :math:`x \\geq 0`.
	"""
	# return prox_box_constr_base(v, t, 0, np.inf)
	# return v.maximum(0) if sparse.issparse(v) else np.maximum(v,0)
	FUNS = SPARSE_FUNS if sparse.issparse(v) else NUMPY_FUNS
	max_elemwise = FUNS["max_elemwise"]
	return max_elemwise(v, 0)

def prox_nonpos_constr_base(v, t):
	"""Proximal operator of the set indicator that :math:`x \\leq 0`.
	"""
	# return prox_box_constr_base(v, t, -np.inf, 0)
	# return v.minimum(0) if sparse.issparse(v) else np.minimum(v,0)
	FUNS = SPARSE_FUNS if sparse.issparse(v) else NUMPY_FUNS
	min_elemwise = FUNS["min_elemwise"]
	return min_elemwise(v, 0)

def prox_psd_cone_base(B, t):
	"""Proximal operator of the set indicator that :math:`B \\succeq 0`, where :math:`B` is a symmetric matrix.
	"""
	# B_symm = (B + B.T)/2.0
	# if not np.allclose(B, B_symm):
	#	raise ValueError("B must be a symmetric matrix.")
	# s, u = np.linalg.eigh(B_symm)
	s, u = np.linalg.eigh(B)
	s_new = np.maximum(s, 0)
	return u.dot(np.diag(s_new)).dot(u.T)

def prox_soc_base(v, t):
	"""Proximal operator of the set indicator that :math:`\\|v_{1:n}\\_2 \\leq v_{n+1}`, where :math:`v` is a vector of
	length n+1, `v_{1:n}` symbolizes its first n elements, and :math:`v_{n+1}` is its last element. This is equivalent
	to the projection of :math:`v` onto the second-order cone :math:`C = {(u,s):\\|u\\|_2 \\leq s}`.
	Parikh and Boyd (2013). "Proximal Algorithms." Foundations and Trends in Optimization. vol. 1, no. 3, Sect. 6.3.2.
	"""
	if sparse.issparse(v):
		v = v.tocsr()
		u = v[:-1]  # u = (v_1,...,v_n)
		s = v[-1]   # s = v_{n+1}
		s = np.asscalar(s.todense())

		u_norm = sparse.linalg.norm(u,'fro')
		if u_norm <= -s:
			return np.zeros(v.shape)
		elif u_norm <= s:
			return v
		else:
			scale = (1 + s / u_norm) / 2
			return scale * sparse.vstack((u, u_norm))
	else:
		u = v[:-1]  # u = (v_1,...,v_n)
		s = v[-1]   # s = v_{n+1}
		s = np.asscalar(s)

		u_norm = np.linalg.norm(u,2)
		if u_norm <= -s:
			return np.zeros(v.shape)
		elif u_norm <= s:
			return v
		else:
			scale = (1 + s / u_norm) / 2
			u_all = np.zeros(v.shape)
			u_all[:-1] = u
			u_all[-1] = u_norm
			return scale * u_all
