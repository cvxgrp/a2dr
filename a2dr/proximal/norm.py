import numpy as np
from scipy import sparse
from a2dr.proximal.projection import proj_l1
from a2dr.proximal.composition import prox_scale

def prox_norm1(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\|x\|_2^2`, where :math:`f(x) = \|x\|_1`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d > 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_norm1_base, *args, **kwargs)(v, t)

def prox_norm2(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\|x\|_2^2`, where :math:`f(x) = \|x\|_2`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d > 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_norm2_base, *args, **kwargs)(v, t)

def prox_norm_inf(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\|x\|_2^2`, where :math:`f(x) = \|x\|_{\infty}`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d > 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_norm_inf_base, *args, **kwargs)(v, t)

def prox_norm1_base(v, t):
	"""Proximal operator of :math:`f(x) = \|x\|_1`.
	"""
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
	else:
		max_elemwise = np.maximum
	return max_elemwise(v - t, 0) - max_elemwise(-v - t, 0)

def prox_norm2_base(v, t):
	"""Proximal operator of :math:`f(x) = \|x\|_2`.
	"""
	if sparse.issparse(v):
		norm = sparse.linalg.norm
		zeros = sparse.csr_matrix
		max_elemwise = lambda x, y: x.maximum(y)
	else:
		norm = np.linalg.norm
		zeros = np.zeros
		max_elemwise = np.maximum

	v_norm = norm(v,'fro')
	if v_norm == 0:
		return zeros(v.shape)
	else:
		return max_elemwise(1 - t*1.0/v_norm, 0) * v

def prox_norm_inf_base(v, t):
	"""Proximal operator of :math:`f(x) = \|x\|_{\infty}`.
	"""
	# TODO: Sparse handling.
	return v - t * proj_l1(v/t)

def prox_norm_nuc(B, t):
	U, s, Vt = np.linalg.svd(B, full_matrices=False)
	s_new = np.maximum(s - t, 0)
	return U.dot(np.diag(s_new)).dot(Vt)

def prox_group_lasso(B, t):
	# TODO: Sparse handling.
	return np.concatenate([prox_norm2(B[:,j], t) for j in range(B.shape[1])])