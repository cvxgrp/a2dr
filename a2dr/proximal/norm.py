import numpy as np
from scipy import sparse
from a2dr.proximal.projection import proj_l1

def prox_norm1(v, t):
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
	else:
		max_elemwise = np.maximum
	return max_elemwise(v - t, 0) - max_elemwise(-v - t, 0)

def prox_norm2(v, t):
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

def prox_norm_inf(v, t):
	# TODO: Sparse handling.
	return v - t * proj_l1(v/t)

def prox_norm_nuc(B, t, order='C'):
	U, s, Vt = np.linalg.svd(B, full_matrices=False)
    s_new = np.maximum(s - t, 0)
    B_new = U.dot(np.diag(s_new)).dot(Vt)
    return B_new.ravel(order=order)

def prox_group_lasso(B, t):
	# TODO: Sparse handling.
	return np.concatenate([prox_norm2(B[:,j], t) for j in range(B.shape[1])])