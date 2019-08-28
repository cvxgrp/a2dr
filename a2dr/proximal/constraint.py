import numpy as np
import numpy.linalg as LA
from scipy import sparse

def prox_box_constr(v, t = 1, v_lo = -np.inf, v_hi = np.inf):
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
		min_elemwise = lambda x, y: x.minimum(y)
	else:
		max_elemwise = np.maximum
		min_elemwise = np.minimum
	return min_elemwise(max_elemwise(v, v_lo), v_hi)

def prox_psd_cone(B, t = 1):
	B_symm = (B + B.T)/2.0
	if not np.allclose(B, B_symm):
		raise Exception("Proximal operator for positive semidefinite cone only operates on symmetric matrices.")
	s, u = LA.eigh(B_symm)
	s_new = np.maximum(s, 0)
	return u.dot(np.diag(s_new)).dot(u.T)