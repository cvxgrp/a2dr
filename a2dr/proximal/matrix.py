import numpy as np
import numpy.linalg as LA
from scipy import sparse
from a2dr.proximal.projection import proj_simplex

def prox_neg_log_det(B, t):
	B_symm = (B + B.T) / 2.0
    if not (np.allclose(B, B_symm)):
        raise Exception("Proximal operator for negative log-determinant only operates on symmetric matrices.")
    s, u = LA.eigh(B_symm)
    s_new = (s + np.sqrt(w**2 + 4*t)) / 2
    return u.dot(np.diag(s_new)).dot(u.T)

def prox_sigma_max(B, t):
	U, s, Vt = LA.svd(B, full_matrices=False)
	s_new = t * proj_simplex(s/t)
	return U.dot(np.diag(s_new)).dot(Vt)

def prox_trace(B, t):
	return B - np.diag(np.full((B.shape[0],), t))