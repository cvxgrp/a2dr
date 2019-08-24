import numpy as np
import numpy.linalg as LA

def prox_psd_cone(B, t):
	B_symm = (B + A.T)/2.0
	if not np.allclose(B, B_symm):
		raise Exception("Proximal operator for positive semidefinite cone only operates on symmetric matrices.")
	s, u = LA.eigh(B_symm)
	s_new = np.maximum(s, 0)
	return u.dot(np.diag(s_new)).dot(u.T)