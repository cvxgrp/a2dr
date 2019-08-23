import numpy as np
from scipy import sparse
from scipy.special import lambertw

def prox_abs(v, t):
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
		min_elemwise = lambda x, y: x.minimum(y)
	else:
		max_elemwise = np.maximum
		min_elemwise = np.minimum
	return max_elemwise(v - 1/t, 0) + min_elemwise(v + 1/t, 0)

def prox_entr(v, t):
	return lambertw(t*v - 1) * np.log(t) / t

def prox_exp(v, t):
	return v - lambertw(np.exp(v - np.log(t)))

def prox_huber(v, t):
	return v / (1 + 1/t) if np.abs(v) < (1 + 1/t) else v - np.sign(v)/t
