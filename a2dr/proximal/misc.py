from a2dr.proximal.projection import proj_simplex

def prox_max(v, t):
	return v - t*proj_simplex(v/t)