import numpy as np
from cvxpy import *
from cvxconsensus import *

def main():
	# Solve the following consensus problem using ADMM:
	# Minimize sum_squares(A*x - b) + gamma*norm(x,1)
	
	# Problem data.
	np.random.seed(1)
	m = 100
	n = 75
	DENSITY = 0.75

	# Problem data.
	A = np.random.randn(m,n)
	xtrue = np.random.randn(n)
	idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
	for idx in idxs:
		xtrue[idx] = 0
	b = A.dot(xtrue) + np.random.randn(m)
	
	# Separate penalty from regularizer.
	x = Variable(n)
	gamma = Parameter(nonneg = True)
	funcs = [sum_squares(A*x - b), gamma*norm(x,1)]
	p_list = [Problem(Minimize(f)) for f in funcs]
	probs = Problems(p_list)

	# Solve via consensus.
	gamma.value = 1.0
	probs.solve(method = "consensus", rho_init = 1.0, max_iter = 1000)
	print("Objective:", probs.value)
	print("Solution:", x.value)

if __name__ == '__main__':
	main()
