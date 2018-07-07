import numpy as np
from cvxpy import *
from cvxconsensus import *

def main():
	# Solve the following consensus problem using ADMM:
	# Minimize sum(f_i(x)), where f_i(x) = square(norm(x - a_i))

	# Generate a_i's.
	np.random.seed(0)
	a = np.random.randn(3,10)
	
	# Construct separate problems.
	x = Variable(3)
	funcs = [square(norm(x - a_i)) for a_i in a.T]
	p_list = [Problem(Minimize(f_i)) for f_i in funcs]
	probs = Problems(p_list)
	probs.pretty_vars()
	
	# Solve via consensus.
	probs.solve(method = "consensus", rho_init = 5, max_iter = 50)
	print("Objective:", probs.value)
	print("Solution:", x.value)

if __name__ == '__main__':
	main()
