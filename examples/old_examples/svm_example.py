import numpy as np
from cvxpy import *
from a2dr import *

NUM_PROCS = 4
SPLIT_SIZE = 250

def main():
	# Problem data.
	np.random.seed(1)
	N = NUM_PROCS*SPLIT_SIZE
	n = 10
	offset = np.random.randn(n,1)
	data = []
	for i in range(int(N/2)):
		data += [(1, offset + np.random.normal(1.0, 2.0, (n, 1)))]
	for i in range(int(N/2)):
		data += [(-1, offset + np.random.normal(-1.0, 2.0, (n, 1)))]
	data_splits = [data[i:i+SPLIT_SIZE] for i in range(0, N, SPLIT_SIZE)]
	
	# Construct problem.
	w = Variable(n + 1)
	def svm(data):
		slack = [pos(1 - b*(a.T*w[:-1] - w[-1])) for (b, a) in data]
		return norm(w, 2) + sum(slack)
	funcs = map(svm, data_splits)
	p_list = [Problem(Minimize(f_i)) for f_i in funcs]
	probs = Problems(p_list)
	
	# Solve via consensus using spectral step size adjustment.
	probs.solve(method = "consensus", rho_init = 1.0, max_iter = 20)
	print("Objective:", probs.value)
	print("Solution:", w.value)
	
	# Count misclassifications.
	def get_error(w):
		error = 0
		for label, sample in data:
			if not label*(np.dot(w[:-1].T, sample) - w[-1])[0] > 0:
				error += 1
		return "%d misclassifications out of %d samples" % (error, N)
	print("Misclassifications:", get_error(w.value))
	
if __name__ == '__main__':
	main()
