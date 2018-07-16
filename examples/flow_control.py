import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from cvxpy import *
from cvxconsensus import *

def main():
	# Problem data.
	m = 40
	m_a = 20
	n = 22
	n_a = 5
	n_b = 5
	np.random.seed(1)
	R = np.vstack((np.hstack((np.round(rand(m_a,n_a)), np.zeros((m_a,n_b)), np.round(rand(m_a,n-n_a-n_b)))),
				   np.hstack((np.zeros((m-m_a,n_a)), np.round(rand(m-m_a,n_b)), np.round(rand(m-m_a,n-n_a-n_b))))
				 ))
	c = 5*rand(m)
	
	# Find optimum directly.
	f_star = Variable(n)
	prob = Problem(Maximize(sum(sqrt(f_star))), [R*f_star <= c])
	prob.solve()
	print("True Objective:", prob.value)
	print("True Solution:", f_star.value)
	
	# Partition data into two groups with overlap.
	R_a = R[:m_a,:n_a]
	R_b = R[m_a:,n_a:(n_a + n_b)]
	S_a = R[:m_a,(n_a + n_b):]
	S_b = R[m_a:,(n_a + n_b):]
	c_a = c[:m_a]
	c_b = c[m_a:]
	n_ab = n - n_a - n_b
	
	# Define separate problem for each group.
	f_a = Variable(n_a)
	f_b = Variable(n_b)
	x = Variable(n_ab)
	p_list = [Problem(Maximize(sum(sqrt(f_a)) + 0.5*sum(sqrt(x))), [R_a*f_a + S_a*x <= c_a]),
		      Problem(Maximize(sum(sqrt(f_b)) + 0.5*sum(sqrt(x))), [R_b*f_b + S_b*x <= c_b])]
	probs = Problems(p_list)
	
	# Solve via consensus.
	probs.solve(method = "consensus", rho_init = 10, max_iter = 20)
	print("Consensus Objective:", -probs.value)   # TODO: All problems recast as minimization, so flip sign of objective to compare
	print("Consensus Solution:", np.hstack((f_a.value, f_b.value, x.value)))

if __name__ == '__main__':
	main()
