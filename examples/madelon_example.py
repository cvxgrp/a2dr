import os
import numpy as np
from cvxpy import *
from cvxconsensus import *
import matplotlib.pyplot as plt

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data/')

def compare_residuals(res_sdrs, res_aa2, m_vals):
    if not isinstance(res_aa2, list):
        res_aa2 = [res_aa2]
    if not isinstance(m_vals, list):
        m_vals = [m_vals]
    if len(m_vals) != len(res_aa2):
        raise ValueError("Must have same number of AA-II residuals as memory parameter values")

    plt.semilogy(range(res_sdrs.shape[0]), res_sdrs, label="S-DRS")
    for i in range(len(m_vals)):
        label = "AA-II S-DRS (m = {})".format(m_vals[i])
        plt.semilogy(range(res_aa2[i].shape[0]), res_aa2[i], linestyle="--", label=label)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.show()

def main():
	# Solve the following consensus problem using S-DRS with AA-II:
	# Minimize sum(f_i(x)) subject to x >= 0, where f_i(x) = sum_squares(A_i*x - b_i)
	
	# Problem parameters.
	m = 50 
	n = 100
	N = 5            # Size of data split.
	rho = 100        # Step size.
	max_iter = 2000  # Maximum iterations.
	eps_tol = 1e-8       # Stopping tolerance.
	eps_abs = 1e-16
	m_accel = 10     # Memory size for Anderson acceleration.
	
	# Import and split data.
	A = np.genfromtxt(os.path.join(DATA_DIR, 'madelon_train.data'), delimiter = ' ')
	b = np.genfromtxt(os.path.join(DATA_DIR, 'madelon_train.labels'), delimiter = ' ')
	A = A[:m,:n]/100
	b = b[:m]
	A_split = np.split(A, N)
	b_split = np.split(b, N)
	
	# Construct separate problems.
	x = Variable(n)
	constr = [x >= 0]
	p_list = []
	for A_sub, b_sub in zip(A_split, b_split):
		obj = sum_squares(A_sub*x - b_sub)
		# p_list += [Problem(Minimize(obj), constr)]
		p_list += [Problem(Minimize(obj))]
	p_list += [Problem(Minimize(0), constr)]
	probs = Problems(p_list)
	probs.pretty_vars()
	
	# Solve with consensus S-DRS.
	obj_sdrs = probs.solve(method = "consensus", rho_init = rho, max_iter = max_iter, \
						   warm_start = False, eps_stop = eps_tol, eps_abs = eps_abs,
						   anderson = False)
	res_sdrs = probs.residuals
	print("S-DRS Objective:", obj_sdrs)

	# Solve with consensus S-DRS using AA-II.
	obj_aa2 = probs.solve(method = "consensus", rho_init = rho, max_iter = max_iter, \
						  warm_start = False, eps_stop = eps_tol, eps_abs = eps_abs, \
						  anderson = True, m_accel = m_accel)
	res_aa2 = probs.residuals
	print("S-DRS with AA-II Objective:", obj_aa2)
	compare_residuals(res_sdrs, res_aa2, m_accel)
	
	# Solve combined problem.
	obj_comb = probs.solve(method = "combined")
	print("Combined Objective:", obj_comb)

if __name__ == '__main__':
	main()
