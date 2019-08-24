import numpy as np
from cvxpy import *
from a2dr import *

# Adapted from EE364B, Exercise 5.1: Distributed method for bi-commodity network flow problem.
# MatLAB code: bicommodity.m, bicommodity_data.m
def main():
    # Incidence matrix.
    A = np.array([[-1, +1,  0,  0,  0,  0,  0,  0, -1],
                  [+1,  0, -1,  0,  0,  0,  0, -1,  0],
                  [ 0, +1, +1, -1,  0,  0,  0,  0,  0],
                  [ 0,  0,  0, +1, -1, -1,  0,  0,  0],
                  [ 0,  0,  0,  0, +1,  0, +1,  0, +1],
                  [ 0,  0,  0,  0,  0, +1, -1, +1,  0]])

    # Problem data.
    p, n = A.shape
    s = np.array([+1,  0, 0, 0, -1,  0])   # Source rates.
    NUM_SPLITS = 3   # Number of splits by node.

    x_star = Variable(n)
    obj = sum_squares(x_star)
    constr = [A * x_star + s == 0, x_star >= 0]
    prob = Problem(Minimize(obj), constr)
    prob.solve()
    print("True Objective:", prob.value)
    print("True Flow x:", x_star.value)

    # Partition by node.
    A_split = np.split(A, NUM_SPLITS)
    s_split = np.split(s, NUM_SPLITS)

    p_list = []
    for A_sub, s_sub in zip(A_split, s_split):
        constr = [A_sub * x_star + s_sub == 0, x_star >= 0]
        p_list += [Problem(Minimize(obj / NUM_SPLITS), constr)]
    probs = Problems(p_list)

    # Solve via consensus.
    probs.solve(method="consensus", rho_init = 0.5, max_iter = 250)
    print("Node Consensus Objective:", probs.value)
    print("Node Consensus Flow x:", x_star.value)

if __name__ == '__main__':
	main()