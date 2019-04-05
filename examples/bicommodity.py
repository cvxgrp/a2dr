import numpy as np
from cvxpy import *
from cvxconsensus import *

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

    # Source rates.
    s = np.array([+1,  0, 0, 0, -1,  0])
    t = np.array([ 0, +1, 0, 0,  0, -1])

    # Problem data.
    p, n = A.shape
    eps = 0.1
    NUM_SPLITS = 3   # Number of splits by node.

    x_star = Variable(n)
    y_star = Variable(n)
    obj = sum_squares(x_star + y_star) + eps*(sum_squares(x_star) + sum_squares(y_star))
    # obj = sum((x_star + y_star)**2 + eps*(x_star**2 + y_star**2))
    constr = [A*x_star + s == 0, A*y_star + t == 0, x_star >= 0, y_star >= 0]
    prob = Problem(Minimize(obj), constr)
    prob.solve()
    print("True Objective:", prob.value)
    print("True Flow x:", x_star.value)
    print("True Flow y:", y_star.value)

    # Partition by commodity.
    obj1 = 0.5*sum_squares(x_star + y_star) + eps*sum_squares(x_star)
    obj2 = 0.5*sum_squares(x_star + y_star) + eps*sum_squares(y_star)
    # obj1 = sum(0.5*(x_star + y_star)**2 + eps*x_star**2)
    # obj2 = sum(0.5*(x_star + y_star)**2 + eps*y_star**2)
    constr1 = [A*x_star + s == 0, x_star >= 0]
    constr2 = [A*y_star + t == 0, y_star >= 0]
    p_list = [Problem(Minimize(obj1), constr1), Problem(Minimize(obj2), constr2)]
    probs = Problems(p_list)

    # Solve via consensus.
    probs.solve(method = "consensus")
    print("Commodity Consensus Objective:", probs.value)
    print("Commodity Consensus Flow x:", x_star.value)
    print("Commodity Consensus Flow y:", y_star.value)

    # Partition by node.
    A_split = np.split(A, NUM_SPLITS)
    s_split = np.split(s, NUM_SPLITS)
    t_split = np.split(t, NUM_SPLITS)

    p_list = []
    for A_sub, s_sub, t_sub in zip(A_split, s_split, t_split):
        constr = [A_sub*x_star + s_sub == 0, A_sub*y_star + t_sub == 0, x_star >= 0, y_star >= 0]
        p_list += [Problem(Minimize(obj/NUM_SPLITS), constr)]
    probs = Problems(p_list)

    # Solve via consensus.
    probs.solve(method = "consensus", rho_init = 0.5, max_iter = 250)
    print("Node Consensus Objective:", probs.value)
    print("Node Consensus Flow x:", x_star.value)
    print("Node Consensus Flow y:", y_star.value)

if __name__ == '__main__':
	main()
