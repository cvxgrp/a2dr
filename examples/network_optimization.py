import numpy as np
from cvxpy import *
from cvxconsensus import *

# Reference:
# P. Yin, S. Diamond, B. Lin, and S. Boyd. "Network Optimization for Unified Packet and Circuit Switched Networks."
#    https://stanford.edu/~boyd/papers/net_opt.html
# Dataset:
# J. Fang, Y. Vardi, and C.-H. Zhang. "An iterative tomogravity algorithm for the estimation of network traffic."
#    Complex Datasets and Inverse Problems: Tomography, Networks and Beyond, vol. 54. IMS, 2007.
#    http://www.cs.utexas.edu/~yzhang/research/AbileneTM/

# Utility fairness function.
def fairness(f, alpha):
    if alpha == 1:
        return log(f)
    elif alpha >= 0:
        return f ** (1 - alpha) / (1 - alpha)
    else:
        raise ValueError("alpha must be a non-negative scalar")

def main():
    # Problem data.
    A = np.loadtxt("data/abilene_topology.csv", delimiter = ",")   # Adjacency matrix.
    c = np.loadtxt("data/abilene_capacity.csv", delimiter = ",")   # Edge capacities.
    n,m = A.shape   # n = Number of nodes, m = Number of (directed) edges.
    alpha = 2   # Fairness parameter.

    T = Variable((n,n))   # Traffic demand matrix.
    F = Variable((n,m), nonneg = True)   # Network flow matrix.

    # Formulate constraints.
    ones = np.ones(n)
    on_diag = np.eye(n, dtype = bool)
    constr = [T + F.dot(A.T) == 0, F.T.dot(ones) == c, T.dot(ones) == 0, T[~on_diag] >= 0]

    # Real-time-based allocation.
    phi = T/r_real
    utility = fairness(phi, alpha)
    prob = Problem(Maximize(sum(utility)), constr)
    prob.solve(solver = "MOSEK")

if __name__ == '__main__':
	main()
