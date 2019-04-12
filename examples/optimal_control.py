import numpy as np
from cvxpy import *
from cvxconsensus import *

# Reference:
# B. O'Donoghue, G. Stathopoulos, and S. Boyd. "A Splitting Method for Optimal Control."
#    http://stanford.edu/~boyd/papers/oper_splt_ctrl.html
def main():
    np.random.seed(1)
    T = 50
    n = 20
    m = 30

    A = np.random.randn(n,n)
    B = np.random.randn(n,m)
    x_init = np.random.randn(n)
    A_eig = np.linalg.eigvals(A)
    A = A/np.max(np.abs(A_eig))   # Scale so maximum eigenvalue has magnitude of one.

    Q = np.random.randn(n,n)
    R = np.random.randn(m,m)
    Q = Q.T.dot(Q)               # Q is positive semidefinite.
    R = R.T.dot(R) + np.eye(m)   # R is positive definite.

    x = []
    u = []
    phi = []
    constr_dyn = []
    constr_psi = []
    for t in range(T+1):
        x.append(Variable(n))
        u.append(Variable(m))
        phi.append(quad_form(x[t], Q) + quad_form(u[t], R))
        constr_psi.append(norm_inf(u[t]) <= 1)
        if t > 0:
            constr_dyn.append(x[t] == A*x[t-1] + B*u[t-1])
    constr_dyn.append(x[0] == x_init)
    prob = Problem(Minimize(sum(phi)), constr_psi + constr_dyn)
    prob.solve()
    print("True Objective:", prob.value)

    p_list = [Problem(Minimize(sum(phi)), constr_dyn), Problem(Minimize(0), constr_psi)]
    probs = Problems(p_list)
    probs.solve(method = "consensus", rho_init = 1.0, max_iter = 250)
    print("Consensus Objective:", probs.value)

if __name__ == '__main__':
    main()