import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cvxpy import *
from cvxconsensus import *

# Solve the following consensus problem:
# Maximize log_det(S) - trace(S*Y)
# subject to S is PSD and norm(S,1) <= alpha
# where Y and alpha >= 0 are parameters

# Problem data.
np.random.seed(0)
n = 10     # Dimension of matrix.
N = 1000   # Number of samples.
A = np.random.randn(n,n)
A[sp.sparse.rand(n, n, 0.85).todense().nonzero()] = 0
S_true = A.dot(A.T) + 0.05*np.eye(n)
R = np.linalg.inv(S_true)

# TODO: Compute multiple sample covariance matrices.
y_sample = sp.linalg.sqrtm(R).dot(np.random.randn(n, N))
Y = np.cov(y_sample)

# The alpha values for each attempt at generating a sparse inverse cov. matrix.
alphas = [10, 2, 1]

# Form the optimization problem with split
# f_0(x) = log_det(S), f_1(x) = -trace(S*Y), f_2(x) = I(norm(S,1) <= alpha)
# over the set of PSD matrices S.
S = Variable(shape=(n,n), PSD=True)
alpha = Parameter(nonneg=True)
constraints = [norm(S,1) <= alpha]

p_list = [Problem(Maximize(log_det(S))),
          Problem(Maximize(-trace(S*Y))),
          Problem(Maximize(0), constraints)]
probs = Problems(p_list)
probs.pretty_vars()

# Empty list of result matrices S.
Ss = []

# Solve the optimization problem for each value of alpha.
for a_val in alphas:
    # Set alpha parameter and solve optimization problem
    alpha.value = a_val
    # prob.solve(solver = "CVXOPT")
    # if prob.status != OPTIMAL:
    #    raise Exception('CVXPY Error')
    probs.solve(method = "consensus", rho_init = 1.0, max_iter = 1000)

    # If the covariance matrix R is desired, here is how it to create it.
    R_hat = np.linalg.inv(S.value)

    # Threshold S element values to enforce exact zeros:
    S_val = S.value
    S_val[np.abs(S_val) <= 1e-4] = 0

    # Store this S in the list of results for later plotting.
    Ss += [S_val]

    print('Completed optimization parameterized by alpha = {}, obj value = {}'.format(alpha.value, -probs.value))

# Plot properties.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create figure.
plt.figure(figsize=(12, 12))

# Plot sparsity pattern for the true covariance matrix.
plt.subplot(2, 2, 1)
plt.spy(S_true)
plt.title('Inverse of true covariance matrix', fontsize=16)

# Plot sparsity pattern for each result, corresponding to a specific alpha.
for i in range(len(alphas)):
    plt.subplot(2, 2, 2+i)
    plt.spy(Ss[i])
    plt.title('Estimated inv. cov matrix, $\\alpha$={}'.format(alphas[i]), fontsize=16)
plt.show()