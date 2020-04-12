"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import block_diag, issparse, csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import norm
import scipy.linalg as sLA
from scipy.stats.mstats import gmean

def precondition(p_list, A_list, b, tol = 1e-3, max_iter = 1000):
    # print('### Preconditioning starts ...')
    if all([Ai.size == 0 for Ai in A_list]):
        return p_list, A_list, b, np.ones(len(A_list))

    n_list = [A.shape[1] for A in A_list]
    sparse_check = [sparse.issparse(A) for A in A_list]
    # Enforce csr format for better matrix operation efficiency.
    if np.sum(sparse_check) == 0:      # all dense
        A = csr_matrix(np.hstack(A_list))
    elif np.prod(sparse_check) == 1:   # all sparse
        A = csr_matrix(sparse.hstack(A_list))
    else:
        A_list_csr = [csr_matrix(A) for A in A_list]
        A = csr_matrix(sparse.hstack(A_list))
    d, e, A_hat, k = mat_equil(A, n_list, tol, max_iter)

    split_idx = np.cumsum(n_list)
    split_idx = np.hstack([0, split_idx])
    A_hat = csc_matrix(A_hat)   # faster column slicing
    A_eq_list = [A_hat[:,split_idx[i]:split_idx[i+1]] for i in range(len(n_list))]
    A_eq_list = [csr_matrix(A_eq_list[i]) for i in range(len(A_eq_list))]   # change back to csr format

    # Note: We must do it this way because the standard pythonic list comprehension, i.e., [f(x) for x in iterable]
    # will create *duplicate* function handles, leading to incorrect results! This is due to late binding:
    # https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
    # https://docs.python-guide.org/writing/gotchas/#late-binding-closures
    def proto(i, p_list, e):
        return lambda v, t: p_list[i](e[i]*v, t*e[i]**2)/e[i]
    p_eq_list = list(map(lambda i: proto(i,p_list,e), range(len(p_list))))
    return p_eq_list, A_eq_list, d*b, e

def mat_equil(A, n_list, tol, max_iter):
    # Block matrix equilibration using regularized Sinkhorn-Knopp
    # Reference: [POGS] http://stanford.edu/~boyd/papers/pdf/pogs.pdf
    '''
    1. Input
    A: a numpy 2d-array or matrix of size m-by-n to be equilibrated
    n_list: a list of size N containing size n_i of each variable block i (n_1+...+n_N = n)
    tol: tolerance for termination
    max_iter: maximum number of iterations
   
    2. Output
    d: left scaling vector (of size m)
    e: right scaling vector (of size N)
    B: equilibrated matrix B = diag(d) A diag(e_1I_{n_1},...,e_NI_{n_N})
    k: number of iterations terminated
    
    3. Requirement:
    import numpy as np
    from scipy.linalg import block_diag
    '''
    
    N = len(n_list)
    m = A.shape[0]
    em, eN = np.ones(m), np.ones(N)
    d, e = em, eN
    
    # Form the size m-by-N matrix A_block, whose (i,j)-th entry is \sum_{k=n_1+...+n_{j-1}}}^{n_1+...+n_j} A_{ik}^2
    gamma = (m + N)/m/N * np.sqrt(np.finfo(float).eps)
    A2 = A.power(2) if issparse(A) else np.power(A,2)
    ave_list = [np.ones([n_i,1]) for n_i in n_list]
    A_block = A2.dot(block_diag(ave_list))
    A_block_T = A_block.transpose()
    # print('Block matrix shape = {}'.format(A_block.shape))
    # print('gamma={}'.format(gamma))
    
    # Apply regularized Sinkhorn-Knopp on A_block
    for k in range(max_iter):
        d1 = N / (A_block.dot(e) + N * gamma * em)
        e1 = m / (A_block_T.dot(d1) + m * gamma * eN)
        err_d = np.linalg.norm(d1 - d)
        err_e = np.linalg.norm(e1 - e)
        d = d1
        e = e1
        # print('k={}, err_d={}, err_e={}'.format(k, err_d/np.sqrt(m), err_e/np.sqrt(N)))
        if err_d/np.sqrt(m) <= tol and err_e/np.sqrt(N) <= tol:
            break
    
    d = np.sqrt(d)
    e = np.sqrt(e)
    I_list = [sparse.eye(n_list[i]) * e[i] for i in range(N)]
    E = csr_matrix(block_diag(I_list))
    D = csr_matrix(diags(d))
    # print('generate D, E')
    B = D.dot(csr_matrix(A).dot(E))
    # print('compute scaled matrix')
    
    # Rescale to have \|DAE\|_2 close to 1
    scale = norm(B, 'fro') / np.sqrt(np.min([m,N]))
    d_mean = gmean(d)
    e_mean = gmean(e)
    q = np.log(d_mean / e_mean * scale) / 2 / np.log(scale)
    d = d * (scale ** (-q))
    e = e * (scale ** (q-1))
    B = B / scale
    
    return d, e, B, k
