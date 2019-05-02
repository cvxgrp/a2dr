import numpy as np
from scipy.linalg import block_diag

def precondition(p_list, A_list, b, tol = 1e-6, max_iter = 1000):
    n_list = [A.shape[1] for A in A_list]
    A = np.hstack(A_list)
    d, e, A_hat, k = mat_equil(A, n_list, tol, max_iter)

    split_idx = np.cumsum(n_list)
    A_eq_list = np.split(A_hat, split_idx, axis=1)[:-1]
    p_eq_list = [lambda v, rho: prox(ei*v, rho/ei**2)/ei for prox, ei in zip(p_list, e)]
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
    
    # form the size m-by-N matrix A_block, whose (i,j)-th entry is \sum_{k=n_1+...+n_{j-1}}}^{n_1+...+n_j} A_{ik}^2
    gamma = (m + N)/m/N * np.sqrt(np.finfo(float).eps)
    A2 = A ** 2 
    ave_list = [np.ones([n_i,1]) for n_i in n_list]
    A_block = A2.dot(block_diag(*ave_list))
    A_block_T = A_block.transpose()
    
    # apply regularized Sinkhorn-Knopp on A_block
    for k in range(max_iter):
        d1 = N / (A_block.dot(e) + N * gamma * em)
        e1 = m / (A_block_T.dot(d) + m * gamma * eN)
        err_d = np.linalg.norm(d1 - d)
        err_e = np.linalg.norm(e1 - e)
        d = d1
        e = e1
        if err_d <= tol and err_e <= tol:
            break
    
    d = np.sqrt(d)
    e = np.sqrt(e)
    I_list = [np.eye(n_list[i]) * e[i] for i in range(N)]
    E = block_diag(*I_list)
    D = np.diag(d)
    B = D.dot(A.dot(E))
    
    # rescale to have \|DAE\|_2 close to 1
    scale = np.linalg.norm(B, 'fro') / np.sqrt(np.min([m,N]))
    d_mean = np.mean(d)
    e_mean = np.mean(e)
    q = np.log(d_mean / e_mean * scale) / 2 / np.log(scale)
    d = d * (scale ** (-q))
    e = e * (scale ** (q-1))
    B = B / scale
    
    return d, e, B, k