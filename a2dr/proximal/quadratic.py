import numpy as np
import numpy.linalg as LA
from scipy import sparse
import scipy.sparse.linalg as spLA

def prox_sum_squares(v, t):
    return v / (1.0 + t/2)

def prox_quad_form(v, t, Q):
    if sparse.issparse(Q):
        Q_min_eigval = spLA.eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        if np.iscomplex(Q_min_eigval) or Q_min_eigval < 0:
            raise Exception("Q must be a symmetric positive semidefinite matrix.")
        return lambda v, t: spLA.lsqr(Q + (1/t)*sparse.eye(v.shape[0]), v/t, atol=1e-16, btol=1e-16)[0]
    else:
        if not np.all(LA.eigvalsh(Q) >= 0):
            raise Exception("Q must be a symmetric positive semidefinite matrix.")
        return lambda v, t: LA.lstsq(Q + (1/t)*np.eye(v.shape[0]), v/t, rcond=None)[0]
