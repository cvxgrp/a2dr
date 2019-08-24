import numpy as np
import numpy.linalg as LA
from scipy import sparse
import scipy.sparse.linalg as spLA

def prox_quad_form(v, t, Q):
    if sparse.issparse(Q):
        Q_min_eigval = spLA.eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        if np.iscomplex(Q_min_eigval) or Q_min_eigval < 0:
            raise Exception("Q must be a symmetric positive semidefinite matrix.")
        return spLA.lsqr(Q + (1/t)*sparse.eye(v.shape[0]), v/t, atol=1e-16, btol=1e-16)[0]
    else:
        if not np.all(LA.eigvalsh(Q) >= 0):
            raise Exception("Q must be a symmetric positive semidefinite matrix.")
        return LA.lstsq(Q + (1/t)*np.eye(v.shape[0]), v/t, rcond=None)[0]

def prox_sum_squares(v, t):
    return v / (1.0 + t/2)

def prox_sum_squares_affine(v, t, X, y, method = "lsqr"):
    if X.shape[0] != y.shape[0]:
        raise ValueError("Dimension mismatch: nrow(X) != nrow(y)")
    if X.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch: ncol(X) != nrow(v)")

    n = v.shape[0]
    if method == "lsqr":
        X = sparse.csr_matrix(X)
        X_stack = sparse.vstack([X, 1/np.sqrt(2*t)*sparse.eye(n)])
        y_stack = np.concatenate([y, 1/np.sqrt(2*t)*v])
        return spLA.lsqr(X_stack, y_stack, atol=1e-16, btol=1e-16)[0]
    elif method == "lstsq":
        X_stack = np.vstack([X, 1/np.sqrt(2*t)*np.eye(n)])
        y_stack = np.concatenate([y, 1/np.sqrt(2*t)*v])
        return LA.lstsq(X_stack, y_stack, rcond=None)[0]
    else:
        raise ValueError("Method not supported:", type)