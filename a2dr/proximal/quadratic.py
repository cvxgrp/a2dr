import numpy as np
import numpy.linalg as LA
from scipy import sparse
import scipy.sparse.linalg as spLA
from a2dr.proximal.composition import prox_scale

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

def prox_sum_squares(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\|x\|_2^2`, where :math:`f(x) = \sum_i x_i^2`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d > 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_sum_squares_base, *args, **kwargs)(v, t)

def prox_sum_squares_base(v, t):
    """Proximal operator of :math:`tf(x) = t\|x\|_2^2 = t\sum_i x_i^2` for scalar t > 0.
    """
    return v / (1.0 + 2*t)

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