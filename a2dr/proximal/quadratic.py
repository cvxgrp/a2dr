import numpy as np
from cvxpy import *
from scipy import sparse
from a2dr.proximal.composition import prox_scale

def prox_quad_form(v, t = 1, Q = None, method = "lsqr", *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = x^TQx` for symmetric
    :math:`Q \\succeq 0`, scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term,
    and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0,
    and d = 0.
    """
    if np.isscalar(v):
        v = np.array([v])
    if Q is None:
        raise ValueError("Q must be a matrix.")
    elif np.isscalar(Q):
        Q = np.array([[Q]])

    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if Q.shape[0] != v.shape[0]:
        raise ValueError("Dimension mismatch: nrow(Q) != nrow(v).")
    if sparse.issparse(Q):
        Q_min_eigval = sparse.linalg.eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        if np.iscomplex(Q_min_eigval) or Q_min_eigval < 0:
            raise ValueError("Q must be a symmetric positive semidefinite matrix.")
    else:
        if not np.all(np.linalg.eigvalsh(Q) >= 0):
            raise ValueError("Q must be a symmetric positive semidefinite matrix.")
    if method not in ["lsqr", "lstsq"]:
        raise ValueError("method must be either 'lsqr' or 'lstsq'")
    return prox_scale(prox_quad_form_base, Q=Q, method=method, *args, **kwargs)(v, t)

def prox_sum_squares(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\sum_i x_i^2`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_sum_squares_base, *args, **kwargs)(v, t)

def prox_sum_squares_affine(v, t = 1, F = None, g = None, method = "lsqr", *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\|Fx - g\\|_2^2`
        for matrix F, vector g, scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term,
        and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0,
        and d = 0.
        """
    if np.isscalar(v):
        v = np.array([v])
    if F is None:
        raise ValueError("F must be a matrix.")
    elif np.isscalar(F):
        F = np.array([[F]])
    if g is None:
        raise ValueError("g must be a vector.")
    elif np.isscalar(g):
        g = np.array([g])

    if F.shape[0] != g.shape[0]:
        raise ValueError("Dimension mismatch: nrow(F) != nrow(g).")
    if F.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch: ncol(F) != nrow(v).")
    if method not in ["lsqr", "lstsq"]:
        raise ValueError("method must be either 'lsqr' or 'lstsq'.")
    return prox_scale(prox_sum_squares_affine_base, F=F, g=g, method=method, *args, **kwargs)(v, t)

def prox_qp(v, t = 1, Q = None, q = None, F = None, g = None, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = x^TQx+q^Tx+I{Fx<=g}`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if Q is None:
        raise ValueError("Q must be a matrix")
    if q is None:
        raise ValueError("q must be a vector")
    if F is None:
        raise ValueError("F must be a matrix")
    if g is None:
        raise ValueError("g must be a vector")
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    if Q.shape[0] != q.shape[0]:
        raise ValueError("Dimension mismatch: nrow(Q) != nrow(q)")
    if Q.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch: ncol(Q) != nrow(v)")
    if F.shape[0] != g.shape[0]:
        raise ValueError("Dimension mismatch: nrow(F) != nrow(g)")
    if F.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch: ncol(F) != nrow(v)")
    return prox_scale(prox_qp_base(Q, q, F, g), *args, **kwargs)(v, t)

def prox_quad_form_base(v, t, Q, method = "lsqr"):
    """Proximal operator of :math:`f(x) = x^TQx`, where :math:`Q \\succeq 0` is a symmetric positive semidefinite matrix.
    """
    if method == "lsqr":
        # Q_min_eigval = spLA.eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        # if np.iscomplex(Q_min_eigval) or Q_min_eigval < 0:
        #    raise Exception("Q must be a symmetric positive semidefinite matrix.")
        Q = sparse.csr_matrix(Q)
        return sparse.linalg.lsqr(2*t*Q + sparse.eye(v.shape[0]), v, atol=1e-16, btol=1e-16)[0]
    elif method == "lstsq":
        # if not np.all(LA.eigvalsh(Q) >= 0):
        #    raise Exception("Q must be a symmetric positive semidefinite matrix.")
        return np.linalg.lstsq(2*t*Q + np.eye(v.shape[0]), v, rcond=None)[0]
    else:
        raise ValueError("method must be 'lsqr' or 'lstsq'")

def prox_sum_squares_base(v, t):
    """Proximal operator of :math:`f(x) = \\sum_i x_i^2`.
    """
    return v / (1.0 + 2*t)

def prox_sum_squares_affine_base(v, t, F, g, method = "lsqr"):
    """Proximal operator of :math:`f(x) = \\|Fx - g\\|_2^2`, where F is a matrix and g is a vector.
    """
    # if F.shape[0] != g.shape[0]:
    #    raise ValueError("Dimension mismatch: nrow(F) != nrow(g)")
    # if F.shape[1] != v.shape[0]:
    #    raise ValueError("Dimension mismatch: ncol(F) != nrow(v)")

    # Only works on dense vectors.
    if sparse.issparse(g):
        g = g.toarray()[:, 0]
    if sparse.issparse(v):
        v = v.toarray()[:, 0]

    n = v.shape[0]
    if method == "lsqr":
        F = sparse.csr_matrix(F)
        F_stack = sparse.vstack([F, 1/np.sqrt(2*t)*sparse.eye(n)])
        g_stack = np.concatenate([g, 1/np.sqrt(2*t)*v])
        return sparse.linalg.lsqr(F_stack, g_stack, atol=1e-16, btol=1e-16)[0]
    elif method == "lstsq":
        if sparse.issparse(F):
            F = F.todense()
        F_stack = np.vstack([F, 1/np.sqrt(2*t)*np.eye(n)])
        g_stack = np.concatenate([g, 1/np.sqrt(2*t)*v])
        return np.linalg.lstsq(F_stack, g_stack, rcond=None)[0]
    else:
        raise ValueError("method must be 'lsqr' or 'lstsq'")

def prox_qp_base(Q, q, F, g):
    # check warmstart/parameter mode -- make sure the problem reduction is only done once
    n = Q.shape[0]
    I = np.eye(n)
    v_par = Parameter(n)
    t_par = Parameter(nonneg=True)
    x = Variable(n)
    obj = quad_form(x, Q) + sum_squares(x)/2/t_par + (q-v_par/t_par)*x
    constr = [F * x <= g]
    prob = Problem(Minimize(obj), constr)
    def prox_qp1(v, t):
        v_par.value, t_par.value = v, t
        prob.solve()
        return x.value
    return prox_qp1
