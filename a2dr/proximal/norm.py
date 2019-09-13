import numpy as np
from scipy import sparse
from a2dr.proximal.interface import NUMPY_FUNS, SPARSE_FUNS, apply_to_nonzeros
from a2dr.proximal.projection import proj_l1
from a2dr.proximal.composition import prox_scale

def prox_norm1(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\|x\\|_1`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(v):
        v = np.array([v])
    return prox_scale(prox_norm1_base, *args, **kwargs)(v, t)

def prox_norm2(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\|x\\|_2`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(v):
        v = np.array([v])
    return prox_scale(prox_norm2_base, *args, **kwargs)(v, t)

def prox_norm_inf(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\|x\\|_{\\infty}`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(v):
        v = np.array([v])
    return prox_scale(prox_norm_inf_base, *args, **kwargs)(v, t)

def prox_norm_nuc(B, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(aB-b) + cB + d\\|B\\|_F^2`, where :math:`f(B) = \\|B\\|_*`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(B):
        B = np.array([[B]])
    return prox_scale(prox_norm_nuc_base, *args, **kwargs)(B, t)

def prox_norm_fro(B, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(aB-b) + cB + d\\|B\\|_F^2`, where :math:`f(B) = \\|B\\|_2`
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(B):
        B = np.array([[B]])
    return prox_scale(prox_norm_fro_base, *args, **kwargs)(B, t)

def prox_group_lasso(B, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(aB-b) + cB + d\\|B\\|_F^2`, where :math:`f(B) = \\|B\\|_{2,1}` is the
    group lasso of :math:`B`, for scalar t > 0, and the optional arguments are a = scale, b = offset,
    c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1,
    b = 0, c = 0, and d = 0.
    """
    if np.isscalar(B):
        B = np.array([[B]])
    return prox_scale(prox_group_lasso_base, *args, **kwargs)(B, t)

def prox_norm1_base(v, t):
    """Proximal operator of :math:`f(x) = \\|x\\|_1`.
    """
    return apply_to_nonzeros(lambda y: np.maximum(y - t, 0) - np.maximum(-y - t, 0), v)

def prox_norm2_base(v, t):
    """Proximal operator of :math:`f(x) = \\|x\\|_2`.
    """
    FUNS = SPARSE_FUNS if sparse.issparse(v) else NUMPY_FUNS
    norm, zeros = FUNS["norm"], FUNS["zeros"]

    if np.isscalar(v):
        v_norm = abs(v)
    elif len(v.shape) == 1:
        v_norm = norm(v,2)
    elif len(v.shape) == 2:
        v_norm = norm(v,'fro')

    if v_norm == 0:
        return zeros(v.shape)
    else:
        return np.maximum(1 - t/v_norm, 0) * v

def prox_norm_inf_base(v, t):
    """Proximal operator of :math:`f(x) = \\|x\\|_{\\infty}`.
    """
    return v - t * proj_l1(v/t)

def prox_norm_fro_base(B, t):
    """Proximal operator of :math:`f(B) = \\|B\\|_2`, the Frobenius norm of :math:`B`.
    """
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    s_new = prox_norm2_base(s, t)
    # s_norm = np.linalg.norm(s, 2)
    # s_new = np.zeros(s.shape) if s_norm == 0 else np.maximum(1 - t/s_norm, 0) * s
    return U.dot(np.diag(s_new)).dot(Vt)

def prox_norm_nuc_base(B, t):
    """Proximal operator of :math:`f(B) = \\|B\\|_*`, the nuclear norm of :math:`B`.
    """
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    s_new = prox_norm1_base(s, t)
    # s_new = np.maximum(s - t, 0)
    return U.dot(np.diag(s_new)).dot(Vt)

def prox_group_lasso_base(B, t):
    """Proximal operator of :math:`f(B) = \\|B\\|_{2,1} = \\sum_j \\|B_j\\|_2`, the group lasso of :math:`B`,
    where :math:`B_j` is the j-th column of :math:`B`.
    """
    # FUNS = SPARSE_FUNS if sparse.issparse(B) else NUMPY_FUNS
    # vstack, hstack = FUNS["vstack"], FUNS["hstack"]
    # prox_cols = [prox_norm2(B[:, j], t) for j in range(B.shape[1])]
    # return vstack(prox_cols).T if prox_cols[0].ndim == 1 else hstack(prox_cols)
    if sparse.issparse(B):
        B = B.tocsc()
        prox_cols = [prox_norm2(B[:, j], t) for j in range(B.shape[1])]
        return sparse.hstack(prox_cols)
    else:
        prox_cols = [prox_norm2(B[:, j], t) for j in range(B.shape[1])]
        return np.column_stack(prox_cols)
