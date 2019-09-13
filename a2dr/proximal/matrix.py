import numpy as np
from scipy import sparse
from a2dr.proximal.composition import prox_scale
from a2dr.proximal.norm import prox_norm_inf_base

def prox_neg_log_det(B, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(aB-b) + cB + d\\|B\\|_F^2`, where :math:`f(B) = -\\log\\det(B)`, where `B` is a
    symmetric matrix. The scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if B.shape[0] != B.shape[1]:
        raise ValueError("B must be a square matrix.")
    B_symm = (B + B.T) / 2.0
    if not (np.allclose(B, B_symm)):
        raise ValueError("B must be a symmetric matrix.")
    return prox_scale(prox_neg_log_det_base, *args, **kwargs)(B_symm, t)

def prox_sigma_max(B, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(aB-b) + cB + d\\|B\\|_F^2`, where :math:`f(B) = \\sigma_{\\max}(B)`
    is the maximum singular value of :math:`B`, for scalar t > 0, and the optional arguments are a = scale,
    b = offset, c = lin_term, and d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default,
    t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_sigma_max_base, *args, **kwargs)(B, t)

def prox_trace(B, t = 1, C = None, *args, **kwargs):
    """Proximal operator of :math:`tf(aB-b) + cB + d\\|B\\|_F^2`, where :math:`f(B) = tr(C^TB)` is the trace of
    :math:`C^TB`, where C is a given matrix quantity. By default, C is the identity matrix, so :math:`f(B) = tr(B)`.
    The scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    if np.isscalar(B):
        B = np.array([[B]])
    if C is None:
        C = sparse.eye(B.shape[0])
    if B.shape[0] != C.shape[0]:
        raise ValueError("Dimension mismatch: nrow(B) != nrow(C)")
    if B.shape[1] != C.shape[1]:
        raise ValueError("Dimension mismatch: ncol(B) != ncol(C)")
    return prox_scale(prox_trace_base, C=C, *args, **kwargs)(B, t)

def prox_neg_log_det_base(B, t):
    """Proximal operator of :math:`f(B) = -\\log\\det(B)`.
    """
    s, u = np.linalg.eigh(B)
    # s_new = (s + np.sqrt(s**2 + 4*t))/2
    id_pos = (s >= 0)
    id_neg = (s < 0)
    s_new = np.zeros(len(s))
    s_new[id_pos] = (s[id_pos] + np.sqrt(s[id_pos] ** 2 + 4.0 * t)) / 2
    s_new[id_neg] = 2.0 * t / (np.sqrt(s[id_neg] ** 2 + 4.0 * t) - s[id_neg])
    return u.dot(np.diag(s_new)).dot(u.T)

def prox_sigma_max_base(B, t):
    """Proximal operator of :math:`f(B) = \\sigma_{\\max}(B)`, the maximum singular value of :math:`B`, otherwise
    known as the spectral norm.
    """
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    s_new = prox_norm_inf_base(s, t)
    return U.dot(np.diag(s_new)).dot(Vt)

def prox_trace_base(B, t, C):
    """Proximal operator of :math:`f(B) = tr(C^TB)`, the trace of :math:`C^TB`, where C is a given matrix quantity
    such that :math:`C^TB` is square.
    """
    return B - t*C
