import numpy as np
from scipy import sparse
from scipy.special import lambertw
from a2dr.proximal.interface import NUMPY_FUNS, SPARSE_FUNS, apply_to_nonzeros
from a2dr.proximal.composition import prox_scale

def prox_abs(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = |x|` applied elementwise
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_abs_base, *args, **kwargs)(v, t)

def prox_constant(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = c` for constant c,
    scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_constant_base, *args, **kwargs)(v, t)

def prox_exp(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\exp(x)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and 
    d = 0.
    """
    return prox_scale(prox_exp_base, *args, **kwargs)(v, t)

def prox_huber(v, t = 1, M = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where
    .. math::
        f(x) =
            \\begin{cases}
                2M|x|-M^2 & \\text{for } |x| \\geq |M| \\\\
                      |x|^2 & \\text{for } |x| \\leq |M|
            \\end{cases}
    applied elementwise for scalar M > 0, t > 0, and the optional arguments are a = scale, b = offset, c = lin_term,
    and d = quad_term. We must have M > 0, t > 0, a = non-zero, and d >= 0. By default, M = 1, t = 1, a = 1, b = 0, 
    c = 0, and d = 0.
    """
    return prox_scale(prox_huber_base, M, *args, **kwargs)(v, t)

def prox_identity(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = x` applied elementwise
    for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and d = quad_term.
    We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_identity_base, *args, **kwargs)(v, t)

def prox_neg(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = -\\min(x,0)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_neg_base, *args, **kwargs)(v, t)

def prox_neg_entr(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = x\\log(x)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_neg_entr_base, *args, **kwargs)(v, t)

def prox_neg_log(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = -\\log(x)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_neg_log_base, *args, **kwargs)(v, t)

def prox_pos(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\max(x,0)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_pos_base, *args, **kwargs)(v, t)

def prox_abs_base(v, t):
	"""Proximal operator of :math:`f(x) = |x|`.
	"""
	return apply_to_nonzeros(lambda y: np.maximum(y - t, 0) + np.minimum(y + t, 0), v)

def prox_constant_base(v, t):
	"""Proximal operator of :math:`f(x) = c` for any constant :math:`c`.
	"""
	return v

def prox_exp_base(v, t):
	"""Proximal operator of :math:`f(x) = \\exp(x)`.
	"""
	if sparse.issparse(v):
		v = v.todense()
	return v - lambertw(np.exp(v + np.log(t)))

def prox_huber_base(v, t, M):
	"""Proximal operator of
	.. math::
        f(x) =
            \\begin{cases}
                2M|x|-M^2 & \\text{for } |x| \\geq |M| \\\\
                      |x|^2 & \\text{for } |x| \\leq |M|
            \\end{cases}
    applied elementwise, where :math:`M` is a positive scalar.
	"""
	return apply_to_nonzeros(lambda y: np.where(np.abs(y) <= (M + 2*M*t), y / (1 + 2*t), y - 2*M*t*np.sign(y)), v)

def prox_identity_base(v, t):
	"""Proximal operator of :math:`f(x) = x`.
	"""
	if sparse.issparse(v):
		v = v.todense()
	return v - t

def prox_neg_base(v, t):
	"""Proximal operator of :math:`f(x) = -\\min(x,0)`, where the minimum is taken elementwise.
	"""
	return apply_to_nonzeros(lambda y: np.where(y + t <= 0, y + t, np.where(y >= 0, y, 0)), v)

def prox_neg_entr_base(v, t):
	"""Proximal operator of :math:`f(x) = x\\log(x)`.
	"""
	if sparse.issparse(v):
		v = v.todense()
	return t * lambertw(np.exp((v/t - 1) - np.log(t)))

def prox_neg_log_base(v, t):
	"""Proximal operator of :math:`f(x) = -\\log(x)`.
	"""
	if sparse.issparse(v):
		v = v.todense()
	return (v + np.sqrt(v**2 + 4*t)) / 2

def prox_pos_base(v, t):
	"""Proximal operator of :math:`f(x) = \\max(x,0)`, where the maximum is taken elementwise.
	"""
	return apply_to_nonzeros(lambda y: np.where(y - t >= 0, y - t, np.where(y <= 0, y, 0)), v)
