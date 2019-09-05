import numpy as np
from scipy import sparse
from scipy.special import lambertw
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

def prox_entr(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = -x\\log(x)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
    """
    return prox_scale(prox_entr_base, *args, **kwargs)(v, t)

def prox_exp(v, t = 1, *args, **kwargs):
    """Proximal operator of :math:`tf(ax-b) + c^Tx + d\\|x\\|_2^2`, where :math:`f(x) = \\exp(x)` applied
    elementwise for scalar t > 0, and the optional arguments are a = scale, b = offset, c = lin_term, and
    d = quad_term. We must have t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0, and d = 0.
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
    and d = quad_term. We must have M > 0, t > 0, a = non-zero, and d >= 0. By default, t = 1, a = 1, b = 0, c = 0,
    and d = 0.
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
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
		min_elemwise = lambda x, y: x.minimum(y)
	else:
		max_elemwise = np.maximum
		min_elemwise = np.minimum
	return max_elemwise(v - t, 0) + min_elemwise(v + t, 0)

def prox_constant_base(v, t):
	"""Proximal operator of :math:`f(x) = c` for any constant :math:`c`.
	"""
	return v

def prox_entr_base(v, t):
	"""Proximal operator of :math:`f(x) = -x\\log(x)`.
	"""
	return lambertw(t*v - 1) * np.log(t) / t

def prox_exp_base(v, t):
	"""Proximal operator of :math:`f(x) = \\exp(x)`.
	"""
	return v - lambertw(np.exp(v - np.log(t)))

def prox_huber_base(v, t, M = 1):
	"""Proximal operator of
	.. math::
        f(x) =
            \\begin{cases}
                2M|x|-M^2 & \\text{for } |x| \\geq |M| \\\\
                      |x|^2 & \\text{for } |x| \\leq |M|
            \\end{cases}
    applied elementwise, where :math:`M` is a positive scalar that defaults to 1.
	"""
	if sparse.issparse(v):
		max_elemwise = lambda x, y: x.maximum(y)
		mul_elemwise = lambda x, y: x.multiply(y)
	else:
		max_elemwise = np.maximum
		mul_elemwise = np.multiply
	# return v / (1 + 1/t) if np.abs(v) < (1 + 1/t) else v - np.sign(v)/t
	return mul_elemwise(1 - (2*M*t) / max_elemwise(abs(v), M + 2*M*t), v)

def prox_identity_base(v, t):
	"""Proximal operator of :math:`f(x) = x`.
	"""
	return v - t

def prox_neg_base(v, t):
	"""Proximal operator of :math:`f(x) = -\\min(x,0)`, where the minimum is taken elementwise.
	"""
	return np.where(v + t <= 0, v + t, np.where(v >= 0, v, 0))

def prox_neg_log_base(v, t):
	"""Proximal operator of :math:`f(x) = -\\log(x)`.
	"""
	return (v + np.sqrt(v**2 + 4*t)) / 2

def prox_pos_base(v, t):
	"""Proximal operator of :math:`f(x) = \\max(x,0)`, where the maximum is taken elementwise.
	"""
	return np.where(v >= t, v - t, np.where(v <= 0, v, 0))
