import numpy as np
from scipy import sparse

def prox_scale(prox, *args, **kwargs):
    """Given the proximal operator of a function :math:`f`, returns the proximal operator of :math:`g` defined as
    .. math::
        g(x) = `tf(ax-b) + <c,x> + d\\|x\\|_F^2`,
    where :math:`t > 0`, :math:`a \\neq 0` is a scaling term, :math:`b` is an offset, :math:`c` is a linear multiplier,
    and :math:`d \\geq 0` is a quadratic multiplier.
    :param prox: Function handle of a proximal operator that takes as input a vector/matrix :math:`v` and a scalar
    :math:`t > 0`, and outputs
    .. math::
        `prox_{tf}(v) = \\min_x f(x) + \\frac{1}{2t}\\|x - v\\|_F^2}`,
    the proximal operator of :math:`tf` evaluated at :math:`v`, where :math:`f` is an arbitrary function.
    :param scale: Scaling term :math:`a \\neq 0`. Defaults to 1.
    :param offset: Offset term :math:`b`. Defaults to 0.
    :param lin_term: Linear term :math:`c`. If a scalar is given, then :math:`c` is its vectorization. Defaults to 0.
    :param quad_term: Quadratic term :math:`d \\geq 0`. Defaults to 0.
    :return: Function handle for the proximal operator of :math:`g`.
    """
    scale = kwargs.pop("scale", 1.0)
    offset = kwargs.pop("offset", 0)
    lin_term = kwargs.pop("lin_term", 0)
    quad_term = kwargs.pop("quad_term", 0)

    if not np.isscalar(scale) or scale == 0:
        raise ValueError("scale must be a non-zero scalar.")
    if not np.isscalar(quad_term) or quad_term < 0:
        raise ValueError("quad_term must be a non-negative scalar.")

    def prox_new(v, t):
        # if sparse.issparse(v):
        #     if not ((lin_term == 0 or sparse.issparse(lin_term)) and \
        #              (quad_term == 0 or sparse.issparse(quad_term))):
        #         v = v.todense()
        v_new = scale*(v - lin_term)/(2*quad_term + 1) - offset
        t_new = t*scale**2/(2*quad_term + 1)
        return (prox(v_new, t_new, *args, **kwargs) + offset)/scale
    return prox_new
