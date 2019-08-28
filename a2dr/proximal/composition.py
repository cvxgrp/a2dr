import numpy as np

def prox_scale(prox, *args, **kwargs):
    scale = kwargs.pop("scale", 1.0)
    offset = kwargs.pop("offset", 0)
    lin_term = kwargs.pop("lin_term", 0)
    quad_term = kwargs.pop("quad_term", 0)

    if not np.isscalar(scale) or scale == 0:
        raise ValueError("scale must be a non-zero scalar.")
    if not np.isscalar(quad_term) or quad_term < 0:
        raise ValueError("quad_term must be a non-negative scalar.")

    def prox_new(v, t):
        v_new = scale*(v - lin_term)/(2*quad_term + 1) - offset
        t_new = t*scale**2/(2*quad_term + 1)
        return (prox(v_new, t_new, *args, **kwargs) + offset)/scale
    return prox_new