import numpy as np
from scipy.optimize import bisect

TOLERANCE = 1e-6

def proj_l1(x, r = 1, method = "bisection"):
    if np.isscalar(x):
        x = np.array([x])
    if not np.isscalar(r) or r < 0:
        raise ValueError("r must be a non-negative scalar.")
    if np.linalg.norm(x,1) <= r:
        return x
    else:
        beta = proj_simplex(np.abs(x), r, method)
        return np.sign(x) * beta

def proj_simplex(x, r = 1, method = "bisection"):
    """Project x onto a simplex with upper bound r.
       Bisection: Liu and Ye (2009). "Efficient Euclidean Projections in Linear Time." Sect. 2.1.
            https://icml.cc/Conferences/2009/papers/123.pdf
       Efficient: Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions."
            Fig. 1 and Sect. 4.
            https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    if np.isscalar(x):
        x = np.array([x])
    if not np.isscalar(r) or r < 0:
        raise ValueError("r must be a non-negative scalar.")
    elif r == 0:
        return np.zeros(x.shape)

    if method == "bisection":
        c_min = np.min(x) - (r + TOLERANCE)/x.size
        c_max = np.max(x) + (r + TOLERANCE)/x.size
        c_star = bisect(lambda c: np.sum(np.maximum(x - c, 0)) - r, c_min, c_max)
    elif method == "sorted":
        x_decr = np.sort(x, axis = None)[::-1]
        x_cumsum = np.cumsum(x_decr)
        denom = np.arange(1, x_decr.size + 1)
        theta = (x_cumsum - r)/denom
        x_diff = x_decr - theta
        idx = np.max(np.argwhere(x_diff > 0).ravel())
        c_star = theta[idx]
    else:
        raise ValueError("method must be either 'bisection' or 'sorted'.")
    return np.maximum(x - c_star, 0)
