import numpy as np
from scipy.optimize import bisect

TOLERANCE = 1e-6

def proj_l1(x, r = 1, method = "bisection"):
    """Project x onto the l1-ball with radius r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 4.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    if method == "bisection":
        if np.linalg.norm(x,1) <= r:
            return x
        else:
            c_min = np.min(np.abs(x)) - r / x.size - TOLERANCE
            c_max = np.max(np.abs(x)) + TOLERANCE
            c_star = bisect(lambda c: np.sum(np.maximum(np.abs(x) - c, 0)) - r, c_min, c_max)
            return np.sign(x) * np.maximum(np.abs(x) - c_star, 0)
    elif method == "simplex":
        beta = proj_simplex(np.abs(x), r)
        return np.sign(x) * beta
    else:
        raise ValueError("method must be either 'bisection' or 'simplex'")

def proj_simplex(x, r = 1):
    """Project x onto a simplex with upper bound r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 3.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    x_decr = np.sort(x, axis = None)[::-1]
    x_cumsum = np.cumsum(x_decr)
    denom = 1 + np.arange(len(x_decr))
    theta = (x_cumsum - r)/denom
    x_diff = x_decr - theta
    # idx = np.squeeze(np.argwhere(x_diff > 0))[-1]
    idx = np.argwhere(x_diff > 0)[0][-1]
    return np.maximum(x - theta[idx], 0)
