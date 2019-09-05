import numpy as np

def proj_l1(x, r = 1):
    """Project x onto the l1-ball with radius r.
       Duchi et al (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." Fig. 1 and Sect. 4.
       https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    beta = proj_simplex(np.abs(x), r)
    return np.sign(x) * beta

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

def proj_soc(v, s):
    """Project (v,s) onto the second-order cone :math:`C = {(x,t):\\|x\\|_2 \\leq t}`.
    Parikh and Boyd (2013). "Proximal Algorithms." Foundations and Trends in Optimization. vol. 1, no. 3, Sect. 6.3.2.
    """
    v_norm = np.linalg.norm(v, 2)
    if v_norm <= -s:
        return np.zeros(v_norm.shape), 0
    elif v_norm <= s:
        return v, s
    else:
        scale = (1 + s/v_norm)/2
        return scale*v, scale*s
