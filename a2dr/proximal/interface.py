import numpy as np
import scipy as sp
from scipy import sparse

def shape_to_2d(shape):
    if np.isscalar(shape) or len(shape) == 0:
        return 1,1
    elif len(shape) == 1:
        return shape[0],1
    else:
        return shape

def apply_to_nonzeros(fun, v):
    if sparse.issparse(v):
        v_new = v.copy()
        v_new.data = fun(v.data)
        return v_new
    else:
        return fun(v)

NUMPY_FUNS = {"hstack": np.hstack,
              "max_elemwise": np.maximum,
              "min_elemwise": np.minimum,
              "mul_elemwise": np.multiply,
              "norm": np.linalg.norm,
              "vstack": np.vstack,
              "zeros": lambda shape, dtype=float: np.zeros(shape, dtype=dtype)
              }

SPARSE_FUNS = {"hstack": sparse.hstack,
               "max_elemwise": lambda x, y: x.maximum(y),
               "min_elemwise": lambda x, y: x.minimum(y),
               "mul_elemwise": lambda x, y: x.multiply(y),
               "norm": sp.sparse.linalg.norm,
               "vstack": sparse.vstack,
               "zeros": lambda shape, dtype=None: sparse.csr_matrix(shape_to_2d(shape), dtype=dtype)
               }
