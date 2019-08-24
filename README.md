# a2dr

`a2dr` is a Python package for solving large-scale non-smooth convex optimization problems with general linear constraints, with separable objective functions accessible through their proximal operators. It exploits the separability of the objective functions and the sparsity in the linear constraints, and utilizes the power of Anderson acceleration to achieve fast and robust convergence and scalability to multiple processors. 

It is an implementation of type-II Anderson accelerated Douglas-Rachford splitting, based on our paper [A. Fu, J. Zhang, and S. Boyd (2019)](http://www.stanford.edu/~boyd/papers/a2dr.html).

### Installation
To install `a2dr`, first make sure that you have [setuptools](https://github.com/pypa/setuptools)
and [nose](https://nose.readthedocs.io/en/latest/) installed. Then follow the steps below:

1. Clone the [`a2dr` git repository](https://github.com/cvxgrp/a2dr).
2. Navigate to the top-level of the cloned directory and run:

```python
python setup.py install
``` 

3. Test the installation with nose:

```python
nosetests a2dr
```

The requirements are:
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [CVXPY](https://github.com/cvxgrp/cvxpy)
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* Python 3.x

Please file an issue on Github if you want Python 2 support.

### Problem
`a2dr` solves problems of the following form:
```
minimize         f_1(x_1) + ... + f_N(x_N)
subject to       A_1x_1 + ... + A_Nx_N = b.
```
where f_i (i=1,...,N) are only accessible through their proximal operators.

### Usage
After installing `a2dr`, you can import `a2dr` using 
```python
import a2dr
```
This module exposes a function **a2dr** (the solver), which can be used via `a2dr.a2dr`, or directly imported using
```python
from a2dr import a2dr
```
The function **a2dr** is called with the command 
```python
x_vals, primal, dual, num_iters, solve_time = a2dr(p_list, 
                                                   A_list=[], 
                                                   b=np.array([]),
                                                   v_init=None, 
                                                   n_list=None
                                                   max_iter=1000,
                                                   t_init=1/10,
                                                   eps_abs=1e-6,
                                                   eps_rel=1e-8,
                                                   precond=True,
                                                   ada_reg=True,
                                                   anderson=True,
                                                   m_accel=10,
                                                   lam_accel=1e-8,
                                                   aa_method='lstsq',
                                                   D_safe=1e6,
                                                   eps_safe=1e-6,
                                                   M_safe=10)
```

#### Paremeters:
The arguments `p_list`, `A_list` and `b` correspond to the problem data.
* `p_list` is the list of proximal operators of f_i. Each element of `p_list` is a Python function,
which takes as input a vector v and parameter t > 0 and outputs the proximal operator of f_i evaluated at (v,t).
* `A_list` is the list of A_i. The lists `p_list` and `A_list` must be given in the same order i = 1,...,N.
* `b` is the vector b. 
Notice that `A_list` and `b` are optional, and when omitted, the solver recognizes the problem as one without linear constraints. For information on the other optional hyper-parameters, please refer to our [companion paper](http://stanford.edu/~boyd/papers/a2dr.html) (Algorithm 2) and the [source code comments of the function **a2dr** in solver.py](https://github.com/cvxgrp/a2dr/tree/master/a2dr).

#### Returns:
* The output `x_vals` is a list of x_1,...,x_N from the iteration with the smallest residuals.
* `primal` and `dual` are arrays containing the primal and dual residual norms for the entire iteration process, respectively. 
* The value `num_iters` is the total number of iterations, and `solve_time` is the algorithm runtime.

#### Other tools
The moduel `a2dr` also comes with several additional tools that facilitates the transformation of the problems into the required input format described above as well as tests and visualization. In particular, it come with a package for proximal operators, which can be imported via
```python
import a2dr.proximal
```
It also comes with some tests and visualization tools, which can be imported via
```python
import a2dr.tests
```

#### Example
We showcase the usage of the solver function **a2dr** as well as the the tool packages `a2dr.proximal` and `a2dr.tests` with the following example. More examples can be found in the [examples/](examples/) directory. 
```python
# Non-negative least squares (see our companion paper for more details)
import numpy as np
import scipy as sp
import numpy.linalg
import matplotlib.pyplot as plt
from scipy import sparse
from a2dr import a2dr
from a2dr.proximal.prox_operators import *
from a2dr.tests.base_test import BaseTest

# Problem data.
np.random.seed(1)
m, n = 150, 300 
density = 0.001
X = sparse.random(m, n, density=density, data_rvs=np.random.randn)
y = np.random.randn(m)

# Convert problem to standard form.
prox_list = [prox_sum_squares(X, y), lambda v, t: np.maximum(v,0)]
A_list = [sparse.eye(n), -sparse.eye(n)]
b = np.zeros(n)

# Solve with DRS.
drs_result = a2dr(prox_list, A_list, b, anderson=False)
# Solve with A2DR.
a2dr_result = a2dr(prox_list, A_list, b, anderson=True)
bt = BaseTest()
bt.compare_total(drs_result, a2dr_result)
```

### Citing
If you wish to cite `a2dr`, please use the following:
```
@article{a2dr,
    author       = {Fu, A. and Zhang, J. and Boyd, S.},
    title        = {Anderson Accelerated Douglas-Rachford Splitting},
    journal      = {http://stanford.edu/~boyd/papers/a2dr.html},
    year         = {2019},
}

@misc{a2dr_code,
    author       = {Fu, A. and Zhang, J. and Boyd, S.},
    title        = {{a2dr}: Anderson accelerated Douglas-Rachford splitting, version 0.1},
    howpublished = {\url{https://github.com/cvxgrp/a2dr}},
    year         = 2019
}
```
