# a2dr

`a2dr` is a Python package for solving large-scale non-smooth convex optimization problems with general linear constraints, with separable objective functions accessible through their proximal operators. It exploits the separability of the objective functions and the sparsity in the linear constraints, and utilizes the power of Anderson acceleration to achieve fast and robust convergence and scalability to multiple processors. 

It is an implementation of type-II Anderson accelerated Douglas-Rachford splitting, based on our paper [A. Fu, J. Zhang, and S. Boyd (2019)](http://www.stanford.edu/~boyd/papers/a2dr.html).

### Installation
To install `a2dr`, please follow the steps below:

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
* [Matplotlib >= 3.0.3](https://github.com/matplotlib/matplotlib)
* [CVXPY >= 1.0](https://github.com/cvxgrp/cvxpy)
* [NumPy >= 1.14](https://github.com/numpy/numpy)
* [SciPy >= 1.2.1](https://github.com/scipy/scipy)
* [scikit-sparse >= 0.1](https://github.com/scikit-sparse/scikit-sparse)
* [nose](https://nose.readthedocs.io/en/latest/)
* [setuptools](https://github.com/pypa/setuptools)
* Python 3.x

Please file an issue on Github if you want Python 2 support.


