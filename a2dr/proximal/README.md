# Proximal Operators
`a2dr` provides a library of proximal operators for common sets and functions in `a2dr.proximal`. Recall that the proximal operator of a function $f:\mathbf{R}^n \rightarrow \mathbf{R}$ is
$$\mathbf{prox}_{tf}(v) = \arg\min_x \left\{f(x) + \frac{1}{2t}\|x - v\|_2^2\right\},$$
where $v \in \mathbf{R}^n$ is the input and $t > 0$ is a parameter. A similar definition exists for a function over matrices, except the $\ell_2$ norm is replaced with the Frobenius norm.

**Notation.** We use $\mathbf{R}$ to denote the real numbers and $\mathbf{Z}$ the integers. We let $\mathbf{R}_+$ and $\mathbf{R}_-$ be the non-negative and non-positive reals, respectively, and similarly for $\mathbf{Z}_+$ and $\mathbf{Z}_-$. The domain $\mathbf{S}^n$ refers to the set of symmetric $n \times n$ matrices, and the domains $\mathbf{S}_+^n$ and $\mathbf{S}_-^n$ refer to the set of positive semidefinite and negative semidefinite matrices, respectively. For a set $C$, the indicator function $I_C(x) = 0$ if $x \in C$ and $\infty$ otherwise.

### Example
To call a proximal operator, we simply import it and pass in the values of $v$ and $t$. For example,
```
import numpy as np
from a2dr.proximal import prox_norm2

v = np.arange(10)
prox_norm2(v, t=2)
```
computes $\mathbf{prox}_{t\|\cdot\|_2}(v)$ for $t = 2$ and $v = (0,1,\ldots,9)$. Most of the proximal operators in `a2dr.proximal` have a closed-form solution. Where possible, we have used this solution in our implementation - see [N. Parikh and S. Boyd (2013)](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf) for mathematical details. Otherwise, we typically provide options for how to numerically evaluate the operator, e.g., `prox_sum_squares_affine` can be called with argument `method = "lsqr"` for [scipy.sparse.linalg.lsqr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html) or `method = "lstsq"` for [numpy.linalg.lstsq](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html) as the least-squares solution method.

Let us solve the problem
$$\begin{array}{ll} \text{minimize} & \|x\|_2 \\ \text{subject to} & Ax = b \end{array}$$
with variable $x \in \mathbf{R}^n$ and problem data $A \in \mathbf{R}^{m \times n}$ and $b \in \mathbf{R}^m$. The code for this is
```
from a2dr import a2dr
a2dr_result = a2dr([prox_norm2], [A], b, anderson=True)
```
Here we pass a list containing the handle `prox_norm2` directly into `a2dr`. Our proximal library provides many such predefined operator handles, so users can quickly and easily solve a wide array of problems.

### Additional arguments
Certain proximal operators require additional input arguments. For instance, `prox_quad_form` is the proximal operator of $f(x) = x^TQx$ for a matrix $Q \in \mathbf{R}^n$. To use this operator, we need to create a wrapper function that takes only $(v, t)$ as input, setting $Q$ directly in its body.
```
Q = np.random.randn(n,n)   # User-defined matrix
prox_quad_form_wrapper = lambda v,t: prox_quad_form(v, t, Q=Q)
a2dr_result = a2dr([prox_quad_form_wrapper], ...)
```

### Compositions
Given a function $f(x)$, define
$$g(x) := tf(ax-b) + c^Tx + d\|x\|_2^2,$$
where $t > 0$, $a \neq 0$, $b \in \mathbf{R}$, $c \in \mathbf{R}^n$, and $d \geq 0$ are parameters. It can be shown that
$$\mathbf{prox}_g(v) = \mathbf{prox}_{(ta^2/(2d+1))f}(a(v - c)/(2d + 1) - b)$$
using Theorems 6.11 to 6.15 in [A. Beck (2017), Ch. 6](https://archive.siam.org/books/mo25/mo25_ch6.pdf). Thus, we can access the proximal operator of the transformation $g(x)$ via the proximal operator of $f(x)$. The `a2dr.proximal` library provides built-in support for such compositions. For instance,
```
prox_norm2(v, t=2, scale=5, offset=10, lin_term=np.ones(n), quad_term=1)
```
evaluates $\mathbf{prox}_g(v)$ with $f(x) = \|x\|_2$ and $t = 2, a = 5, b = 10, c = (1,1,\ldots,1)$, and $d = 1$.

# Reference Tables
In the following tables, we describe our library of proximal operators: their Python handle, main input arguments, domain, and corresponding function $f(x)$.

## Scalar functions
A scalar function takes a scalar, vector, or matrix as input and returns a scalar.
### Vector input
|    Python Handle        |  Arguments                      |   Domain             |   Function                       |
| ----------------------- | ------------------------------- | -------------------- | -------------------------------- |
| prox_logistic           | (v, t = 1, y = $-\mathbf{1}^n$) | $v \in \mathbf{R}^n$ | $\sum_i \log(1 + \exp(-y_ix_i))$ |
| prox_max                | (v, t = 1)                      | $v \in \mathbf{R}^n$ | $\max_i x_i$                     |
| prox_norm1              | (v, t = 1)                      | $v \in \mathbf{R}^n$ | $\sum_i \vert x_i \vert$         |
| prox_norm2              | (v, t = 1)                      | $v \in \mathbf{R}^n$ | $\sqrt{\sum_i x_i^2}$            |
| prox_norm_inf           | (v, t = 1)                      | $v \in \mathbf{R}^n$ | $\max_i \vert x_i \vert$         |
| prox_quad_form          | (v, t = 1, Q)                   | $v \in \mathbf{R}^n$ <br> $Q \in \mathbf{R}^{n \times n}$ | $x^TQx$ |
| prox_sum_squares        | (v, t = 1)                      | $v \in \mathbf{R}^n$ | $\sum_i x_i^2$                   |
| prox_sum_squares_affine | (v, t = 1, F, g)                | $v \in \mathbf{R}^n$ <br> $F \in \mathbf{R}^{m \times n}$ <br> $g \in \mathbf{R}^m$ | $\|Fx - g\|_2^2$ |

### Matrix input
|  Python Handle   |  Arguments          |   Domain                        |   Function                         |
| ---------------- | ------------------- | ------------------------------- | ---------------------------------- |
| prox_group_lasso | (B, t = 1)          | $B \in \mathbf{R}^{m \times n}$ | $\sum_j \sqrt{\sum_i X_{ij}^2}$    |
| prox_neg_log_det | (B, t = 1)          | $B \in \mathbf{S}_+^n$          | $-\log\det(X)$                     |
| prox_norm_fro    | (B, t = 1)          | $B \in \mathbf{R}^{m \times n}$ | $\sqrt{\sum_{ij} X_{ij}^2}$        |
| prox_norm_nuc    | (B, t = 1)          | $B \in \mathbf{R}^{m \times n}$ | $\text{tr}((X^TX)^{1/2})$          |
| prox_sigma_max   | (B, t = 1)          | $B \in \mathbf{R}^{m \times n}$ | maximum singular <br> value of $X$ |
| prox_trace       | (B, t = 1, C = $I$) | $B \in \mathbf{R}^{m \times n}$ <br> $C \in \mathbf{R}^{m \times n}$ | $\text{tr}(C^TX)$ |

## Elementwise functions
These functions apply to each element of the input, which can be a scalar, vector, or matrix. In `a2dr`, the corresponding proximal operators also apply elementwise, i.e., given $v \in \mathbf{R}^n$, they return $(\mathbf{prox}_{tf}(v_1), \ldots, (\mathbf{prox}_{tf}(v_n))$.
|  Python Handle  |   Arguments       |   Domain            |   Function                       |
| --------------- | ----------------- | ------------------- | -------------------------------- |
| prox_abs        | (v, t = 1)        | $v \in \mathbf{R}$  | $\vert x\vert$                   |
| prox_constant   | (v, t = 1)        | $v \in \mathbf{R}$  | any constant $c \in \mathbf{R}$  |
| prox_exp        | (v, t = 1)        | $v \in \mathbf{R}$  | $e^x$                            |
| prox_huber      | (v, t = 1, M = 1) | $v \in \mathbf{R}$ <br> $M \geq 0$  | $\begin{cases} x^2 & \vert x\vert \leq M \\ 2M\vert x\vert - M^2 & \vert x\vert > M \end{cases}$ |
| prox_identity   | (v, t = 1)        | $v \in \mathbf{R}$ | $x$                               |
| prox_neg        | (v, t = 1)        | $v \in \mathbf{R}$ | $-\min(x,0)$                      |
| prox_neg_entr   | (v, t = 1)        | $v > 0$            | $x\log x$                         |
| prox_neg_log    | (v, t = 1)        | $v > 0$            | $-\log x$                         |
| prox_pos        | (v, t = 1)        | $v \in \mathbf{R}$ | $\max(x,0)$                       |

## Set indicators
Below we describe the proximal operators for several constraint sets $C$. The proximal operator of $f = I_C$ is the minimum norm projection $\mathbf{prox}_{tI_C}(v) = \arg\min_{x \in C}\{\|x - v\|_2^2\}$ when $C \subseteq \mathbf{R}^n$.
### Convex sets
|  Python Handle     |  Arguments    |  Domain                |  Constraint Set                           |
| ------------------ | ------------- | ---------------------- | ----------------------------------------- |
| prox_box_constr    | (v, v_lo = $-\infty$, v_hi = $\infty$) | $v \in \mathbf{R}^n$ <br> $v^{lo}, v^{hi} \in \mathbf{R}^n$  | $v^{lo} \leq x \leq v^{hi}$ |
| prox_nonneg_constr | v             | $v \in \mathbf{R}^n$   | $x \geq 0$                                |
| prox_nonpos_constr | v             | $v \in \mathbf{R}^n$   | $x \leq 0$                                |
| prox_psd_cone      | B             | $B \in \mathbf{S}^n$   | $X \succeq 0$                             |
| prox_soc           | v             | $v \in \mathbf{R}^n$   | $\sqrt{\sum_{i=1}^{n-1} x_i^2} \leq x_n$  |

### Nonconvex sets
These sets are currently in the experimental phase.
|  Python Handle   |  Arguments  |  Domain              | Constraint Set       |
| ---------------- | ----------- | -------------------- | -------------------- |
| prox_cardinality | (v, k = 10) | $v \in \mathbf{R}^n$ <br> $k \geq 0$ | $\mathbf{card}(\{v_i\vert v_i \neq 0\}) \leq k$ |
| prox_rank        | (B, k = 10) | $B \in \mathbf{R}^{m \times n}$ <br> $k \geq 0$ | $\mathbf{rank}(B) \leq k$ |
| prox_boolean     | v           | $v \in \mathbf{R}^n$ | $x \in \{0,1\}^n$    |
| prox_integer     | v           | $v \in \mathbf{R}^n$ | $x \in \mathbf{Z}^n$ |
