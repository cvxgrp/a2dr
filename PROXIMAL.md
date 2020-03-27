# Proximal Operators
## Scalar functions
### Vector input
|    Operator   |  Arguments  |   Domain   |   Function   |
| ------------- | ----------- | ---------- | ------------ |
| prox_logistic   | (v, t = 1, y = $-\mathbf{1}^n$) | $v \in \mathbf{R}^n$  | $\sum_i \log(1 + \exp(-y_ix_i))$  |
| prox_max   | (v, t = 1)  | $v \in \mathbf{R}^n$  | $\max_i x_i$  |
| prox_norm1 | (v, t = 1) | $v \in \mathbf{R}^n$ | $\sum_i \vert x_i \vert$ |
| prox_norm2 | (v, t = 1) | $v \in \mathbf{R}^n$ | $\sqrt{\sum_i x_i^2}$ |
| prox_norm_inf | (v, t = 1) | $v \in \mathbf{R}^n$ | $\max_i \vert x_i \vert$   |
| prox_quad_form   | (v, t = 1, Q)  | $v \in \mathbf{R}^n$ <br> $Q \in \mathbf{R}^{n \times n}$ | $x^TQx$  |
| prox_sum_squares | (v, t = 1) | $v \in \mathbf{R}^n$ | $\sum_i x_i^2$ |
| prox_sum_squares_affine   | (v, t = 1, F, g)  | $v \in \mathbf{R}^n$ <br> $F \in \mathbf{R}^{m \times n}$ <br> $g \in \mathbf{R}^m$ | $\|Fx - g\|_2^2$  |

### Matrix input
|    Operator   |  Arguments  |   Domain   |   Function    |
| ------------- | ----------- | ---------- | ------------- |
| prox_group_lasso | (B, t = 1) | $B \in \mathbf{R}^{m \times n}$ | $\sum_j \sqrt{\sum_i X_{ij}^2}$ |
| prox_neg_log_det | (B, t = 1) | $B \in \mathbf{S}_+^n$ | $-\log\det(X)$ |
| prox_norm_fro  | (B, t = 1) | $B \in \mathbf{R}^{m \times n}$ | $\sqrt{\sum_{ij} X_{ij}^2}$ |
| prox_norm_nuc  | (B, t = 1) | $B \in \mathbf{R}^{m \times n}$ | $\text{tr}((X^TX)^{1/2})$ |
| prox_sigma_max | (B, t = 1) | $B \in \mathbf{R}^{m \times n}$ | maximum singular <br> value of $X$ |
| prox_trace | (B, t = 1, C = $I$) | $B \in \mathbf{R}^{m \times n}$ <br> $C \in \mathbf{R}^{m \times n}$ | $\text{tr}(C^TX)$ |

## Elementwise functions
|    Operator   |  Arguments  |   Domain   |   Function    |
| ------------- | ----------- | ---------- | ------------- |
| prox_abs   | (v, t = 1)  | $v \in \mathbf{R}$  | $\vert x\vert$  |
| prox_constant   | (v, t = 1)  | $v \in \mathbf{R}$  | any constant $c \in \mathbf{R}$  |
| prox_exp   | (v, t = 1)  | $v \in \mathbf{R}$  | $e^x$ |
| prox_huber   | (v, t = 1, M = 1)  | $v \in \mathbf{R}$ <br> $M \geq 0$  | $\begin{cases} x^2 & \vert x\vert \leq M \\ 2M\vert x\vert - M^2 & \vert x\vert > M \end{cases}$  |
| prox_identity   | (v, t = 1) | $v \in \mathbf{R}$ | $x$ |
| prox_neg   | (v, t = 1)  | $v \in \mathbf{R}$  | $-\min(x,0)$  |
| prox_neg_entr   | (v, t = 1)  | $v > 0$  | $x\log x$  |
| prox_neg_log   | (v, t = 1) | $v > 0$  | $-\log x$  |
| prox_pos  | (v, t = 1)  | $v \in \mathbf{R}$  | $\max(x,0)$  |

## Set projections
### Convex sets
| Operator           | Arguments          | Domain                | Constraint Set                            |
| ------------------ | ------------------ | --------------------- | ----------------------------------------- |
| prox_box_constr    | (v, v_lo = $-\infty$, v_hi = $\infty$) | $v \in \mathbf{R}^n$ <br> $v^{lo},v^{hi} \in \mathbf{R}^n$  | $v^{lo} \leq x \leq v^{hi}$               |
| prox_nonneg_constr | (v)             | $v \in \mathbf{R}^n$  | $x \geq 0$                                |
| prox_nonpos_constr | (v)             | $v \in \mathbf{R}^n$  | $x \leq 0$                                |
| prox_psd           | (B)             | $B \in \mathbf{S}^n$  | $X \succeq 0$                             |
| prox_soc           | (v)             | $v \in \mathbf{R}^n$  | $\sqrt{\sum_{i=1}^{n-1} x_i^2} \leq x_n$  |

### Nonconvex sets
| Operator   | Arguments   | Domain   | Constraint Set   |
| ---------- | ----------- | -------- | ---------------- |
| prox_cardinality   | (v, k = 10)  | $v \in \mathbf{R}^n$  | $\mathbf{card}(\{v_i\vert v_i \neq 0\}) \leq k$ |
| prox_rank   | (B, k = 10) | $B \in \mathbf{R}^{m \times n}$  | $\mathbf{rank}(B) \leq k$  |
| prox_boolean   | (v)  | $v \in \mathbf{R}^n$  | $x \in \{0,1\}^n$ |
| prox_integer   | (v)  | $v \in \mathbf{R}^n$  | $x \in \mathbf{Z}^n$  |
