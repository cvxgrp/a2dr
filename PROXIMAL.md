# Proximal Operators

|  Operator     |  Arguments  | Function  | Domain     |
| ------------- | ----------- | --------- | ---------- |
| prox_norm1 | (v, t) | $\sum_i \vert x_i \vert$ | $v \in \mathbf{R}^n$ |
| prox_norm2 | (v, t) |  $\sqrt{\sum_i x_i^2}$ | $v \in \mathbf{R}^n$ |
| prox_norm_inf | (v, t) | $\max_i \vert x_i \vert$   | $v \in \mathbf{R}^n$ |
| prox_norm_nuc  | (B, t) | $\text{tr}((X^TX)^{1/2})$  | $B \in \mathbf{R}^{m \times n}$  |
| prox_norm_fro  | (B, t) | $\sqrt{\sum_{ij} X_{ij}^2}$  | $B \in \mathbf{R}^{m \times n}$  |
| prox_group_lasso | (B, t) | $\sum_j \sqrt{\sum_i X_{ij}^2}$  | $B \in \mathbf{R}^{m \times n}$  |
| prox_neg_log_det | (B, t) | $-\log\det(X)$ | $B \in \mathbf{S}_+^n$ |
| prox_sigma_max | (B, t) | maximum singular <br> value of $X$  | $B \in \mathbf{R}^{m \times n}$  |
| prox_trace | (B, t, C) | $\text{tr}(C^TX)$  | $B \in \mathbf{R}^{m \times n}$  |
| prox_sum_squares | (v, t) | $\sum_i x_i^2$ | $v \in \mathbf{R}^n$ |


| Operator           | Arguments          | Constraint Set                            | Domain                |
| ------------------ | ------------------ | ----------------------------------------- | --------------------- |
| prox_box_constr    | (v, t, v_lo, v_hi) | $v^{lo} \leq x \leq v^{hi}$               | $v \in \mathbf{R}^n$  |
| prox_nonneg_constr | (v, t)             | $x \geq 0$                                | $v \in \mathbf{R}^n$  |
| prox_nonpos_constr | (v, t)             | $x \leq 0$                                | $v \in \mathbf{R}^n$  |
| prox_soc           | (v, t)             | $\sqrt{\sum_{i=1}^{n-1} x_i^2} \leq v_n$  | $v \in \mathbf{R}^n$  |
| prox_psd           | (B, t)             | $X \succeq 0$                             | $B \in \mathbf{S}^n$  |
