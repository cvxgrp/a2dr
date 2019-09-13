"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

from a2dr.proximal.constraint import prox_box_constr, prox_nonneg_constr, prox_nonpos_constr, prox_psd_cone, prox_soc
from a2dr.proximal.elementwise import prox_abs, prox_constant, prox_neg_entr, prox_exp, prox_huber, prox_identity, \
    prox_neg, prox_pos, prox_neg_log
from a2dr.proximal.matrix import prox_neg_log_det, prox_sigma_max, prox_trace
from a2dr.proximal.misc import prox_logistic, prox_max
from a2dr.proximal.norm import prox_norm1, prox_norm2, prox_norm_inf, prox_norm_fro, prox_norm_nuc, prox_group_lasso
from a2dr.proximal.quadratic import prox_quad_form, prox_sum_squares, prox_sum_squares_affine, prox_qp
