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

from a2dr.proximal.constraint import prox_box_constr
from a2dr.proximal.constraint import prox_psd_cone

from a2dr.proximal.elementwise import prox_abs
from a2dr.proximal.elementwise import prox_entr
from a2dr.proximal.elementwise import prox_exp
from a2dr.proximal.elementwise import prox_huber

from a2dr.proximal.matrix import prox_neg_log_det
from a2dr.proximal.matrix import prox_sigma_max
from a2dr.proximal.matrix import prox_trace

from a2dr.proximal.misc import prox_logistic
from a2dr.proximal.misc import prox_max

from a2dr.proximal.norm import prox_group_lasso
from a2dr.proximal.norm import prox_norm1
from a2dr.proximal.norm import prox_norm2
from a2dr.proximal.norm import prox_norm_inf
from a2dr.proximal.norm import prox_norm_nuc

from a2dr.proximal.quadratic import prox_quad_form
from a2dr.proximal.quadratic import prox_sum_squares
from a2dr.proximal.quadratic import prox_sum_squares_affine
