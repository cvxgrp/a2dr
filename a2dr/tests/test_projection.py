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

import numpy as np
from cvxpy import *

from a2dr.proximal.projection import *
from a2dr.tests.base_test import BaseTest

class TestBasic(BaseTest):
    """Unit tests for projections"""

    def setUp(self):
        np.random.seed(1)
        self.TOLERANCE = 1e-6
        self.v = np.random.randn(100)

    def test_simplex(self):
        for radius in [0, 1, 2.5]:
            # Solve with CVXPY.
            x_var = Variable(self.v.shape)
            obj = Minimize(sum_squares(x_var - self.v))
            constr = [x_var >= 0, sum(x_var) == radius]
            Problem(obj, constr).solve()

            # Solve with bisection algorithm.
            x_proj = proj_simplex(self.v, r = radius, method="bisection")
            self.assertItemsAlmostEqual(x_var.value, x_proj, places = 4)

            # Solve with efficient projection algorithm.
            # x_proj = proj_simplex(self.v, r = radius, method="efficient")
            # self.assertItemsAlmostEqual(x_var.value, x_proj, places = 4)

    def test_l1_ball(self):
        for radius in [0, 1, 2.5]:
            # Solve with CVXPY.
            x_var = Variable(self.v.shape)
            obj = Minimize(sum_squares(x_var - self.v))
            constr = [norm1(x_var) <= radius]
            Problem(obj, constr).solve()

            # Solve with bisection algorithm.
            x_proj = proj_l1(self.v, r = radius, method="bisection")
            self.assertItemsAlmostEqual(x_var.value, x_proj, places = 4)

            # Solve with efficient projection algorithm.
            # x_proj = proj_simplex(self.v, r = radius, method="efficient")
            # self.assertItemsAlmostEqual(x_var.value, x_proj, places = 4)
