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

class TestProjection(BaseTest):
    """Unit tests for projections"""

    def setUp(self):
        np.random.seed(1)
        self.TOLERANCE = 1e-6
        self.c = np.random.randn()
        self.v = np.random.randn(100)
        self.B = np.random.randn(50,10)

        self.radii = [0, 1, 2.5]
        self.radii += [np.abs(self.c) + self.TOLERANCE]
        self.radii += [np.max(np.abs(self.v)) + self.TOLERANCE,
                       (np.max(np.abs(self.v)) + np.min(np.abs(self.v)))/2]
        self.radii += [np.max(np.abs(self.B)) + self.TOLERANCE,
                       (np.max(np.abs(self.B)) + np.min(np.abs(self.B)))/2]

    def check_simplex(self, val, radius=1, method="bisection", places=4, *args, **kwargs):
        # Solve with CVXPY.
        x_var = Variable() if np.isscalar(val) else Variable(val.shape)
        obj = Minimize(sum_squares(x_var - val))
        constr = [x_var >= 0, sum(x_var) == radius]
        Problem(obj, constr).solve(*args, **kwargs)

        # Solve with projection algorithm.
        x_proj = proj_simplex(val, r=radius, method=method)
        self.assertItemsAlmostEqual(x_var.value, x_proj, places=places)

    def check_l1_ball(self, val, radius=1, method="bisection", places=4, *args, **kwargs):
        # Solve with CVXPY.
        x_var = Variable() if np.isscalar(val) else Variable(val.shape)
        obj = Minimize(sum_squares(x_var - val))
        constr = [norm1(x_var) <= radius]
        Problem(obj, constr).solve(*args, **kwargs)

        # Solve with projection algorithm.
        x_proj = proj_l1(val, r=radius, method=method)
        self.assertItemsAlmostEqual(x_var.value, x_proj, places=places)

    def test_simplex(self):
        for radius in self.radii:
            for method in ["bisection", "sorted"]:
                self.check_simplex(self.c, radius, method)
                self.check_simplex(self.v, radius, method)
                self.check_simplex(self.B, radius, method)

    def test_l1_ball(self):
        for radius in self.radii:
            for method in ["bisection", "sorted"]:
                self.check_l1_ball(self.c, radius, method, solver='ECOS')
                self.check_l1_ball(self.v, radius, method)
                self.check_l1_ball(self.B, radius, method)
