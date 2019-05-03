"""
Copyright 2018 Anqi Fu

This file is part of CVXConsensus.

CVXConsensus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXConsensus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXConsensus. If not, see <http://www.gnu.org/licenses/>.
"""

# Base class for unit tests.
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

class BaseTest(TestCase):
    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b, places=4):
        if np.isscalar(a):
            a = [a]
        else:
            a = self.mat_to_list(a)
        if np.isscalar(b):
            b = [b]
        else:
            b = self.mat_to_list(b)
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], places)

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=4):
        super(BaseTest, self).assertAlmostEqual(a, b, places=places)

    def mat_to_list(self, mat):
        """Convert a numpy matrix to a list.
        """
        if isinstance(mat, (np.matrix, np.ndarray)):
            return np.asarray(mat).flatten('F').tolist()
        else:
            return mat

    def plot_residuals(self, r_primal, r_dual, normalize = False, show = True, title = None, semilogy = False):
        if normalize:
            r_primal = r_primal / r_primal[0] if r_primal[0] != 0 else r_primal
            r_dual = r_dual / r_dual[0] if r_dual[0] != 0 else r_dual

        if semilogy:
            plt.semilogy(range(len(r_primal)), r_primal, label = "Primal")
            plt.semilogy(range(len(r_dual)), r_dual, label = "Dual")
        else:
            plt.plot(range(len(r_primal)), r_primal, label = "Primal")
            plt.plot(range(len(r_dual)), r_dual, label = "Dual")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        if title:
            plt.title(title)
        if show:
            plt.show()

    def compare_results(self, probs, obj_a2dr, obj_comb, x_a2dr, x_comb):
        N = len(probs.variables())
        for i in range(N):
            print("\nA2DR Solution:\n", x_a2dr[i])
            print("Base Solution:\n", x_comb[i])
            print("MSE: ", np.mean(np.square(x_a2dr[i] - x_comb[i])), "\n")
        print("A2DR Objective: %f" % obj_a2dr)
        print("Base Objective: %f" % obj_comb)
        print("Iterations: %d" % probs.solver_stats["num_iters"])
        print("Elapsed Time: %f" % probs.solver_stats["solve_time"])

    def compare_residuals(self, res_drs, res_a2dr, m_vals):
        if not isinstance(res_a2dr, list):
            res_a2dr = [res_a2dr]
        if not isinstance(m_vals, list):
            m_vals = [m_vals]
        if len(m_vals) != len(res_a2dr):
            raise ValueError("Must have same number of AA-II residuals as memory parameter values")

        plt.semilogy(range(res_drs.shape[0]), res_drs, label="DRS")
        for i in range(len(m_vals)):
            label = "A2DR (m = {})".format(m_vals[i])
            plt.semilogy(range(res_a2dr[i].shape[0]), res_a2dr[i], linestyle="--", label=label)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.show()