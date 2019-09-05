"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
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

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=4):
        super(BaseTest, self).assertAlmostEqual(a.real, b.real, places=places)
        super(BaseTest, self).assertAlmostEqual(a.imag, b.imag, places=places)

    def mat_to_list(self, mat):
        """Convert a numpy matrix to a list.
        """
        if isinstance(mat, (np.matrix, np.ndarray)):
            return np.asarray(mat).flatten('F').tolist()
        else:
            return mat

    def plot_residuals(self, r_primal, r_dual, normalize = False, show = True, title = None, semilogy = False, savefig = None, *args, **kwargs):
        if normalize:
            r_primal = r_primal / r_primal[0] if r_primal[0] != 0 else r_primal
            r_dual = r_dual / r_dual[0] if r_dual[0] != 0 else r_dual

        if semilogy:
            plt.semilogy(range(len(r_primal)), r_primal, label = "Primal", *args, **kwargs)
            plt.semilogy(range(len(r_dual)), r_dual, label = "Dual", *args, **kwargs)
        else:
            plt.plot(range(len(r_primal)), r_primal, label = "Primal", *args, **kwargs)
            plt.plot(range(len(r_dual)), r_dual, label = "Dual", *args, **kwargs)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        if title:
            plt.title(title)
        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")

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

    def compare_primal_dual(self, drs_result, a2dr_result, savefig = None):
        # Compare residuals
        plt.semilogy(range(drs_result["num_iters"]), drs_result["primal"], color="blue", linestyle="--",
                     label="Primal (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["primal"], color="blue", label="Primal (A2DR)")
        plt.semilogy(range(drs_result["num_iters"]), drs_result["dual"], color="darkorange", linestyle="--",
                     label="Dual (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), a2dr_result["dual"], color="darkorange", label="Dual (A2DR) ")
        # plt.title("Residuals")
        plt.legend()
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
        plt.show()
        
    def compare_total(self, drs_result, a2dr_result, savefig = None):
        # Compare residuals
        plt.semilogy(range(drs_result["num_iters"]), np.sqrt(drs_result["primal"]**2+drs_result["dual"]**2), color="blue", label="Residuals (DRS)")
        plt.semilogy(range(a2dr_result["num_iters"]), np.sqrt(a2dr_result["primal"]**2+a2dr_result["dual"]**2), color="darkorange", label="Residuals (A2DR)")
        # plt.title("Residuals")
        plt.legend()
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
        plt.show()
        
    def compare_total_all(self, results, names, savefig = None):
        # Compare residuals in the results list
        # len(names) must be equal to len(results)
        for i in range(len(names)):
            result = results[i]
            name = names[i]
            plt.semilogy(range(result["num_iters"]), np.sqrt(result["primal"]**2+result["dual"]**2), 
                         label="Residuals (" + name + ")")
        # plt.title("Residuals")
        plt.legend()
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
        plt.show()
