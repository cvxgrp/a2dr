"""
Copyright 2018 Anqi Fu

This file is part of CVXConsensus.

CVXConsensus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXConsensus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXConsensus. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cvxpy.utilities as u
from cvxpy.problems.problem import Problem, Minimize
from cvxconsensus.consensus import consensus

class Problems(object):
	"""A list of convex optimization problems.

    Problems are immutable, save for modification through the specification
    of :class:`~cvxpy.expressions.constants.parameters.Parameter`

    Parameters
    ----------
    problems : list
        The list of problems.
    """
	def __init__(self, problems):
		# Check that list contains only problems.
		for prob in problems:
			if not isinstance(prob, Problem):
				raise TypeError("Input list must contain only Problems.")
		self._problems = problems
		self._value = None
		self._status = None
		self._residuals = None
		self._solver_stats = None
		
		self.combined = self._combined()
		self.partition = self._partition()
		self.args = problems
	
	@property
	def value(self):
		"""float : The value from the last time the problem was solved
				   (or None if not solved).
		"""
		return self._value
	
	@property
	def status(self):
		"""str : The status from the last time the problem was solved; one
                 of optimal, infeasible, or unbounded.
        """
		return self._status
	
	@property
	def problems(self):
		"""list : The list of problems.
		"""
		return self._problems
	
	@property
	def residuals(self):
		"""list : The l2-normed residuals for each iteration, i.e.
				  ||G(v^(k))||^2 where v^(k) = (y^(k), s^(k)) and G(.)
				  is the mapping G(v^(k)) = v^(k) - v^(k+1).
		"""
		return self._residuals
	
	def _combined(self):
		"""Sum list of problems with sign flip if objective is maximization.
		
		The combined problem's objective is to minimize the sum (with sign flip)
		of the component problem's objectives, subject to the union of all
		the constraints.
		"""
		p_comb = Problem(Minimize(0))
		for prob in self.problems:
			if isinstance(prob.objective, Minimize):
				p_comb += prob
			else:
				p_comb -= prob
		return p_comb
	
	def _partition(self):
		"""Determine which variables are in each problem.
		
		Returns a boolean array with variables (rows) by problems (columns).
		True at (i,j) indicates variable i is in problem j, either as part
		of the objective and/or constraints.
		"""
		v_ids = [v.id for v in self.variables()]
		n_vars = len(self.variables())
		n_probs = len(self.problems)
		
		p_idx = 0
		v_table = np.zeros((n_vars, n_probs), dtype=bool)
		for prob in self.problems:
			for var in prob.variables():
				v_idx = v_ids.index(var.id)
				v_table[v_idx, p_idx] = True
			p_idx = p_idx + 1
		return v_table
	
	@property
	def objective(self):
		"""Minimize : The combined problem's objective.

        Note that the objective cannot be reassigned after creation,
        and modifying the objective after creation will result in
        undefined behavior.
        """
		return self.combined.objective
	
	@property
	def constraints(self):
		"""A shallow copy of the combined problem's constraints.

		Note that constraints cannot be reassigned, appended to, or otherwise
		modified after creation, except through parameters.
		"""
		return self.combined.constraints

	def variables(self):
		"""Accessor method for variables.

        Returns
        ----------
        list of :class:`~cvxpy.expressions.variable.Variable`
            A list of the variables in the combined problem.
        """
		return self.combined.variables()
	
	def parameters(self):
		"""Accessor method for parameters.

        Returns
        -------
        list of :class:`~cvxpy.expressions.constants.parameter.Parameter`
            A list of the parameters in the combined problem.
        """
		return self.combined.parameters()
	
	def constants(self):
		"""Accessor method for constants.

        Returns
        ----------
        list of :class:`~cvxpy.expressions.constants.constant.Constant`
            A list of the constants in the combined problem.
        """
		return self.combined.constants()
	
	def pretty_vars(self):
		"""Pretty print variable partition across problems.
		"""
		from tabulate import tabulate
		N = len(self.problems)
		v_ids = ["Var%d" % v.id for v in self.variables()]
		table = np.column_stack((v_ids, self.partition))
		headers = [''] + ["Prob%d" % idx for idx in range(0,N)]
		print(tabulate(table, headers=headers, tablefmt='orgtbl'))
        
	@property
	def solver_stats(self):
		""":class:`~cvxpy.problems.problem.SolverStats` : Information returned by the solver.
		"""
		return self._solver_stats
	
	def solve(self, *args, **kwargs):
		"""Solves the problem using the specified method.
		
		Parameters
		----------
		method : str
		     The solve method to use. "combined" solves the combined problem directly, while
		     "consensus" solves the list of problems by consensus ADMM.
		"""
		func_name = kwargs.pop("method", None)
		if func_name == "combined":
			return self.combined.solve(*args, **kwargs)
		elif func_name == "consensus":
			sol = consensus(self.problems, *args, **kwargs)
			self.unpack_results(sol)
			return self.value
		else:
			raise NotImplementedError
	
	def unpack_results(self, solution):
		"""Updates the problem state given consensus results.
		
		Updates problem.status, problem.value, and value of the primal variables.
		
		Parameters
		----------
		solution : dict
		     Consensus solution to the combined problem. "zvals" refers to the
		     final average of the primal value over all the workers.
		"""
		# Save primal values.
		for v in self.variables():
			v.save_value(solution["zvals"][v.id])
	
		# TODO: Save dual values (for constraints too?).
		
		# Save combined objective.
		self._value = self.objective.value
		if not np.isscalar(self._value):
			self._value = np.asscalar(self._value)
		
		# Save residual from fixed point mapping.
		self._residuals = solution["residuals"]
		
		# TODO: Handle statuses.
		self._solver_stats = {"num_iters": solution["num_iters"],
							  "solve_time": solution["solve_time"]}
	
	def plot_residuals(self, normalize = True, semilogy = False):
		"""Plot the l2-normed residual over all iterations.
		
		Parameters
		----------
		normalize : logical
		     Should the residuals be normalized by their initial iteration value?
		"""
		if self._solver_stats is None:
			raise ValueError("Solver stats is empty. Nothing to plot.")
		iters = range(self._solver_stats["num_iters"])
		resid = self._residuals/self._residuals[0] if normalize and self._residuals[0] != 0 else self._residuals

		if semilogy:
			plt.semilogy(iters, resid)
		else:
			plt.plot(iters, resid)
		plt.xlabel("Iteration")
		plt.ylabel("Residual")
		plt.show()
