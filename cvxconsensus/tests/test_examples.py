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

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from cvxpy import Variable, Parameter, Problem, Minimize, Maximize
from cvxpy.atoms import *
import cvxconsensus
from cvxconsensus import Problems
from cvxconsensus.utilities import assign_rho, partition_vars
from cvxconsensus.tests.base_test import BaseTest

class TestExamples(BaseTest):
	"""Unit tests for examples"""

	def setUp(self):
		self.eps_stop = 1e-8
		self.eps_abs = 1e-16
		self.MAX_ITER = 1000
	
	def test_ols(self):
		# Solve the following consensus problem using ADMM:
		# Minimize sum(f_i(x)), where f_i(x) = square(norm(x - a_i))

		# Generate a_i's.
		np.random.seed(0)
		a = np.random.randn(3,10)
		
		# Construct separate problems.
		x = Variable(3)
		funcs = [square(norm(x - a_i)) for a_i in a.T]
		p_list = [Problem(Minimize(f_i)) for f_i in funcs]
		probs = Problems(p_list)
		probs.pretty_vars()
		
		# Solve via consensus.
		probs.solve(method = "consensus", rho_init = 5, max_iter = 50)
		print("Objective:", probs.value)
		print("Solution:", x.value)
	
	def test_lasso(self):
		# Solve the following consensus problem using ADMM:
		# Minimize sum_squares(A*x - b) + gamma*norm(x,1)
		
		# Problem data.
		m = 100
		n = 75
		np.random.seed(1)
		A = np.random.randn(m,n)
		b = np.random.randn(m)

		# Problem parameters.
		rho = 1.0
		m_accel = 5
		
		# Separate penalty from regularizer.
		x = Variable(n)
		gamma = Parameter(nonneg = True)
		funcs = [sum_squares(A*x - b), gamma*norm(x,1)]
		p_list = [Problem(Minimize(f)) for f in funcs]
		probs = Problems(p_list)
		
		# Solve via consensus.
		gamma.value = 1.0
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER, \
					warm_start = False, eps_stop = self.eps_stop)
		res_sdrs = probs.residuals
		print("S-DRS Objective:", probs.value)

		# Solve via consensus with Anderson acceleration.
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER, \
					warm_start = False, eps_stop = self.eps_stop, anderson = True, m_accel = m_accel)
		res_aa2 = probs.residuals
		print("S-DRS with AA-II Objective:", probs.value)

		# Plot and compare residuals.
		self.compare_residuals(res_sdrs, [res_aa2], [m_accel])

	def test_svm(self):
		NUM_PROCS = 4
		SPLIT_SIZE = 250
		
		# Problem data.
		np.random.seed(1)
		N = NUM_PROCS*SPLIT_SIZE
		n = 10
		offset = np.random.randn(n,1)
		data = []
		for i in range(int(N/2)):
			data += [(1, offset + np.random.normal(1.0, 2.0, (n, 1)))]
		for i in range(int(N/2)):
			data += [(-1, offset + np.random.normal(-1.0, 2.0, (n, 1)))]
		data_splits = [data[i:i+SPLIT_SIZE] for i in range(0, N, SPLIT_SIZE)]
		
		# Construct problem.
		w = Variable(n + 1)
		def svm(data):
			slack = [pos(1 - b*(a.T*w[:-1] - w[-1])) for (b, a) in data]
			return norm(w, 2) + sum(slack)
		funcs = map(svm, data_splits)
		p_list = [Problem(Minimize(f_i)) for f_i in funcs]
		probs = Problems(p_list)
		
		# Solve via consensus using spectral step size adjustment.
		probs.solve(method = "consensus", rho_init = 1.0, max_iter = 20)
		print("Objective:", probs.value)
		print("Solution:", w.value)
	
		# Count misclassifications.
		def get_error(w):
			error = 0
			for label, sample in data:
				if not label*(np.dot(w[:-1].T, sample) - w[-1])[0] > 0:
					error += 1
			return "%d misclassifications out of %d samples" % (error, N)
		print("Misclassifications:", get_error(w.value))
	
	def test_flow_control(self):
		# Problem data.
		np.random.seed(1)
		m = 40
		m_a = 20
		n = 22
		n_a = 5
		n_b = 5
		R = np.vstack((np.hstack((np.round(rand(m_a,n_a)), np.zeros((m_a,n_b)), np.round(rand(m_a,n-n_a-n_b)))),
					   np.hstack((np.zeros((m-m_a,n_a)), np.round(rand(m-m_a,n_b)), np.round(rand(m-m_a,n-n_a-n_b))))
					 ))
		c = 5*rand(m)

		# Problem parameters.
		rho = 10
		m_accel = 5
		
		# Find optimum directly.
		f_star = Variable(n)
		prob = Problem(Maximize(sum(sqrt(f_star))), [R*f_star <= c])
		prob.solve()
		print("True Objective:", prob.value)
		print("True Solution:", f_star.value)
		
		# Partition data into two groups with overlap.
		R_a = R[:m_a,:n_a]
		R_b = R[m_a:,n_a:(n_a + n_b)]
		S_a = R[:m_a,(n_a + n_b):]
		S_b = R[m_a:,(n_a + n_b):]
		c_a = c[:m_a]
		c_b = c[m_a:]
		n_ab = n - n_a - n_b
		
		# Define separate problem for each group.
		f_a = Variable(n_a)
		f_b = Variable(n_b)
		x = Variable(n_ab)
		p_list = [Problem(Maximize(sum(sqrt(f_a)) + 0.5*sum(sqrt(x))), [R_a*f_a + S_a*x <= c_a]),
				  Problem(Maximize(sum(sqrt(f_b)) + 0.5*sum(sqrt(x))), [R_b*f_b + S_b*x <= c_b])]
		probs = Problems(p_list)

		# Solve via consensus.
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER, \
					warm_start = False, eps_stop = self.eps_stop)
		res_sdrs = probs.residuals
		print("S-DRS Objective:", -probs.value)  # TODO: All problems recast as minimization, so flip sign of objective to compare.
		print("S-DRS Solution:", np.hstack((f_a.value, f_b.value, x.value)))

		# Solve via consensus with Anderson acceleration.
		probs.solve(method = "consensus", rho_init = rho, max_iter = self.MAX_ITER, \
					warm_start = False, eps_stop = self.eps_stop, anderson = True, m_accel = m_accel)
		res_aa2 = probs.residuals
		print("S-DRS with AA-II Objective:", -probs.value)
		print("S-DRS with AA-II Solution:", np.hstack((f_a.value, f_b.value, x.value)))

		# Plot and compare residuals.
		self.compare_residuals(res_sdrs, [res_aa2], [m_accel])
	
	def test_vehicle_formation(self):
		# References:
		# EE364B Exercises, Chapter 12, Question 12.1 (MPC for output tracking).
		#    http://stanford.edu/class/ee364b/364b_exercises.pdf
		# Raffard, Tomlin, Boyd. "Distributed Optimization for Cooperative Agents: Application to Formation Flight."
		#    Proceedings IEEE Conference on Decision and Control, 3:2453-2459, Nassau, Bahamas, December 2004.
		#    http://stanford.edu/~boyd/papers/form_flight.html
		def plot_control(T, u, Umax, title = None):
			Umax_vec = np.repeat(Umax, T)
			Umax_lines = np.column_stack((Umax_vec, -Umax_vec))
			plt.plot(range(T), u)
			plt.plot(range(T), Umax_lines, color = "red", linestyle = "dashed")
			plt.xlabel("Time (t)")
			plt.ylabel("Input (u(t))")
			if title is not None:
				plt.title(title)
			plt.show()

		def plot_output(T, y, ydes, title = None):
			plt.plot(range(T), y)
			plt.plot(range(T), ydes, color = "red", linestyle = "dashed")
			plt.xlabel("Time (t)")
			plt.ylabel("Output (y(t))")
			if title is not None:
				plt.title(title)
			plt.show()
		
		# Problem data.
		T = 100
		Umax = 0.1
		A = np.array([[1, 1, 0],
					  [0, 1, 1],
					  [0, 0, 1]])
		B = np.array([[0], [0.5], [1]])
		C = np.array([[-1, 0, 1]])
		ydes = np.zeros((1,T))
		ydes[0,30:70] = 10

		# Problem parameters.
		rho = 0.5
		m_accel = 5
		
		# Define leader vehicle.
		x = Variable((3,T+1))
		y = Variable((1,T))
		u = Variable((1,T))
		
		J = sum_squares(y - ydes)
		constr = [x[:,0] == 0, x[:,1:] == A*x[:,:T] + B*u, \
				  y == C*x[:,:T], norm(u, "inf") <= Umax]
		prob = Problem(Minimize(J), constr)
		prob.solve()
		print("Single Vehicle Objective:", prob.value)
		
		# Plot input and output dynamics.
		plot_control(T, u.value.T, Umax, title = "Single Vehicle Control Input")
		plot_output(T, y.value.T, ydes.T, title = "Single Vehicle Path Dynamics")
		
		# Define follower vehicles.
		ydlt_l = -1
		x_l = Variable((3,T+1))
		y_l = Variable((1,T))
		u_l = Variable((1,T))
		J_l = sum_squares(y_l - y - ydlt_l)
		constr_l = [x_l[:,0] == 0, x_l[:,1:] == A*x_l[:,:T] + B*u_l, \
				  y_l == C*x_l[:,:T], norm(u_l, "inf") <= Umax]
		prob_l = Problem(Minimize(J_l), constr_l)
		
		ydlt_r = 1
		x_r = Variable((3,T+1))
		y_r = Variable((1,T))
		u_r = Variable((1,T))
		J_r = sum_squares(y_r - y - ydlt_r)
		constr_r = [x_r[:,0] == 0, x_r[:,1:] == A*x_r[:,:T] + B*u_r, \
					y_r == C*x_r[:,:T], norm(u_r, "inf") <= Umax]
		prob_r = Problem(Minimize(J_r), constr_r)
		
		# Solve formation consensus problem.
		probs = Problems([prob, prob_l, prob_r])
		probs.solve(method = "consensus", solver = "ECOS", rho_init = rho, max_iter = self.MAX_ITER, \
					warm_start = False, eps_stop = self.eps_stop, abstol = self.eps_abs)
		res_sdrs = probs.residuals
		print("Leader-Follower S-DRS Objective:", probs.value)

		# Solve formation consensus problem with AA-II.
		probs.solve(method = "consensus", solver = "ECOS", rho_init = rho, max_iter = self.MAX_ITER, \
					warm_start = False, eps_stop = self.eps_stop, abstol = self.eps_abs, \
					anderson = True, m_accel = m_accel)
		res_aa2 = probs.residuals
		print("Leader-Follower S-DRS with AA-II Objective:", probs.value)

		# Plot and compare residuals.
		self.compare_residuals(res_sdrs, [res_aa2], [m_accel])
		
		# Plot input and output dynamics.
		u_comb = np.column_stack((u.value.T, u_l.value.T, u_r.value.T))
		y_comb = np.column_stack((y.value.T, y_l.value.T, y_r.value.T))
		plot_control(T, u_comb, Umax, title = "Leader-Follower Control Input")
		plot_output(T, y_comb, ydes.T, title = "Leader-Follower Path Dynamics")
