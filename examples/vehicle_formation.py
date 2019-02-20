import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from cvxconsensus import *

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

def main():
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
	probs.solve(method = "consensus", solver = "ECOS", rho_init = 0.5)
	print("Leader-Follower Objective:", probs.value)
	
	# Plot input and output dynamics.
	u_comb = np.column_stack((u.value.T, u_l.value.T, u_r.value.T))
	y_comb = np.column_stack((y.value.T, y_l.value.T, y_r.value.T))
	plot_control(T, u_comb, Umax, title = "Leader-Follower Control Input")
	plot_output(T, y_comb, ydes.T, title = "Leader-Follower Path Dynamics")

if __name__ == '__main__':
	main()
