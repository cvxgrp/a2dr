import numpy as np
import scipy as sp
import numpy.linalg as LA
import matplotlib.pyplot as plt

from cvxpy import *
from scipy import sparse
from scipy.optimize import nnls
from sklearn.datasets import make_sparse_spd_matrix

from cvxconsensus import a2dr
from cvxconsensus.proximal.prox_operators import prox_logistic
from cvxconsensus.tests.base_test import BaseTest

def prox_norm1(alpha = 1.0):
    return lambda v, t: np.maximum(v - t*alpha,0) - np.maximum(-v - t*alpha,0)

def prox_norm2(alpha = 1.0):
    return lambda v, t: np.maximum(1 - 1.0/(t*alpha*LA.norm(v,2)),0) * v

def prox_norm_inf(bound):
    if bound < 0:
        raise ValueError("bound must be a non-negative scalar.")
    return lambda v, t: np.maximum(np.minimum(v, bound), -bound)

def prox_nuc_norm(alpha = 1.0, order = 'C'):
    def prox(Q, t):
        U, s, Vt = np.linalg.svd(Q, full_matrices=False)
        s_new = np.maximum(s - t*alpha, 0)
        Q_new = U.dot(np.diag(s_new)).dot(Vt)
        return Q_new.ravel(order=order)
    return prox

def prox_group_lasso(alpha = 1.0):
    prox_inner = prox_norm2(alpha)
    return lambda Q, t: np.concatenate([prox_inner(Q[:,j], t) for j in range(Q.shape[1])])

def prox_quad_form(Q):
    if not np.all(LA.eigvals(Q) >= 0):
        raise Exception("Q must be a positive semidefinite matrix.")
    return lambda v, t: LA.lstsq(Q + (1/t)*np.eye(v.shape[0]), v/t, rcond=None)[0]

def prox_sum_squares(X, y, type = "lsqr"):
    n = X.shape[1]
    if type == "lsqr":
        X = sparse.csc_matrix(X)
        def prox(v, t):
            A = sparse.vstack((X, 1/np.sqrt(2*t)*sparse.eye(n)))
            b = np.concatenate((y, 1/np.sqrt(2*t)*v))
            return sparse.linalg.lsqr(A, b, atol=1e-16, btol=1e-16)[0]
    elif type == "lstsq":
        def prox(v, t):
           A = np.vstack((X, 1/np.sqrt(2*t)*np.eye(n)))
           b = np.concatenate((y, 1/np.sqrt(2*t)*v))
           return LA.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError("Algorithm type not supported:", type)
    return prox

def prox_neg_log_det(Q, t, order = 'C', PSD = False):
    Q_symm = (Q + Q.T) / 2.0
    # if not np.allclose(Q, Q_symm):
    #     raise Exception("Proximal operator for negative log-determinant only operates on symmetric matrices.")
    s, u = LA.eig(Q_symm)
    if PSD:
    	s = np.maximum(s, 0)
    s_new = (s + np.sqrt(s**2 + 4.0*t))/2
    Q_new = u.dot(np.diag(s_new)).dot(u.T)
    return Q_new.ravel(order=order)

class TestPaper(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8
        self.eps_abs = 1e-6
        self.MAX_ITER = 1000

	def test_nnls(self):
        # minimize ||y - X\beta||_2^2 subject to \beta >= 0.

        # Problem data.
        n = 1000
        beta_true = np.array(np.arange(-n/2,n/2) + 1)
        X = np.random.randn(n,n)
        y = X.dot(beta_true) + np.random.randn(n)

        # Solve with SciPy.
        sp_result = nnls(X, y)
        sp_beta = sp_result[0]
        sp_obj = sp_result[1]**2   # SciPy objective is ||y - X\beta||_2.

        # Convert problem to standard form.
        # f_1(\beta_1) = ||y - X\beta_1||_2^2, f_2(\beta_2) = I(\beta_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [prox_sum_squares(X, y), lambda v, t: np.maximum(v, 0)]
        A_list = [sparse.eye(n), -sparse.eye(n)]
        b = sparse.csc_matrix((n,))

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False)
        drs_beta = drs_result["x_vals"][-1]
        drs_obj = np.sum((y - X.dot(drs_beta))**2)
        self.assertAlmostEqual(sp_obj, drs_obj)
        self.assertItemsAlmostEqual(sp_beta, drs_beta, places=3)

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True)
        a2dr_beta = a2dr_result["x_vals"][-1]
        a2dr_obj = np.sum((y - X.dot(a2dr_beta))**2)
        self.assertAlmostEqual(sp_obj, a2dr_obj)
        self.assertItemsAlmostEqual(sp_beta, a2dr_beta, places=3)
        self.compare_primal_dual(drs_result, a2dr_result)

	def test_sparse_inv_covariance(self):
        # minimize -log(det(S)) + trace(S*Y) + \alpha*||S||_1 subject to S is symmetric PSD.

        # Problem data.
        n = 25    # Dimension of matrix.
        m = 1000  # Number of samples.
        ratio = 0.85   # Fraction of zeros in S.
        alpha = 1.0

        S_true = sparse.csc_matrix(make_sparse_spd_matrix(n, ratio))
        R = sparse.linalg.inv(S_true).todense()
        q_sample = sp.linalg.sqrtm(R).dot(np.random.randn(n,m))
        Q = np.cov(q_sample)

        # Solve with CVXPY.
        S = Variable((n,n), PSD=True)
        obj = -log_det(S) + trace(S*Q) + alpha*norm1(S)
        prob = Problem(Minimize(obj))
        prob.solve(eps=self.eps_abs)
        cvxpy_obj = prob.value
        cvxpy_S = S.value

        # Convert problem to standard form.
        # f_1(S) = -log(det(S)) on symmetric PSD matrices, f_2(S) = trace(S*Q), f_3(S) = \alpha*||S||_1.
        # A_1 = [I; 0], A_2 = [-I; I], A_3 = [0; -I], b = 0.
        prox_list = [lambda v, t: prox_neg_log_det(v.reshape((n,n), order='C'), t, order='C', PSD=True),
                  	 lambda v, t: v - t*Y.ravel(order='C'),
                  	 lambda v, t: prox_norm1(alpha)]
        A_list = [sparse.vstack([sparse.eye(n*n), sparse.csc_matrix((n*n,n*n))]),
        		  sparse.vstack([-sparse.eye(n*n), sparse.eye(n*n)]),
        		  sparse.vstack([sparse.csc_matrix((n*n,n*n)), -sparse.eye(n*n)])]
        b = np.zeros(2*n*n)

   		# Ensure initial point is symmetric PSD.
        S_init = np.random.randn(n,n)
        S_init = S_init.T.dot(S_init)
        v_init = 3*[S_init.ravel(order='C')]

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, v_init, eps_abs=self.eps_abs, anderson=False)
        drs_S = drs_result["x_vals"][-1].reshape((n,n), order='C')
        drs_obj = -LA.slogdet(drs_S)[1] + np.sum(np.diag(drs_S.dot(Q))) + alpha*np.sum(np.abs(drs_S))
        # self.assertAlmostEqual(cvxpy_obj, drs_obj, places=3)
        # self.assertItemsAlmostEqual(cvxpy_S, drs_S, places=3)

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, v_init, eps_abs=self.eps_abs, anderson=True)
        a2dr_S = a2dr_result["x_vals"][-1].reshape((n,n), order='C')
        a2dr_obj = -LA.slogdet(a2dr_S)[1] + np.sum(np.diag(a2dr_S.dot(Q))) + alpha*np.sum(np.abs(a2dr_S))
        # self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        # self.assertItemsAlmostEqual(cvxpy_S, a2dr_S)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_l1_trend_filtering(self):
        # minimize (1/2)||y - x||_2^2 + \alpha*||Dx||_1,
        # where (Dx)_{t-1} = x_{t-1} - 2*x_t + x_{t+1} for t = 2,...,n-1.
        # Reference: https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

        # Problem data.
        n = 1000
        alpha = 1.0
        y = np.random.randn(n)

        # Solve with CVXPY.
        x = Variable(n)
        obj = sum_squares(y - x)/2 + alpha*norm1(diff(x,2))
        prob = Problem(Minimize(obj))
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_x = x.value

        # Form second difference matrix.
        D = sparse.lil_matrix(sparse.eye(n))
        D.setdiag(-2, k = 1)
        D.setdiag(1, k = 2)
        D = D[:(n-2),:]

        # Convert problem to standard form.
        # f_1(x_1) = (1/2)||y - x_1||_2^2, f_2(x_2) = \alpha*||x_2||_1.
        # A_1 = D, A_2 = -I_{n-2}, b = 0.
        prox_list = [lambda v, t: (t*y + v)/(t + 1.0), prox_norm1(alpha)]
        A_list = [D, -sparse.eye(n-2)]
        b = sparse.csc_matrix((n-2,))

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False)
        drs_x = drs_result["x_vals"][0]
        drs_obj = np.sum((y - drs_x)**2)/2 + alpha*np.sum(np.abs(np.diff(drs_x,2)))
        self.assertAlmostEqual(cvxpy_obj, drs_obj)
        self.assertItemsAlmostEqual(cvxpy_x, drs_x)

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True)
        a2dr_x = a2dr_result["x_vals"][0]
        a2dr_obj = np.sum((y - a2dr_x)**2)/2 + alpha*np.sum(np.abs(np.diff(a2dr_x,2)))
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj)
        self.assertItemsAlmostEqual(cvxpy_x, a2dr_x)
        self.compare_primal_dual(drs_result, a2dr_result)

    def test_multi_task_logistic(self):
        # minimize \sum_{ik} log(1 + exp(-Y_{ik}*Z_{ik})) + \alpha*||\theta||_{2,1} + \beta*||\theta||_*
        # subject to Z = X\theta, ||.||_{2,1} = group lasso, ||.||_* = nuclear norm.

        # Problem data.
        K = 5     # Number of tasks.
        p = 20    # Number of features.
        m = 100   # Number of samples.
        alpha = 1.0
        beta = 1.0

        X = np.random.randn(m,p)
        theta_true = np.random.randn(p,K)
        Z_true = X.dot(theta_true)
        Y = 2*(Z_true > 0) - 1   # Y_{ij} = 1 or -1.

        def calc_obj(theta):
            obj = np.sum(-np.log(sp.special.expit(np.multiply(Y, X.dot(theta)))))
            reg = alpha*np.sum([LA.norm(theta[:,k], 2) for k in range(K)])
            reg += beta*LA.norm(theta, ord='nuc')
            return obj + reg

        # Solve with CVXPY.
        theta = Variable((p,K))
        loss = sum(logistic(-multiply(Y, X*theta)))
        reg = alpha*sum(norm(theta, 2, axis=0)) + beta*normNuc(theta)
        prob = Problem(Minimize(loss + reg))
        prob.solve()
        cvxpy_obj = prob.value
        cvxpy_theta = theta.value

        # Convert problem to standard form. 
        # f_1(Z) = \sum_{ik} log(1 + exp(-Y_{ik}*Z_{ik})), 
        # f_2(\theta) = \alpha*||\theta||_{2,1}, 
        # f_3(\tilde \theta) = \beta*||\tilde \theta||_*.
        # A_1 = [I; 0], A_2 = [-X; I], A_3 = [0; -I], b = 0.
        prox_list = [lambda v, t: prox_logistic(v, t, y = Y.ravel(order='F')),   # TODO: Calculate in parallel for k = 1,...K.
        		  	 lambda v, t: prox_group_lasso(alpha)(v.reshape((p,K), order='F'), t),
                  	 lambda v, t: prox_nuc_norm(beta, order='F')(v.reshape((p,K), order='F'), t)]
        A_list = [sparse.vstack([sparse.eye(m*K), sparse.csc_matrix((p*K,m*K))]),
		  		  sparse.vstack([-sparse.block_diag(K*[X]), sparse.eye(p*K)]),
		  		  sparse.vstack([sparse.csc_matrix((m*K,p*K)), -sparse.eye(p*K)])]
        b = np.zeros(m*K + 2*p*K)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False)
        drs_theta = drs_result["x_vals"][-1].reshape((p,K), order='F')
        drs_obj = calc_obj(drs_theta)
        self.assertAlmostEqual(cvxpy_obj, drs_obj, places=3)
        # self.assertItemsAlmostEqual(cvxpy_theta, drs_theta)

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True)
        a2dr_theta = a2dr_result["x_vals"][-1].reshape((p,K), order='F')
        a2dr_obj = calc_obj(a2dr_theta)
        self.assertAlmostEqual(cvxpy_obj, a2dr_obj, places=3)
        self.assertItemsAlmostEqual(cvxpy_theta, a2dr_theta)
        self.compare_primal_dual(drs_result, a2dr_result)