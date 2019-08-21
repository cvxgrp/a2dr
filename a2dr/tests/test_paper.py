import numpy as np
import scipy as sp
import numpy.linalg as LA
import copy
import time
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from cvxpy import *
from scipy import sparse
from scipy.optimize import nnls
from sklearn.datasets import make_sparse_spd_matrix

from a2dr import a2dr
from a2dr.proximal.prox_operators import prox_logistic
from a2dr.tests.base_test import BaseTest

def prox_norm1(alpha = 1.0):
    return lambda v, t: (v - t*alpha).maximum(0) - (-v - t*alpha).maximum(0) if sparse.issparse(v) else \
                        np.maximum(v - t*alpha,0) - np.maximum(-v - t*alpha,0)

def prox_norm2(alpha = 1.0):
    def prox_norm2_inner(v, t):
        if np.linalg.norm(v) == 0:
            return np.zeros(len(v))
        elif sparse.issparse(v):
            return (1 - t*alpha*1.0/(sparse.linalg.norm(v,'fro'))).maximum(0) * v
        else: 
            return np.maximum(1 - t*alpha*1.0/(LA.norm(v,2)),0) * v 
            
    return lambda v, t: prox_norm2_inner(v, t)

def prox_norm_inf(bound):
    if bound < 0:
        raise ValueError("bound must be a non-negative scalar.")
    return lambda v, t: v.minimum(bound).maximum(-bound) if sparse.issparse(v) else \
                        np.maximum(np.minimum(v, bound), -bound)

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
    if sparse.issparse(Q):
        if not np.all(LA.eigvals(Q.todense()) >= 0):
            raise Exception("Q must be a positive semidefinite matrix.")
        return lambda v, t: sparse.linalg.lsqr(Q + (1/t)*sparse.eye(v.shape[0]), v/t, atol=1e-16, btol=1e-16)[0]
    else:
        if not np.all(LA.eigvals(Q) >= 0):
            raise Exception("Q must be a positive semidefinite matrix.")
        return lambda v, t: LA.lstsq(Q + (1/t)*np.eye(v.shape[0]), v/t, rcond=None)[0]
    
def prox_square(v, t):
    return v/(1.0+2*t)

# def prox_sat(v, t):
#     def sat(u, c):
#         return np.maximum(np.minimum(u, c), -c)
#     return sat(v/(1.0+2*t), 1)

def prox_sat(c, x_max):
    def sat(u, b):
        return np.maximum(np.minimum(u, b), -b)
    return lambda v, t: sat(v/(1.0+2*t*c), x_max)

def prox_sat_pos(c, x_max):
    def sat_pos(u, b):
        return np.maximum(np.minimum(u, b), 0)
    return lambda v, t: sat_pos(v/(1.0+2*t*c), x_max)

    
def prox_sum_squares(X, y, type = "lsqr"):
    n = X.shape[1]
    if type == "lsqr":
        X = sparse.csr_matrix(X)
        def prox(v, t):
            A = sparse.vstack([X, 1/np.sqrt(2*t)*sparse.eye(n)])
            b = np.concatenate([y, 1/np.sqrt(2*t)*v])
            return sparse.linalg.lsqr(A, b, atol=1e-16, btol=1e-16)[0]
    elif type == "lstsq":
        def prox(v, t):
            A = np.vstack([X, 1/np.sqrt(2*t)*np.eye(n)])
            b = np.concatenate([y, 1/np.sqrt(2*t)*v])
            return LA.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError("Algorithm type not supported:", type)
    return prox

def prox_qp(Q, q, F, g):
    # check warmstart/parameter mode -- make sure the problem reduction is only done once
    n = Q.shape[0]
    I = np.eye(n)
    v_par = Parameter(n)
    t_par = Parameter(nonneg=True)
    x = Variable(n)
    obj = quad_form(x, Q) + sum_squares(x)/2/t_par + (q-v_par/t_par)*x
    constr = [F * x <= g]
    prob = Problem(Minimize(obj), constr)
    def prox_qp1(v, t):
        v_par.value, t_par.value = v, t
        prob.solve()
        return x.value
    return prox_qp1

def prox_neg_log_det_aff(Q, S, t, order = 'C'):
    Q_symm = (Q + Q.T) / 2.0
    Q_symm_S = Q_symm - t*S
    s, u = LA.eigh(Q_symm_S)
    s_new = (s + np.sqrt(s**2 + 4.0*t))/2
    Q_new = u.dot(np.diag(s_new)).dot(u.T)
    return Q_new.ravel(order=order)

class TestPaper(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8   # specify these in all examples?
        self.eps_abs = 1e-6
        self.MAX_ITER = 1000

    def test_nnls(self, figname):
        # minimize ||y - X\beta||_2^2 subject to \beta >= 0.

        # Problem data.
        m, n = 10000, 8000
        density = 0.001
        X = sparse.random(m, n, density=density, data_rvs=np.random.randn)
        y = np.random.randn(m)

        # Convert problem to standard form.
        # f_1(\beta_1) = ||y - X\beta_1||_2^2, f_2(\beta_2) = I(\beta_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [prox_sum_squares(X, y), lambda v, t: v.maximum(0) if sparse.issparse(v) else np.maximum(v,0)]
        A_list = [sparse.eye(n), -sparse.eye(n)]
        b = np.zeros(n)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finish DRS.')
    
        # Solve with A2DR.
        t0 = time.time()
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        t1 = time.time()
        a2dr_beta = a2dr_result["x_vals"][-1]
        print('nonzero entries proportion = {}'.format(np.sum(a2dr_beta > 0)*1.0/len(a2dr_beta)))
        print('Finish A2DR.')
        self.compare_total(drs_result, a2dr_result, figname)
        
        # Check solution correctness.
        print('run time of A2DR = {}'.format(t1-t0))
        print('constraint violation of A2DR = {}'.format(np.min(a2dr_beta)))
        print('objective value of A2DR = {}'.format(np.linalg.norm(X.dot(a2dr_beta)-y)))

    def test_nnls_reg(self, figname):
        # minimize ||y - X\beta||_2^2 subject to \beta >= 0.

        # Problem data.
        m, n = 300, 500
        density = 0.001
        X = sparse.random(m, n, density=density, data_rvs=np.random.randn)
        y = np.random.randn(m)

        # Convert problem to standard form.
        # f_1(\beta_1) = ||y - X\beta_1||_2^2, f_2(\beta_2) = I(\beta_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [prox_sum_squares(X, y), lambda v, t: v.maximum(0) if sparse.issparse(v) else np.maximum(v,0)]
        A_list = [sparse.eye(n), -sparse.eye(n)]
        b = np.zeros(n)

        # Solve with no regularization.
        a2dr_noreg_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, lam_accel=0, max_iter=self.MAX_ITER)
        print('Finish A2DR no regularization.')
        
        # Solve with constant regularization.
        a2dr_consreg_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, ada_reg=False, max_iter=self.MAX_ITER)
        print('Finish A2DR constant regularization.')
        
        # Solve with adaptive regularization.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, ada_reg=True, max_iter=self.MAX_ITER) 
        print('Finish A2DR adaptive regularization.')
        
        self.compare_total_all([a2dr_noreg_result, a2dr_consreg_result, a2dr_result], 
                               ['no-reg', 'constant-reg', 'ada-reg'], figname)

    def test_sparse_inv_covariance(self, n, alpha_ratio, figname):
        # minimize -log(det(S)) + trace(S*Y) + \alpha*||S||_1 subject to S is symmetric PSD.

        # Problem data.
        # n: Dimension of matrix.
        m = 1000  # Number of samples.
        ratio = 0.9   # Fraction of zeros in S.

        S_true = sparse.csc_matrix(make_sparse_spd_matrix(n, ratio))
        R = sparse.linalg.inv(S_true).todense()
        q_sample = sp.linalg.sqrtm(R).dot(np.random.randn(n,m))
        Q = np.cov(q_sample)
        mask = np.ones(Q.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        alpha_max = np.max(np.abs(Q)[mask])
        alpha = alpha_ratio*alpha_max #0.001 for n=100, 0.01 for n=50

        # Convert problem to standard form.
        # f_1(S) = -log(det(S)) on symmetric PSD matrices, f_2(S) = trace(S*Q), f_3(S) = \alpha*||S||_1.
        # A_1 = [I; 0], A_2 = [-I; I], A_3 = [0; -I], b = 0.
        prox_list = [lambda v, t: prox_neg_log_det_aff(v.reshape((n,n), order='C'), Q, t, order='C'),
                     prox_norm1(alpha)]
        A_list = [sparse.eye(n*n), -sparse.eye(n*n)]
        b = np.zeros(n*n)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER) 
        #lam_accel=0 seems to work well sometimes, although oscillating very much
        a2dr_S = a2dr_result["x_vals"][-1].reshape((n,n), order='C')
        self.compare_total(drs_result, a2dr_result, figname)
        print('Finished A2DR.')
        print('recovered sparsity = {}'.format(np.sum(a2dr_S!=0)*1.0/a2dr_S.shape[0]**2))

    def test_l1_trend_filtering(self, figname):
        # minimize (1/2)||y - x||_2^2 + \alpha*||Dx||_1,
        # where (Dx)_{t-1} = x_{t-1} - 2*x_t + x_{t+1} for t = 2,...,n-1.
        # Reference: https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

        # Problem data.
        n = int(2*10**4)
        y = np.random.randn(n)
        alpha = 0.01*np.linalg.norm(y, np.inf)

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
        b = np.zeros(n-2)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result, figname)
        print('Finished A2DR.')

    def test_optimal_control(self, figname):
        # Problem data/
        m = 80
        n = 150
        K = 20
        A = np.random.randn(n,n)
        B = np.random.randn(n,m)
        c = np.random.randn(n)
        x_init = np.random.randn(n)
        A = A / np.max(np.abs(LA.eigvals(A)))
        xhat = x_init
        for k in range(K-1):
            uhat = np.random.randn(m)
            uhat = uhat / np.max(np.abs(uhat))
            xhat = A.dot(xhat) + B.dot(uhat) + c
        x_term = xhat
        # uhat no normalization actually leads to more significant improvement of A2DR over DRS, and also happens to be feasible
        # x_term = 0 also happens to be feasible
        
        # Convert problem to standard form.
        prox_list = [prox_square, prox_sat(1,1)]
        A1 = sparse.lil_matrix(((K+1)*n,K*n))
        A1[n:K*n,:(K-1)*n] = -sparse.block_diag((K-1)*[A])
        A1.setdiag(1)
        A1[K*n:,(K-1)*n:] = sparse.eye(n)
        A2 = sparse.lil_matrix(((K+1)*n,K*m))
        A2[n:K*n,:(K-1)*m] = -sparse.block_diag((K-1)*[B])
        A_list = [sparse.csr_matrix(A1), sparse.csr_matrix(A2)]
        b_list = [x_init]
        b_list.extend((K-1)*[c])
        b_list.extend([x_term])
        b = np.concatenate(b_list)
        
        # Solve with CVXPY
        x = Variable((K,n))
        u = Variable((K,m))
        obj = sum([sum_squares(x[k]) + sum_squares(u[k]) for k in range(K)])
        constr = [x[0] == x_init, norm_inf(u) <= 1]
        constr += [x[k+1] == A*x[k] + B*u[k] + c for k in range(K-1)]
        constr += [x[K-1] == x_term]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver='SCS', verbose=True) 
        # OSQP fails for m=50, n=100, K=30, and also for m=100, n=200, K=30
        # SCS also fails to converge
        cvxpy_obj = prob.value
        cvxpy_x = x.value.ravel(order='C')
        cvxpy_u = u.value.ravel(order='C')
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result, figname)
        print('Finished A2DR.')
        
        # check solution correctness
        a2dr_x = a2dr_result['x_vals'][0]
        a2dr_u = a2dr_result['x_vals'][1]
        a2dr_obj = np.sum(a2dr_x**2) + np.sum(a2dr_u**2)
        cvxpy_obj_raw = np.sum(cvxpy_x**2) + np.sum(cvxpy_u**2)
        cvxpy_X = cvxpy_x.reshape([K,n], order='C')
        cvxpy_U = cvxpy_u.reshape([K,m], order='C')
        a2dr_X = a2dr_x.reshape([K,n], order='C')
        a2dr_U = a2dr_u.reshape([K,m], order='C')
        cvxpy_constr_vio = [np.linalg.norm(cvxpy_X[0]-x_init), np.linalg.norm(cvxpy_X[K-1]-x_term)]
        a2dr_constr_vio = [np.linalg.norm(a2dr_X[0]-x_init), np.linalg.norm(a2dr_X[K-1]-x_term)]
        for k in range(K-1):
            cvxpy_constr_vio.append(np.linalg.norm(cvxpy_X[k+1]-A.dot(cvxpy_X[k])-B.dot(cvxpy_U[k])-c))
            a2dr_constr_vio.append(np.linalg.norm(a2dr_X[k+1]-A.dot(a2dr_X[k])-B.dot(a2dr_U[k])-c))    
        print('linear constr vio cvxpy = {}, linear constr_vio a2dr = {}'.format(
            np.mean(cvxpy_constr_vio), np.mean(a2dr_constr_vio)))
        print('norm constr vio cvxpy = {}, norm constr vio a2dr = {}'.format(np.max(np.abs(cvxpy_u)), 
                                                                             np.max(np.abs(a2dr_u))))
        print('objective cvxpy = {}, objective cvxpy raw = {}, objective a2dr = {}'.format(cvxpy_obj, 
                                                                                           cvxpy_obj_raw,
                                                                                           a2dr_obj))

        
    def test_coupled_qp(self, figname):
        # Problem data.
        K = 8 # number of blocks
        p = 50 # number of coupling constraints
        nk = 300 # variable dimension of each subproblem QP
        mk = 200 # constrain dimension of each subproblem QP
        A_list = [np.random.randn(p, nk) for k in range(K)]
        F_list = [np.random.randn(mk, nk) for k in range(K)]
        q_list = [np.random.randn(nk) for k in range(K)]
        x_list = [np.random.randn(nk) for k in range(K)]
        g_list = [F_list[k].dot(x_list[k])+0.1 for k in range(K)]
        A = np.hstack(A_list)
        x = np.hstack(x_list)
        b = A.dot(x)
        P_list = [np.random.randn(nk,nk) for k in range(K)]
        Q_list = [P_list[k].T.dot(P_list[k]) for k in range(K)]
        
        # Convert problem to standard form.
        def tmp(k, Q_list, q_list, F_list, g_list):
            return prox_qp(Q_list[k], q_list[k], F_list[k], g_list[k])
            
        prox_list = list(map(lambda k: tmp(k,Q_list,q_list,F_list,g_list), range(K)))
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result, figname)
        
        # Check solution correctness.
        a2dr_x = a2dr_result['x_vals']
        a2dr_obj = np.sum([a2dr_x[k].dot(Q_list[k]).dot(a2dr_x[k]) 
                           + q_list[k].dot(a2dr_x[k]) for k in range(K)])
        a2dr_constr_vio = [np.linalg.norm(np.maximum(F_list[k].dot(a2dr_x[k])-g_list[k],0))**2 
                                  for k in range(K)]
        a2dr_constr_vio += [np.linalg.norm(A.dot(np.hstack(a2dr_x))-b)**2]
        a2dr_constr_vio_val = np.sqrt(np.sum(a2dr_constr_vio))
        print('objective value of A2DR = {}'.format(a2dr_obj))
        print('constraint violation of A2DR = {}'.format(a2dr_constr_vio_val))

    def test_commodity_flow(self, figname):
        # Problem data.
        m = 4000  # Number of sources.
        n = 7000  # Number of flows.

        # Construct a random incidence matrix.
        B = sparse.lil_matrix((m,n))
        for i in range(n-m+1):
            idxs = np.random.choice(m, size=2, replace=False)
            tmp = np.random.rand()
            if tmp > 0.5:
                B[idxs[0],i] = 1
                B[idxs[1],i] = -1
            else:
                B[idxs[0],i] = -1
                B[idxs[1],i] = 1
        for j in range(n-m+1,n):
            B[j-(n-m+1),j] = 1
            B[j-(n-m),j] = -1 
        B = sparse.csr_matrix(B)
        
        # Generate source and flow range data
        s_tilde = np.random.randn(m)
        m1, m2, m3 = int(m/3), int(m/3*2), int(m/6*5)
        s_tilde[:m1] = 0
        s_tilde[m1:m2] = -np.abs(s_tilde[m1:m2])
        s_tilde[m2:] = np.sum(np.abs(s_tilde[m1:m2])) / (m-m2)
        L = s_tilde[m1:m2]
        s_max = np.hstack([s_tilde[m2:m3]+0.001, (s_tilde[m3:]+0.001)*2])
        res = sparse.linalg.lsqr(B, -s_tilde, atol=1e-16, btol=1e-16)
        x_tilde = res[0]
        n1 = int(n/2)
        x_max = np.abs(x_tilde)+0.001
        x_max[n1:] = x_max[n1:]*2
        
        # Generate cost coefficients
        c = np.random.rand(n)
        d = np.random.rand(m)
        
        # Solve by CVXPY
        x = Variable(n)
        s = Variable(m)
        C = sparse.diags(c)
        D = sparse.diags(d)
        obj = quad_form(x, C) + quad_form(s, D)
        constr = [-x_max<=x, x<=x_max, s[:m1]==0, s[m1:m2]==L, 0<=s[m2:], s[m2:]<=s_max, B*x+s==0]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver='SCS', verbose=True) #'OSQP'
        cvxpy_x = x.value
        cvxpy_s = s.value

        # Convert problem to standard form.
        # f_1(x) = \sum_j c_j*x_j^2 + I(-x_max <= x_j <= x_max),
        # f_2(s) = \sum_i d_i*s_i^(source)^2 + I(0 <= s_i^(source) <= s_max) 
        #          + \sum_{i'} I(s_{i'}^{transfer}=0) + \sum_{i''}
        #          + \sum_{i"} I(s_{i"}^{sink}=L_{i"}).
        # A = [B, I], b = 0
        z = np.zeros(m1)
        prox_list = [prox_sat(c, x_max), lambda v, t: np.hstack([z, L, prox_sat_pos(d[m2:], s_max)(v[m2:],t)])]
        A_list = [B, sparse.eye(m)]
        b = np.zeros(m)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=2*self.MAX_ITER)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=2*self.MAX_ITER)
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result, figname)
        
        # Check solution correctness
        a2dr_x = a2dr_result['x_vals'][0]
        a2dr_s = a2dr_result['x_vals'][1]
        cvxpy_obj_raw = np.sum(c*cvxpy_x**2) + np.sum(d*cvxpy_s**2)
        a2dr_obj = np.sum(c*a2dr_x**2) + np.sum(d*a2dr_s**2)
        cvxpy_constr_vio = [np.maximum(np.abs(cvxpy_x) - x_max, 0), 
                            cvxpy_s[:m1], 
                            np.abs(cvxpy_s[m1:m2]-L), 
                            np.maximum(-cvxpy_s[m2:],0),
                            np.maximum(cvxpy_s[m2:]-s_max,0),
                            B.dot(cvxpy_x)+cvxpy_s]
        cvxpy_constr_vio_val = np.linalg.norm(np.hstack(cvxpy_constr_vio))
        a2dr_constr_vio = [np.maximum(np.abs(a2dr_x) - x_max, 0), 
                            a2dr_s[:m1], 
                            np.abs(a2dr_s[m1:m2]-L), 
                            np.maximum(-a2dr_s[m2:],0),
                            np.maximum(a2dr_s[m2:]-s_max,0),
                            B.dot(a2dr_x)+a2dr_s]
        a2dr_constr_vio_val = np.linalg.norm(np.hstack(a2dr_constr_vio))
        print('objective cvxpy raw = {}, objective a2dr = {}'.format(cvxpy_obj_raw, a2dr_obj))
        print('constraint violation cvxpy = {}, constraint violation a2dr = {}'.format(
            cvxpy_constr_vio_val, a2dr_constr_vio_val))

    def test_multi_task_logistic(self, figname):
        # minimize \sum_{ik} log(1 + exp(-Y_{ik}*Z_{ik})) + \alpha*||\theta||_{2,1} + \beta*||\theta||_*
        # subject to Z = X\theta, ||.||_{2,1} = group lasso, ||.||_* = nuclear norm.

        # Problem data.
        K = 10 # Number of tasks.
        p = 500 # Number of features.
        m = 300 # Number of samples.
        alpha = 0.1
        beta = 0.1

        X = np.random.randn(m,p)
        theta_true = np.random.randn(p,K)
        Z_true = X.dot(theta_true)
        Y = 2*(Z_true > 0) - 1   # Y_{ij} = 1 or -1.

        def calc_obj(theta):
            obj = np.sum(-np.log(sp.special.expit(np.multiply(Y, X.dot(theta)))))
            reg = alpha*np.sum([LA.norm(theta[:,k], 2) for k in range(K)])
            reg += beta*LA.norm(theta, ord='nuc')
            return obj + reg

        # Convert problem to standard form. 
        # f_1(Z) = \sum_{ik} log(1 + exp(-Y_{ik}*Z_{ik})), 
        # f_2(\theta) = \alpha*||\theta||_{2,1}, 
        # f_3(\tilde \theta) = \beta*||\tilde \theta||_*.
        # A_1 = [I; 0], A_2 = [-X; I], A_3 = [0; -I], b = 0.
        prox_list = [lambda v, t: prox_logistic(v, 1.0/t, y = Y.ravel(order='F')),   
                     # TODO: Calculate in parallel for k = 1,...K.
                     lambda v, t: prox_group_lasso(alpha)(v.reshape((p,K), order='F'), t),
                     lambda v, t: prox_nuc_norm(beta, order='F')(v.reshape((p,K), order='F'), t)]
        A_list = [sparse.vstack([sparse.eye(m*K), sparse.csr_matrix((p*K,m*K))]),
                  sparse.vstack([-sparse.block_diag(K*[X]), sparse.eye(p*K)]),
                  sparse.vstack([sparse.csr_matrix((m*K,p*K)), -sparse.eye(p*K)])]
        b = np.zeros(m*K + p*K)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        a2dr_theta = a2dr_result["x_vals"][-1].reshape((p,K), order='F')
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result, figname)
