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
from a2dr.proximal import *
from a2dr.tests.base_test import BaseTest

class TestPaper(BaseTest):
    """Reproducible tests and plots for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-8   # specify these in all examples?
        self.eps_abs = 1e-6
        self.MAX_ITER = 1000

    def test_nnls(self, figname):
        # minimize ||Fz - g||_2^2 subject to z >= 0.

        # Problem data.
        p, q = 10000, 8000
        density = 0.001
        F = sparse.random(p, q, density=density, data_rvs=np.random.randn)
        g = np.random.randn(p)

        # Convert problem to standard form.
        # f_1(x_1) = ||Fx_1 - g||_2^2, f_2(x_2) = I(x_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F, g), prox_nonneg_constr]
        A_list = [sparse.eye(q), -sparse.eye(q)]
        b = np.zeros(q)
        
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
        print('objective value of A2DR = {}'.format(np.linalg.norm(F.dot(a2dr_beta)-g)))

    def test_nnls_reg(self, figname):
        # minimize ||Fz - g||_2^2 subject to z >= 0.

        # Problem data.
        p, q = 300, 500
        density = 0.001
        F = sparse.random(p, q, density=density, data_rvs=np.random.randn)
        g = np.random.randn(p)

        # Convert problem to standard form.
        # f_1(x_1) = ||Fx_1 - g||_2^2, f_2(x_2) = I(x_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F, g), prox_nonneg_constr]
        A_list = [sparse.eye(q), -sparse.eye(q)]
        b = np.zeros(q)

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

    def test_sparse_inv_covariance(self, q, alpha_ratio, figname):
        # minimize -log(det(S)) + trace(S*Q) + \alpha*||S||_1 subject to S is symmetric PSD.

        # Problem data.
        # q: Dimension of matrix.
        p = 1000      # Number of samples.
        ratio = 0.9   # Fraction of zeros in S.

        S_true = sparse.csc_matrix(make_sparse_spd_matrix(q, ratio))
        Sigma = sparse.linalg.inv(S_true).todense()
        z_sample = np.real(sp.linalg.sqrtm(Sigma)).dot(np.random.randn(q,p))   # make sure it's real matrices.
        Q = np.cov(z_sample)
        print('Q is positive definite? {}'.format(bool(LA.slogdet(Q)[0])))
        
        mask = np.ones(Q.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        alpha_max = np.max(np.abs(Q)[mask])
        alpha = alpha_ratio*alpha_max   # 0.001 for q = 100, 0.01 for q = 50.

        # Convert problem to standard form.
        # f_1(S_1) = -log(det(S_1)) + trace(S_1*Q) on symmetric PSD matrices, f_2(S_2) = \alpha*||S_2||_1.
        # A_1 = I, A_2 = -I, b = 0.
        prox_list = [lambda v, t: prox_neg_log_det(v.reshape((q,q), order='C'), t, lin_term=t*Q).ravel(order='C'), 
                     lambda v, t: prox_norm1(v, t*alpha)]
        A_list = [sparse.eye(q*q), -sparse.eye(q*q)]
        b = np.zeros(q*q)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER) 
        # lam_accel = 0 seems to work well sometimes, although it oscillates a lot.
        a2dr_S = a2dr_result["x_vals"][-1].reshape((q,q), order='C')
        self.compare_total(drs_result, a2dr_result, figname)
        print('Finished A2DR.')
        print('recovered sparsity = {}'.format(np.sum(a2dr_S != 0)*1.0/a2dr_S.shape[0]**2))
        
    def test_l1_trend_filtering(self, figname):
        # minimize (1/2)||y - z||_2^2 + \alpha*||Dz||_1,
        # where (Dz)_{t-1} = z_{t-1} - 2*z_t + z_{t+1} for t = 2,...,q-1.
        # Reference: https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

        # Problem data.
        q = int(2*10**4)
        y = np.random.randn(q)
        alpha = 0.01*np.linalg.norm(y, np.inf)

        # Form second difference matrix.
        D = sparse.lil_matrix(sparse.eye(q))
        D.setdiag(-2, k = 1)
        D.setdiag(1, k = 2)
        D = D[:(q-2),:]

        # Convert problem to standard form.
        # f_1(x_1) = (1/2)||y - x_1||_2^2, f_2(x_2) = \alpha*||x_2||_1.
        # A_1 = D, A_2 = -I_{n-2}, b = 0.
        prox_list = [lambda v, t: prox_sum_squares(v, t = 0.5*t, offset = y),
                     lambda v, t: prox_norm1(v, t = alpha*t)]
        A_list = [D, -sparse.eye(q-2)]
        b = np.zeros(q-2)

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result, figname)
        print('Finished A2DR.')

    def test_optimal_control(self, figname):
        # Problem data/
        p = 80
        q = 150
        L = 20
        
        F = np.random.randn(q,q)
        G = np.random.randn(q,p)
        h = np.random.randn(q)
        z_init = np.random.randn(q)
        F = F / np.max(np.abs(LA.eigvals(F)))
        
        z_hat = z_init
        for l in range(L-1):
            u_hat = np.random.randn(p)
            u_hat = u_hat / np.max(np.abs(u_hat))
            z_hat = F.dot(z_hat) + G.dot(u_hat) + h
        z_term = z_hat
        # no normalization of u_hat actually leads to more significant improvement of A2DR over DRS, and also happens to be feasible
        # x_term = 0 also happens to be feasible
        
        # Convert problem to standard form.
        def prox_sat(v, t, v_lo = -np.inf, v_hi = np.inf):
            return prox_box_constr(prox_sum_squares(v, t), t, v_lo, v_hi)
        prox_list = [prox_sum_squares, lambda v, t: prox_sat(v, t, -1, 1)]
        A1 = sparse.lil_matrix(((L+1)*q,L*q))
        A1[q:L*q,:(L-1)*q] = -sparse.block_diag((L-1)*[F])
        A1.setdiag(1)
        A1[L*q:,(L-1)*q:] = sparse.eye(q)
        A2 = sparse.lil_matrix(((L+1)*q,L*p))
        A2[q:L*q,:(L-1)*p] = -sparse.block_diag((L-1)*[G])
        A_list = [sparse.csr_matrix(A1), sparse.csr_matrix(A2)]
        b_list = [z_init]
        b_list.extend((L-1)*[h])
        b_list.extend([z_term])
        b = np.concatenate(b_list)
        
        # Solve with CVXPY
        z = Variable((L,q))
        u = Variable((L,p))
        obj = sum([sum_squares(z[l]) + sum_squares(u[l]) for l in range(L)])
        constr = [z[0] == z_init, norm_inf(u) <= 1]
        constr += [z[l+1] == F*z[l] + G*u[l] + h for l in range(L-1)]
        constr += [z[L-1] == z_term]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver='SCS', verbose=True) 
        # OSQP fails for p=50, q=100, L=30, and also for p=100, q=200, L=30
        # SCS also fails to converge
        cvxpy_obj = prob.value
        cvxpy_z = z.value.ravel(order='C')
        cvxpy_u = u.value.ravel(order='C')
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('Finished DRS.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        self.compare_total(drs_result, a2dr_result, figname)
        print('Finished A2DR.')
        
        # check solution correctness
        a2dr_z = a2dr_result['x_vals'][0]
        a2dr_u = a2dr_result['x_vals'][1]
        a2dr_obj = np.sum(a2dr_z**2) + np.sum(a2dr_u**2)
        cvxpy_obj_raw = np.sum(cvxpy_z**2) + np.sum(cvxpy_u**2)
        cvxpy_Z = cvxpy_z.reshape([L,q], order='C')
        cvxpy_U = cvxpy_u.reshape([L,p], order='C')
        a2dr_Z = a2dr_z.reshape([L,q], order='C')
        a2dr_U = a2dr_u.reshape([L,p], order='C')
        cvxpy_constr_vio = [np.linalg.norm(cvxpy_Z[0]-z_init), np.linalg.norm(cvxpy_Z[L-1]-z_term)]
        a2dr_constr_vio = [np.linalg.norm(a2dr_Z[0]-z_init), np.linalg.norm(a2dr_Z[L-1]-z_term)]
        for l in range(L-1):
            cvxpy_constr_vio.append(np.linalg.norm(cvxpy_Z[l+1]-F.dot(cvxpy_Z[l])-G.dot(cvxpy_U[l])-h))
            a2dr_constr_vio.append(np.linalg.norm(a2dr_Z[l+1]-F.dot(a2dr_Z[l])-G.dot(a2dr_U[l])-h))
        print('linear constr vio cvxpy = {}, linear constr_vio a2dr = {}'.format(
            np.mean(cvxpy_constr_vio), np.mean(a2dr_constr_vio)))
        print('norm constr vio cvxpy = {}, norm constr vio a2dr = {}'.format(np.max(np.abs(cvxpy_u)), 
                                                                             np.max(np.abs(a2dr_u))))
        print('objective cvxpy = {}, objective cvxpy raw = {}, objective a2dr = {}'.format(cvxpy_obj, 
                                                                                           cvxpy_obj_raw,
                                                                                           a2dr_obj))
        
    def test_coupled_qp(self, figname):
        # Problem data.
        L = 8      # number of blocks
        s = 50     # number of coupling constraints
        ql = 300   # variable dimension of each subproblem QP
        pl = 200   # constraint dimension of each subproblem QP
        
        G_list = [np.random.randn(s,ql) for l in range(L)]
        F_list = [np.random.randn(pl,ql) for l in range(L)]
        c_list = [np.random.randn(ql) for l in range(L)]
        z_tld_list = [np.random.randn(ql) for l in range(L)]
        d_list = [F_list[l].dot(z_tld_list[l])+0.1 for l in range(L)]
        
        G = np.hstack(G_list)
        z_tld = np.hstack(z_tld_list)
        h = G.dot(z_tld)
        H_list = [np.random.randn(ql,ql) for l in range(L)]
        Q_list = [H_list[l].T.dot(H_list[l]) for l in range(L)]
        
        # Convert problem to standard form.
        def tmp(l, Q_list, c_list, F_list, d_list):
            return lambda v, t: prox_qp(v, t, Q_list[l], c_list[l], F_list[l], d_list[l])
        # Use "map" method to avoid implicit overriding, which would make all the proximal operators the same
        prox_list = list(map(lambda l: tmp(l, Q_list, c_list, F_list, d_list), range(L)))
        A_list = G_list
        b = h
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result, figname)
        
        # Check solution correctness.
        a2dr_z = a2dr_result['x_vals']
        a2dr_obj = np.sum([a2dr_z[l].dot(Q_list[l]).dot(a2dr_z[l]) 
                           + c_list[l].dot(a2dr_z[l]) for l in range(L)])
        a2dr_constr_vio = [np.linalg.norm(np.maximum(F_list[l].dot(a2dr_z[l])-d_list[l],0))**2 
                                  for l in range(L)]
        a2dr_constr_vio += [np.linalg.norm(G.dot(np.hstack(a2dr_z))-h)**2]
        a2dr_constr_vio_val = np.sqrt(np.sum(a2dr_constr_vio))
        print('objective value of A2DR = {}'.format(a2dr_obj))
        print('constraint violation of A2DR = {}'.format(a2dr_constr_vio_val))

    def test_commodity_flow(self, figname):
        # Problem data.
        p = 4000   # Number of sources.
        q = 7000   # Number of flows.

        # Construct a random incidence matrix.
        B = sparse.lil_matrix((p,q))
        for i in range(q-p+1):
            idxs = np.random.choice(p, size=2, replace=False)
            tmp = np.random.rand()
            if tmp > 0.5:
                B[idxs[0],i] = 1
                B[idxs[1],i] = -1
            else:
                B[idxs[0],i] = -1
                B[idxs[1],i] = 1
        for j in range(q-p+1,q):
            B[j-(q-p+1),j] = 1
            B[j-(q-p),j] = -1 
        B = sparse.csr_matrix(B)
        
        # Generate source and flow range data
        s_tilde = np.random.randn(p)
        p1, p2, p3 = int(p/3), int(p/3*2), int(p/6*5)
        s_tilde[:p1] = 0
        s_tilde[p1:p2] = -np.abs(s_tilde[p1:p2])
        s_tilde[p2:] = np.sum(np.abs(s_tilde[p1:p2])) / (p-p2)
        L = s_tilde[p1:p2]
        s_max = np.hstack([s_tilde[p2:p3]+0.001, 2*(s_tilde[p3:]+0.001)])
        res = sparse.linalg.lsqr(B, -s_tilde, atol=1e-16, btol=1e-16)
        z_tilde = res[0]
        q1 = int(q/2)
        z_max = np.abs(z_tilde)+0.001
        z_max[q1:] = 2*z_max[q1:]
        
        # Generate cost coefficients
        c = np.random.rand(q)
        d = np.random.rand(p)
        
        # Solve by CVXPY
        z = Variable(q)
        s = Variable(p)
        C = sparse.diags(c)
        D = sparse.diags(d)
        obj = quad_form(z, C) + quad_form(s, D)
        constr = [-z_max<=z, z<=z_max, s[:p1]==0, s[p1:p2]==L, 0<=s[p2:], s[p2:]<=s_max, B*z+s==0]
        prob = Problem(Minimize(obj), constr)
        prob.solve(solver='SCS', verbose=True)   # 'OSQP'
        cvxpy_z = z.value
        cvxpy_s = s.value

        # Convert problem to standard form.
        # f_1(z) = \sum_j c_j*z_j^2 + I(-z_max <= z_j <= z_max),
        # f_2(s) = \sum_i d_i*s_i^(source)^2 + I(0 <= s_i^(source) <= s_max) 
        #          + \sum_{i'} I(s_{i'}^{transfer}=0) + \sum_{i''}
        #          + \sum_{i"} I(s_{i"}^{sink}=L_{i"}).
        # A = [B, I], b = 0
        zeros = np.zeros(p1)
        def prox_sat(v, t, c, v_lo = -np.inf, v_hi = np.inf):
            return prox_box_constr(prox_sum_squares(v, t*c), t, v_lo, v_hi)
        prox_list = [lambda v, t: prox_sat(v, t, c, -z_max, z_max),
                     lambda v, t: np.hstack([zeros, L, prox_sat(v[p2:], t, d[p2:], 0, s_max)])]
        A_list = [B, sparse.eye(p)]
        b = np.zeros(p)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=2*self.MAX_ITER)
        print('DRS finished.')
        
        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=2*self.MAX_ITER)
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result, figname)
        
        # Check solution correctness
        a2dr_z = a2dr_result['x_vals'][0]
        a2dr_s = a2dr_result['x_vals'][1]
        cvxpy_obj_raw = np.sum(c*cvxpy_z**2) + np.sum(d*cvxpy_s**2)
        a2dr_obj = np.sum(c*a2dr_z**2) + np.sum(d*a2dr_s**2)
        cvxpy_constr_vio = [np.maximum(np.abs(cvxpy_z) - z_max, 0), 
                            cvxpy_s[:p1], 
                            np.abs(cvxpy_s[p1:p2]-L), 
                            np.maximum(-cvxpy_s[p2:],0),
                            np.maximum(cvxpy_s[p2:]-s_max,0),
                            B.dot(cvxpy_z)+cvxpy_s]
        cvxpy_constr_vio_val = np.linalg.norm(np.hstack(cvxpy_constr_vio))
        a2dr_constr_vio = [np.maximum(np.abs(a2dr_z) - z_max, 0), 
                            a2dr_s[:p1], 
                            np.abs(a2dr_s[p1:p2]-L), 
                            np.maximum(-a2dr_s[p2:],0),
                            np.maximum(a2dr_s[p2:]-s_max,0),
                            B.dot(a2dr_z)+a2dr_s]
        a2dr_constr_vio_val = np.linalg.norm(np.hstack(a2dr_constr_vio))
        print('objective cvxpy raw = {}, objective a2dr = {}'.format(cvxpy_obj_raw, a2dr_obj))
        print('constraint violation cvxpy = {}, constraint violation a2dr = {}'.format(
            cvxpy_constr_vio_val, a2dr_constr_vio_val))

    def test_multi_task_logistic(self, figname):
        # minimize \sum_{il} log(1 + exp(-Y_{il}*Z_{il})) + \alpha*||\theta||_{2,1} + \beta*||\theta||_*
        # subject to Z = W\theta, ||.||_{2,1} = group lasso, ||.||_* = nuclear norm.

        # Problem data.
        L = 10    # Number of tasks.
        s = 500   # Number of features.
        p = 300   # Number of samples.
        alpha = 0.1
        beta = 0.1

        W = np.random.randn(p,s)
        theta_true = np.random.randn(s,L)
        Z_true = W.dot(theta_true)
        Y = 2*(Z_true > 0) - 1   # Y_{ij} = 1 or -1.

        def calc_obj(theta):
            obj = np.sum(-np.log(sp.special.expit(np.multiply(Y, W.dot(theta)))))
            reg = alpha*np.sum([LA.norm(theta[:,l], 2) for l in range(L)])
            reg += beta*LA.norm(theta, ord='nuc')
            return obj + reg

        # Convert problem to standard form. 
        # f_1(Z) = \sum_{il} log(1 + exp(-Y_{il}*Z_{il})), 
        # f_2(\theta) = \alpha*||\theta||_{2,1}, 
        # f_3(\tilde \theta) = \beta*||\tilde \theta||_*.
        # A_1 = [I; 0], A_2 = [-W; I], A_3 = [0; -I], b = 0.
        prox_list = [lambda v, t: prox_logistic(v, t, y = Y.ravel(order='F')),   
                     # TODO: Calculate in parallel for l = 1,...L.
                     lambda v, t: prox_group_lasso(v.reshape((s,L), order='F'), t*alpha).ravel(order='F'),
                     lambda v, t: prox_norm_nuc(v.reshape((s,L), order='F'), t*beta).ravel(order='F')]
        A_list = [sparse.vstack([sparse.eye(p*L), sparse.csr_matrix((s*L,p*L))]),
                  sparse.vstack([-sparse.block_diag(L*[W]), sparse.eye(s*L)]),
                  sparse.vstack([sparse.csr_matrix((p*L,s*L)), -sparse.eye(s*L)])]
        b = np.zeros(p*L + s*L)
        
        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER)
        print('DRS finished.')

        # Solve with A2DR.
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER)
        a2dr_theta = a2dr_result["x_vals"][-1].reshape((s,L), order='F')
        print('A2DR finished.')
        self.compare_total(drs_result, a2dr_result, figname)
