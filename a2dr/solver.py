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
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.stats.mstats import gmean
from time import time
from multiprocessing import Process, Pipe
from a2dr.precondition import precondition
from a2dr.acceleration import aa_weights

def a2dr_worker(pipe, prox, v_init, A, t, anderson, m_accel):
    # Initialize AA-II parameters.
    if anderson:   # TODO: Store and update these efficiently as arrays.
        F_hist = []  # History of F(v^(k)).
    v_vec = v_init.copy()
    v_res = np.zeros(v_init.shape[0])

    # A2DR loop.
    while True:
        # Proximal step for x^(k+1/2).
        x_half = prox(v_vec, t)

        # Calculate v^(k+1/2) = 2*x^(k+1/2) - v^(k).
        v_half = 2*x_half - v_vec

        # Project to obtain x^(k+1) = v^(k+1/2) - A^T(AA^T)^{-1}(Av^(k+1/2) - b).
        pipe.send(v_half)
        dk, k = pipe.recv()   # dk = A^\dagger(Av^(k+1/2) - b)[i] for node i.
        x_new = v_half - dk

        if anderson and k > 0: # for k = 0, always do the vanilla DRS update
            m_k = min(m_accel, k)  # Keep F(v^(j)) for iterations (k-m_k) through k.

            # Save history of F(v^(k)).
            F_hist.append(v_vec + x_new - x_half)
            if len(F_hist) > m_k + 1:
                F_hist.pop(0)

            # Send s^(k-1) = v^(k) - v^(k-1) and g^(k) = v^(k) - F(v^(k)) = x^(k+1/2) - x^(k+1).
            pipe.send((v_res, x_half - x_new))

            # Receive safeguarding decision.
            AA_update = pipe.recv()
            if AA_update:
                # Receive AA-II weights for v^(k+1).
                alpha = pipe.recv()

                # Weighted update of v^(k+1).
                v_new = np.column_stack(F_hist).dot(alpha) #.dot(alpha[:(k + 1)]) ### Why truncate to (k+1)???
            else:
                # Revert to DRS update of v^(k+1).
                v_new = v_vec + x_new - x_half

            # Save v^(k+1) - v^(k) for next iteration.
            v_res = v_new - v_vec
        elif anderson and k == 0: 
            # Update v^(k+1) = v^(k) + x^(k+1) - x^(k+1/2).
            v_new = v_vec + x_new - x_half
            ## only useful when anderson = True but k == 0
            # Store v_res in case anderson = True
            v_res = v_new - v_vec
            # Update F_hist in case anderson = True
            F_hist.append(v_vec + x_new - x_half)
            # Send g^(k) = v^(k) - F(v^(k)) = x^(k+1/2) - x^(k+1).
            pipe.send(x_half - x_new)
        else:
            # Update v^(k+1) = v^(k) + x^(k+1) - x^(k+1/2).
            v_new = v_vec + x_new - x_half

        # Send x^(k+1/2) along with A*x^(k+1/2) and x^(k+1/2) - v^(k) for computing residuals.
        
        #pipe.send((x_half, A.dot(x_half), x_half - v_vec))
        Ax_half = A.dot(x_half)
        pipe.send((x_half, Ax_half, x_half - v_vec))
        
        v_vec = v_new


def a2dr(p_list, A_list = [], b = np.array([]), v_init = None, n_list = None, *args, **kwargs):
    # Problem parameters.
    max_iter = kwargs.pop("max_iter", 1000)
    t_init = kwargs.pop("t_init", 1/10)  # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)   # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-8)   # Relative stopping tolerance.
    precond = kwargs.pop("precond", True)  # Precondition A and b?
    ada_reg = kwargs.pop("ada_reg", True)   # Adaptive regularization?

    # AA-II parameters.
    anderson = kwargs.pop("anderson", True)
    m_accel = int(kwargs.pop("m_accel", 10))       # Maximum past iterations to keep (>= 0).
    lam_accel = kwargs.pop("lam_accel", 1e-8)      # AA-II regularization weight.
    aa_method = kwargs.pop("aa_method", "lstsq")   # Algorithm for solving AA LS problem.

    # Safeguarding parameters.
    D_safe = kwargs.pop("D_safe", 1e6)
    eps_safe = kwargs.pop("eps_safe", 1e-6)
    M_safe = kwargs.pop("M_safe", max_iter/100)

    # Validate parameters.
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if t_init <= 0:
        raise ValueError("t_init must be a positive scalar.")
    if eps_abs < 0:
        raise ValueError("eps_abs must be a non-negative scalar.")
    if eps_rel < 0:
        raise ValueError("eps_rel must be a non-negative scalar.")
    if m_accel <= 0:
        raise ValueError("m_accel must be a positive integer.")
    if lam_accel < 0:
        raise ValueError("lam_accel must be a non-negative scalar.")
    if not aa_method in ["lstsq", "lsqr"]:
        raise ValueError("aa_method must be either 'lstsq' or 'lsqr'.")
    if D_safe < 0:
        raise ValueError("D_safe must be a non-negative scalar.")
    if eps_safe < 0:
        raise ValueError("eps_safe must be a non-negative scalar.")
    if M_safe <= 0:
        raise ValueError("M_safe must be a positive integer.")

    # DRS parameters.
    N = len(p_list)   # Number of subproblems.
    has_constr = len(A_list) != 0
    if len(A_list) == 0:
        if b.size != 0:
            raise ValueError("Dimension mismatch: nrow(A_i) != nrow(b)")
        if n_list is not None:
            if len(n_list) != N:
                raise ValueError("n_list must have exactly {} entries".format(N))
            A_list = [sp.csr_matrix((0, ni)) for ni in n_list]
        elif v_init is not None:
            if len(v_init) != N:
                raise ValueError("v_init must be None or contain exactly {} entries".format(N))
            A_list = [sp.csr_matrix((0, vi.shape[0])) for vi in v_init]
        else:
            raise ValueError("n_list or v_init must be defined if A_list and b are empty")
    if len(A_list) != N:
        raise ValueError("A_list must be empty or contain exactly {} entries".format(N))
    if v_init is None:
        # v_init = [np.random.randn(A.shape[1]) for A in A_list]
        v_init = [np.zeros(A.shape[1]) for A in A_list]
        # v_init = [sp.csc_matrix((A.shape[1],1)) for A in A_list]
    if len(v_init) != N:
        raise ValueError("v_init must be None or contain exactly {} entries".format(N))

    # Variable size list.
    if n_list is None:
        n_list = [A_list[i].shape[1] for i in range(N)]
    if len(n_list) != N:
        raise ValueError("n_list must be None or contain exactly {} entries".format(N))
    n_list_cumsum = np.insert(np.cumsum(n_list), 0, 0)

    for i in range(N):
        if A_list[i].shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch: nrow(A_i) != nrow(b)")
        elif A_list[i].shape[1] != v_init[i].shape[0]:
            raise ValueError("Dimension mismatch: ncol(A_i) != nrow(v_i)")
        elif A_list[i].shape[1] != n_list[i]:
            raise ValueError("Dimension mismatch: ncol(A_i) != n_i")
        if not sp.issparse(A_list[i]):
            A_list[i] = sp.csr_matrix(A_list[i])

    # Precondition data.
    if precond and has_constr:
        p_list, A_list, b, e_pre = precondition(p_list, A_list, b)
        t_init = 1/gmean(e_pre)**2/10
        print('After preconditioning, t_init changed to {}'.format(t_init))

    # Store constraint matrix for projection step.
    A = sp.csr_matrix(sp.hstack(A_list))

    # Set up the workers.
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=a2dr_worker, args=(remote, p_list[i], v_init[i], A_list[i], \
                                                    t_init, anderson, m_accel) + args)]
        procs[-1].start()

    # Initialize AA-II variables.
    if anderson:   # TODO: Store and update these efficiently as arrays.
        n_sum = np.sum([np.prod(v.shape) for v in v_init])
        g_vec = np.zeros(n_sum)   # g^(k) = v^(k) - F(v^(k)).
        s_hist = []  # History of s^(j) = v^(j+1) - v^(j), kept in S^(k) = [s^(k-m_k) ... s^(k-1)].
        y_hist = []  # History of y^(j) = g^(j+1) - g^(j), kept in Y^(k) = [y^(k-m_k) ... y^(k-1)].
        n_AA = M_AA = 0   # Safeguarding counters.

    # A2DR loop.
    k = 0
    finished = False
    safeguard = True
    r_primal = np.zeros(max_iter)
    r_dual = np.zeros(max_iter)
    r_best = np.inf

    # Warm start terms.
    dk = np.zeros(A.shape[1])
    sol = np.zeros(A.shape[0])

    start = time()
    while not finished:
        # TODO: Add verbose printout.
        if k % 10 == 0:
            print("Iteration:", k)

        # Gather v_i^(k+1/2) from nodes.
        v_halves = [pipe.recv() for pipe in pipes]

        # Projection step for x^(k+1).
        v_half = np.concatenate(v_halves, axis=0)
        dk = sp.linalg.lsqr(A, A.dot(v_half) - b, atol=1e-10, btol=1e-10, x0=dk)[0]

        # Scatter d^k = A^\dagger(Av^(k+1/2) - b).
        for i in range(N):
            pipes[i].send((dk[n_list_cumsum[i]:n_list_cumsum[i+1]], k))

        if anderson and k > 0: # for k = 0, always do the vanilla DRS update
            m_k = min(m_accel, k)  # Keep (y^(j), s^(j)) for iterations (k-m_k) through (k-1).

            # Gather s_i^(k-1) and g_i^(k) from nodes.
            sg_update = [pipe.recv() for pipe in pipes]
            s_new, g_new = map(list, zip(*sg_update))
            s_new = np.concatenate(s_new, axis=0)   # s_i^(k-1) = v_i^(k) - v_i^(k-1).
            g_new = np.concatenate(g_new, axis=0)   # g_i^(k) = v_i^(k) - F(v_i^(k)) = x_i^(k+1/2) - x_i^(k+1).

            # Save newest column y^(k-1) = g^(k) - g^(k-1) of matrix Y^(k).
            y_hist.append(g_new - g_vec)
            if len(y_hist) > m_k:
                y_hist.pop(0)
            g_vec = g_new

            # Save newest column s^(k-1) = v^(k) - v^(k-1) of matrix S^(k).
            s_hist.append(s_new)
            if len(s_hist) > m_k:
                s_hist.pop(0)

            # Safeguard update.
            if safeguard or M_AA >= M_safe:
                if LA.norm(g_vec) <= D_safe*g0_norm*(n_AA/M_safe + 1)**(-(1 + eps_safe)):
                    AA_update = True
                    n_AA = n_AA + 1
                    M_AA = 0
                    safeguard = False
                else:
                    AA_update = False
                    M_AA = 0
            else:
                AA_update = True
                M_AA = M_AA + 1
                n_AA = n_AA + 1

            # Scatter safeguarding decision.
            for pipe in pipes:
                pipe.send(AA_update)
            if AA_update:
                # Compute and scatter AA-II weights.
                Y_mat = np.column_stack(y_hist)
                S_mat = np.column_stack(s_hist)
                if ada_reg:
                    reg = lam_accel * (LA.norm(Y_mat)**2 + LA.norm(S_mat)**2)  # AA-II regularization.
                else:
                    reg = lam_accel
                alpha = aa_weights(Y_mat, g_new, reg, type=aa_method, rcond=None)
                for pipe in pipes:
                    pipe.send(alpha)
              
        elif anderson and k == 0:
            AA_update = False   # Initial step is always DRS.
            g_new = [pipe.recv() for pipe in pipes]
            g_vec = np.concatenate(g_new, axis=0)
            g0_norm = LA.norm(g_vec)

        # Compute l2-norm of primal and dual residuals.
        r_update = [pipe.recv() for pipe in pipes]
        x_halves, Ax_halves, xv_diffs = map(list, zip(*r_update))
        r_primal_vec = sum(Ax_halves) - b
        r_primal[k] = LA.norm(r_primal_vec, ord=2)

        subgrad = np.concatenate(xv_diffs)/t_init
        # sol = LA.lstsq(A.T, subgrad, rcond=None)[0]
        sol = sp.linalg.lsqr(A.T, subgrad, atol=1e-10, btol=1e-10, x0=sol)[0]
        r_dual_vec = A.T.dot(sol) - subgrad
        r_dual[k] = LA.norm(r_dual_vec, ord=2)

        # Save x_i^(k+1/2) if residual norm is smallest so far.
        r_all = LA.norm(np.concatenate([r_primal_vec, r_dual_vec]), ord=2)
        if k == 0:   # Store ||r^0||_2 for stopping criterion.
            r_all_0 = r_all
        if k == 0 or r_all < r_best:
            x_final = x_halves

        # Stop when residual norm falls below tolerance.
        k = k + 1
        finished = k >= max_iter or (r_all <= eps_abs + eps_rel * r_all_0)

    # Unscale and return x_i^(k+1/2).
    [p.terminate() for p in procs]
    if precond and has_constr:
        x_final = [ei*x for x, ei in zip(x_final, e_pre)]
    end = time()
    return {"x_vals": x_final, "primal": np.array(r_primal[:k]), "dual": np.array(r_dual[:k]), \
            "num_iters": k, "solve_time": (end - start)}
