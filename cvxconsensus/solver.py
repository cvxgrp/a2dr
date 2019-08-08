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
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.stats.mstats import gmean
from scipy.stats import hmean
from time import time
from multiprocessing import Process, Pipe
from cvxconsensus.proximal.prox_point import prox_point
from cvxconsensus.precondition import precondition
from cvxconsensus.acceleration import aa_weights

NNZ_RATIO = 0.1   # Maximum number of nonzeros to be considered sparse.

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

        # Send A*x^(k+1/2) and x^(k+1/2) - v^(k) for computing residuals.
        pipe.send((A.dot(x_half), x_half - v_vec))
        v_vec = v_new

        # Send x_i^(k+1/2) if A2DR terminated.
        finished = pipe.recv()
        if finished:
            pipe.send(x_half)

def a2dr(p_list, A_list = [], b = np.array([]), v_init = None, *args, **kwargs):
    # Problem parameters.
    max_iter = kwargs.pop("max_iter", 1000)
    t_init = kwargs.pop("t_init", 10)  # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)   # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-8)   # Relative stopping tolerance.
    precond = kwargs.pop("precond", False)  # Precondition A and b?
    ada_reg = kwargs.pop("ada_reg", True) # Adaptive regularization?

    # AA-II parameters.
    anderson = kwargs.pop("anderson", False)
    m_accel = int(kwargs.pop("m_accel", 10))    # Maximum past iterations to keep (>= 0).
    lam_accel = kwargs.pop("lam_accel", 1e-8) #1e-10   # AA-II regularization weight.

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
    if D_safe < 0:
        raise ValueError("D_safe must be a non-negative scalar.")
    if eps_safe < 0:
        raise ValueError("eps_safe must be a non-negative scalar.")
    if M_safe <= 0:
        raise ValueError("M_safe must be a positive integer.")

    # DRS parameters.
    N = len(p_list)   # Number of subproblems.
    if len(A_list) == 0:
        return prox_point(p_list, v_init, *args, **kwargs)
    if len(A_list) != N:
        raise ValueError("A_list must be empty or contain exactly {} entries".format(N))
    if v_init is None:
        # v_init = [np.random.randn(A.shape[1]) for A in A_list]
        v_init = [np.zeros(A.shape[1]) for A in A_list]
        # v_init = [sp.csc_matrix((A.shape[1],1)) for A in A_list]
    if len(v_init) != N:
        raise ValueError("v_init must None or contain exactly {} entries".format(N))
    for i in range(N):
        if A_list[i].shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch: nrow(A_i) != nrow(b)")
        elif A_list[i].shape[1] != v_init[i].shape[0]:
            raise ValueError("Dimension mismatch: ncol(A_i) != nrow(v_i)")
            
    # variable size list
    n_list = [A_list[i].shape[1] for i in range(N)]
    n_list_cumsum = np.insert(np.cumsum(n_list), 0, 0)

    # Precondition data.
    if precond:
        p_list, A_list, b, e_pre = precondition(p_list, A_list, b)
        t_init = 1/gmean(e_pre)**2/10
        print('after preconditioning, t_init changed to {}'.format(t_init))

    # Store constraint matrix for projection step.
    # A = np.hstack(A_list)
    A = sp.csr_matrix(sp.hstack(A_list))
    if A.count_nonzero() <= NNZ_RATIO*np.prod(A.shape):   # If sparse, define linear operator.
        AATx_fun = lambda x: A.dot(A.T.dot(x))
        AAT = sp.linalg.LinearOperator((A.shape[0], A.shape[0]), matvec=AATx_fun, rmatvec=AATx_fun)
    else:
        AAT = A.dot(A.T)   # If dense, calculate directly and cache.

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
        dk = sp.linalg.lsqr(A, A.dot(v_half) - b, atol=1e-16, btol=1e-16, x0=dk)[0]

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
                alpha = aa_weights(Y_mat, g_new, reg, rcond=None)
                for pipe in pipes:
                    pipe.send(alpha)
              
        elif anderson and k == 0:
            AA_update = False   # Initial step is always DRS.
            g_new = [pipe.recv() for pipe in pipes]
            g_vec = np.concatenate(g_new, axis=0)
            g0_norm = LA.norm(g_vec)

        # Compute l2-norm of primal and dual residuals.
        r_update = [pipe.recv() for pipe in pipes]
        Ax_halves, xv_diffs = map(list, zip(*r_update))
        r_primal[k] = LA.norm(sum(Ax_halves) - b, ord=2)
        subgrad = np.concatenate(xv_diffs)/t_init
        # sol = LA.lstsq(A.T, subgrad, rcond=None)[0]
        sol = sp.linalg.lsqr(A.T, subgrad, atol=1e-16, btol=1e-16, x0=sol)[0]
        r_dual[k] = LA.norm(A.T.dot(sol) - subgrad, ord=2)

        # Stop if residual norms fall below tolerance.
        k = k + 1
        finished = k >= max_iter or (r_primal[k-1] <= eps_abs + eps_rel * r_primal[0] and \
                                     r_dual[k-1] <= eps_abs + eps_rel * r_dual[0])
        for pipe in pipes:
            pipe.send(finished)

    # Gather and return x_i^(k+1/2) from nodes.
    x_final = [pipe.recv() for pipe in pipes]
    [p.terminate() for p in procs]
    if precond:
        x_final = [ei*x for x, ei in zip(x_final, e_pre)]
    end = time()
    return {"x_vals": x_final, "primal": np.array(r_primal[:k]), "dual": np.array(r_dual[:k]), \
            "num_iters": k, "solve_time": (end - start)}
