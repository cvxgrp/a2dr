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
from time import time
from multiprocessing import Process, Pipe
from cvxconsensus.acceleration import aa_weights

def prox_worker_alt(pipe, prox, x_init, rho, anderson, m_accel, lam_accel):
    n = x_init.shape[0]
    x_vec = x_init.copy()

    # Initialize AA-II parameters.
    if anderson:   # TODO: Should we accelerate individual workers or joint problem?
        x_res = np.zeros(n)  # s^(k-1) = x^(k) - x^(k-1).
        g_vec = np.zeros(n)  # g^(k) = x^(k) - F(x^(k)).
        F_hist = []  # History of F(x^(k))
        s_hist = []  # History of s^(j) = x^(j+1) - x^(j), kept in S^(k) = [s^(k-m_k) ... s^(k-1)].
        y_hist = []  # History of y^(j) = g^(j+1) - g^(j), kept in Y^(k) = [y^(k-m_k) ... y^(k-1)].

    # Proximal point loop.
    k = 0
    while True:
        # Proximal step for x^(k+1).
        x_new = prox(x_vec, rho)
        g_new = x_vec - x_new
        pipe.send(np.sum(g_new**2))

        if anderson:
            m_k = min(m_accel, k+1)

            # Save history of F(x^(k)).
            F_hist.append(x_new)
            if len(F_hist) > m_k + 1:
                F_hist.pop(0)

            # Save newest column y^(k-1) = g^(k) - g^(k-1) of matrix Y^(k).
            y_hist.append(g_new - g_vec)
            if len(y_hist) > m_k:
                y_hist.pop(0)
            g_vec = g_new

            # Save newest column s^(k-1) = x^(k) - x^(k-1) of matrix S^(k).
            s_hist.append(x_res)
            if len(s_hist) > m_k:
                s_hist.pop(0)

            # Compute AA-II weights.
            Y_mat = np.column_stack(y_hist)
            S_mat = np.column_stack(s_hist)
            reg = lam_accel*(np.linalg.norm(Y_mat)**2 + np.linalg.norm(S_mat)**2)  # AA-II regularization.
            alpha = aa_weights(Y_mat, g_new, reg, rcond=None)

            # Weighted update of x^(k+1).
            x_new = np.column_stack(F_hist).dot(alpha[:(k+1)])

            # Save x^(k+1) - x^(k) for next iteration.
            x_res = x_new - x_vec

        # Send x^(k+1) if loop terminated.
        finished = pipe.recv()
        if finished:
            pipe.send(x_new)
        x_vec = x_new
        k = k + 1

def prox_point_alt(p_list, x_init, *args, **kwargs):
    # Problem parameters.
    N = len(p_list)   # Number of subproblems.
    max_iter = kwargs.pop("max_iter", 1000)
    rho_init = kwargs.pop("rho_init", 1.0)  # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)  # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-8)  # Relative stopping tolerance.

    # AA-II parameters.
    anderson = kwargs.pop("anderson", False)
    m_accel = int(kwargs.pop("m_accel", 5))  # Maximum past iterations to keep (>= 0).
    if m_accel <= 0:
        raise ValueError("m_accel must be a positive integer.")
    lam_accel = kwargs.pop("lam_accel", 0)  # AA-II regularization weight.

    # Set up the workers.
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=prox_worker_alt, args=(remote, p_list[i], x_init[i], rho_init, \
                                                    anderson, m_accel, lam_accel) + args)]
        procs[-1].start()

    # Proximal point loop.
    k = 0
    finished = False
    resid = np.zeros(max_iter)

    start = time()
    while not finished:
        x_sses = [pipe.recv() for pipe in pipes]
        resid[k] = rho_init*np.sqrt(np.sum(x_sses))

        # Stop if residual norms fall below tolerance.
        k = k + 1
        finished = k >= max_iter or (resid[k-1] <= eps_abs + eps_rel * resid[0])
        for pipe in pipes:
            pipe.send(finished)

    # Gather and return x_i^(k+1) from nodes.
    x_final = [pipe.recv() for pipe in pipes]
    [p.terminate() for p in procs]
    end = time()
    return {"x_vals": x_final, "primal": np.zeros(k), "dual": np.array(resid[:k]), \
            "num_iters": k, "solve_time": (end - start)}

def prox_worker(pipe, prox, x_init, rho, anderson, m_accel):
    # Initialize AA-II parameters.
    if anderson:   # TODO: Store and update these efficiently as arrays.
        F_hist = []  # History of F(x^(k))
        x_res = np.zeros(x_init.shape[0])  # s^(k-1) = x^(k) - x^(k-1).
    x_vec = x_init.copy()

    # Proximal point loop.
    k = 0
    while True:
        # Proximal step for x^(k+1).
        x_half = prox(x_vec, rho)

        if anderson:
            m_k = min(m_accel, k+1)

            # Save history of F(x^(k)).
            F_hist.append(x_half)
            if len(F_hist) > m_k + 1:
                F_hist.pop(0)

            # Send s^(k-1) = x^(k) - x^(k-1) and g^(k) = x^(k) - F(x^(k)).
            pipe.send((x_res, x_vec - x_half))

            # Receive AA-II weights for x^(k+1).
            alpha = pipe.recv()

            # Weighted update of x^(k+1).
            x_new = np.column_stack(F_hist).dot(alpha[:(k+1)])

            # Save residual x^(k+1) - x^(k) for next iteration.
            x_res = x_new - x_vec
        else:
            x_new = x_half

        # Send sum-of-squared residual.
        resid = x_half - x_vec
        pipe.send(np.sum(resid**2))
        x_vec = x_new

        # Send x^(k+1) if loop terminated.
        finished = pipe.recv()
        if finished:
            pipe.send(x_half)
        k = k + 1

def prox_point(p_list, x_init, *args, **kwargs):
    # Problem parameters.
    N = len(p_list)  # Number of subproblems.
    max_iter = kwargs.pop("max_iter", 1000)
    rho_init = kwargs.pop("rho_init", 1.0)  # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)  # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-8)  # Relative stopping tolerance.

    # AA-II parameters.
    anderson = kwargs.pop("anderson", False)
    m_accel = int(kwargs.pop("m_accel", 5))  # Maximum past iterations to keep (>= 0).
    lam_accel = kwargs.pop("lam_accel", 0)  # AA-II regularization weight.

    # Validate parameters.
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if rho_init <= 0:
        raise ValueError("rho_init must be a positive scalar.")
    if eps_abs < 0:
        raise ValueError("eps_abs must be a non-negative scalar.")
    if eps_rel < 0:
        raise ValueError("eps_rel must be a non-negative scalar.")
    if m_accel <= 0:
        raise ValueError("m_accel must be a positive integer.")
    if lam_accel < 0:
        raise ValueError("lam_accel must be a non-negative scalar.")

    # Set up the workers.
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=prox_worker, args=(remote, p_list[i], x_init[i], rho_init, \
                                                        anderson, m_accel) + args)]
        procs[-1].start()

    # Initialize AA-II parameters.
    if anderson:  # TODO: Should we accelerate individual workers or joint problem?
        n_sum = np.sum([x.size for x in x_init])
        g_vec = np.zeros(n_sum)  # g^(k) = x^(k) - F(x^(k)).
        s_hist = []  # History of s^(j) = x^(j+1) - x^(j), kept in S^(k) = [s^(k-m_k) ... s^(k-1)].
        y_hist = []  # History of y^(j) = g^(j+1) - g^(j), kept in Y^(k) = [y^(k-m_k) ... y^(k-1)].

    # Proximal point loop.
    k = 0
    finished = False
    r_dual = np.zeros(max_iter)

    start = time()
    while not finished:
        if anderson:
            m_k = min(m_accel, k+1)  # Keep (y^(j), s^(j)) for iterations (k-m_k) through (k-1).

            # Gather s_i^(k-1) and g_i^(k) from nodes.
            sg_update = [pipe.recv() for pipe in pipes]
            s_new, g_new = map(list, zip(*sg_update))
            s_new = np.concatenate(s_new, axis=0)   # s_i^(k-1) = x_i^(k) - x_i^(k-1)
            g_new = np.concatenate(g_new, axis=0)   # g_i^(k) = x_i^(k) - F(x_i^(k))

            # Save newest column y^(k-1) = g^(k) - g^(k-1) of matrix Y^(k).
            y_hist.append(g_new - g_vec)
            if len(y_hist) > m_k:
                y_hist.pop(0)
            g_vec = g_new

            # Save newest column s^(k-1) = x^(k) - x^(k-1) of matrix S^(k).
            s_hist.append(s_new)
            if len(s_hist) > m_k:
                s_hist.pop(0)

            # Compute and scatter AA-II weights.
            Y_mat = np.column_stack(y_hist)
            S_mat = np.column_stack(s_hist)
            reg = lam_accel*(np.linalg.norm(Y_mat)**2 + np.linalg.norm(S_mat)**2)  # AA-II regularization.
            alpha = aa_weights(Y_mat, g_new, reg, rcond=None)
            for pipe in pipes:
                pipe.send(alpha)

        # Compute l2-norm of residual.
        x_sses = [pipe.recv() for pipe in pipes]
        r_dual[k] = rho_init*np.sqrt(np.sum(x_sses))

        # Stop if residual norm falls below tolerance.
        k = k + 1
        finished = k >= max_iter or (r_dual[k-1] <= eps_abs + eps_rel*r_dual[0])
        for pipe in pipes:
            pipe.send(finished)

    # Gather and return x_i^(k+1) from nodes.
    x_final = [pipe.recv() for pipe in pipes]
    [p.terminate() for p in procs]
    end = time()
    return {"x_vals": x_final, "primal": np.zeros(k), "dual": np.array(r_dual[:k]), \
            "num_iters": k, "solve_time": (end - start)}
