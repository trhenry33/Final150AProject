import numpy as np
from typing import List, Tuple


def em(sequences: List[List[int]], N: int, V: int, n_iters: int = 20, tol: float = 1e-4, verbose: bool = False, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    sequences: list of observation sequences (each a list of ints 0..V-1)
    N: number of hidden states
    V: vocabulary size (number of distinct observation symbols)
    Returns: (pi, A, B, loglik_history)
    """
    rng = np.random.default_rng(seed)

    pi = rng.dirichlet(np.ones(N))
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(V), size=N)

    loglik_prev = -np.inf
    loglik_history = []

    for it in range(n_iters):
        pi_acc = np.zeros(N)
        A_acc = np.zeros((N, N))
        B_acc = np.zeros((N, V))
        loglik_total = 0.0

        for obs in sequences:
            obs = np.asarray(obs, dtype=int)
            T = len(obs)

            # Forward with scaling
            alpha = np.zeros((T, N))
            c = np.zeros(T)

            alpha[0] = pi * B[:, obs[0]]
            c[0] = alpha[0].sum()
            alpha[0] /= c[0]

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
                c[t] = alpha[t].sum()
                alpha[t] /= c[t]

            loglik_total += -np.sum(np.log(c))

            # Backward with scaling
            beta = np.zeros((T, N))
            beta[-1] = 1.0 / c[-1]
            for t in range(T - 2, -1, -1):
                beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
                beta[t] /= c[t]

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            pi_acc += gamma[0]

            for t in range(T - 1):
                num = A * (alpha[t][:, None] * (B[:, obs[t + 1]] * beta[t + 1])[None, :])
                denom = num.sum()
                xi_t = num / denom
                A_acc += xi_t

            for t in range(T):
                B_acc[:, obs[t]] += gamma[t]

        # M-step
        pi = pi_acc / pi_acc.sum()
        A = A_acc / A_acc.sum(axis=1, keepdims=True)
        B = B_acc / B_acc.sum(axis=1, keepdims=True)

        loglik_history.append(loglik_total)
        if verbose:
            print(f"iter {it:02d}  loglik = {loglik_total:.3f}")

        if abs(loglik_total - loglik_prev) < tol:
            break
        loglik_prev = loglik_total

    return pi, A, B, loglik_history
