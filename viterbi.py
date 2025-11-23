from numpy.typing import NDArray
import numpy as np
from typing import List


def viterbi(emissions: NDArray, initials: NDArray, observations: NDArray, transitions: NDArray) -> NDArray:
    """
    emissions: B matrix shape (S, M)
    initials: pi shape (S,)
    observations: array of ints shape (T,)
    transitions: A matrix shape (S, S)
    """
    n = transitions.shape[0]
    T = observations.shape[0]

    log_a = np.log(transitions)
    log_b = np.log(emissions)
    log_pi = np.log(initials)

    viterbi_table = np.zeros((n, T))
    backpointer_table = np.zeros((n, T), dtype=int)

    first_obs = observations[0]
    viterbi_table[:, 0] = log_pi + log_b[:, first_obs]
    backpointer_table[:, 0] = 0

    for t in range(1, T):
        obs = observations[t]
        prev_viterbi_col = viterbi_table[:, t - 1][:, np.newaxis]
        transition_probs = prev_viterbi_col + log_a
        max_probs = np.max(transition_probs, axis=0)
        best_prev = np.argmax(transition_probs, axis=0)
        viterbi_table[:, t] = max_probs + log_b[:, obs]
        backpointer_table[:, t] = best_prev

    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(viterbi_table[:, T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer_table[path[t + 1], t + 1]

    return path.tolist()
