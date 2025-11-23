
import pandas as pd
import numpy as np


def load_bird_csv(path):
    df = pd.read_csv(path)

    # Ensure proper dtypes
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['tag-local-identifier', 'timestamp'])

    # Keep only needed columns
    df = df[['tag-local-identifier', 'timestamp', 
             'location-long', 'location-lat']]

    return df
# We need discrete variables to run HMM stuff
def discretize_positions(df, lon_bin=0.1, lat_bin=0.1):
    df['lon_bin'] = (df['location-long'] // lon_bin).astype(int)
    df['lat_bin'] = (df['location-lat'] // lat_bin).astype(int)

    df['obs'] = df['lon_bin'].astype(str) + '_' + df['lat_bin'].astype(str)
    return df

# need to have a sequence for each bird
def build_sequences(df):
    sequences = {}

    for bird_id, group in df.groupby('tag-local-identifier'):
        sequences[bird_id] = list(group['obs'])
    
    return sequences

K = 4  # number of hidden movement states, idk what we want yet

# make the HMM
def initialize_hmm(K, obs_vocab_size):
    pi = np.full(K, 1/K)                          # uniform start
    A = np.random.dirichlet(np.ones(K), size=K)   # KxK transitions
    B = np.random.dirichlet(np.ones(obs_vocab_size), size=K)  # emissions
    return pi, A, B
# obs vocab mapping
def build_obs_vocab(sequences):
    vocab = {}
    idx = 0
    for seq in sequences.values():
        for obs in seq:
            if obs not in vocab:
                vocab[obs] = idx
                idx += 1
    return vocab
# convert sequences into integer IDs
def encode_sequences(sequences, vocab):
    encoded = {}
    for bird, seq in sequences.items():
        encoded[bird] = [vocab[o] for o in seq]
    return encoded

# 1. Load CSV
df = load_bird_csv("birds.csv")

# 2. Discretize or cluster
df = discretize_positions(df)

# 3. Build sequences
sequences = build_sequences(df)

# 4. Build vocabulary + encode
vocab = build_obs_vocab(sequences)
enc_sequences = encode_sequences(sequences, vocab)

# 5. Initialize HMM
K = 4
pi, A, B = initialize_hmm(K, obs_vocab_size=len(vocab))

print("num sequences:", len(sequences))
print("vocab size:", len(vocab))
print("initial pi:", pi)
print("initial A:", A)
print("initial B shape:", B.shape)


"""
Notation for the Bird Migration HMM
-----------------------------------

We convert raw GPS data into an HMM-friendly representation:

1. vocab
   - Dictionary mapping each unique discretized location to an integer ID.
   - Example: vocab["220_627"] = 13
   - Size of vocab = V (number of distinct observation symbols).

2. sequences
   - Dictionary mapping each bird ID to its chronological list of observations.
   - Example: sequences["D118746"] = ["220_627", "220_627", "219_628", ...]
   - This is before converting observations to integers.

3. enc_sequences
   - Same structure as `sequences`, but each observation replaced by its vocab ID.
   - Example: enc_sequences["D118746"] = [13, 13, 21, ...]

4. Hidden states
   - We assume K hidden movement states: {1, 2, ..., K}
   - EM will infer these states (they are not in the CSV).

5. pi (initial state distribution)
   - Vector of length K.
   - pi[k] = P(Z_1 = k) = probability the sequence starts in hidden state k.

6. A (transition matrix)
   - K x K matrix.
   - A[i][j] = P(Z_{t+1} = j | Z_t = i)
   - Row i represents transitions *from* state i.

7. B (emission matrix)
   - K x V matrix.
   - B[k][o] = P(O_t = o | Z_t = k)
   - Row k represents emission probabilities for hidden state k.
   - Column o corresponds to an observation symbol from vocab.

In summary:
- vocab defines the observation space.
- sequences / enc_sequences define the temporal data for each bird.
- pi, A, B define the parameters of the Hidden Markov Model.
"""
#########################################################################################
#########################################################################################

# below can be tweaked and improved probably, I did it cause I could (vibe coded too close to the sun)

#########################################################################################
#########################################################################################



import numpy as np

def baum_welch(sequences, K, V, n_iters=20, tol=1e-4, verbose=False, seed=0):
    """
    Run EM (Baum–Welch) to learn HMM parameters.

    Args:
        sequences: list of observation sequences, each a list[int] of length T_n
        K: number of hidden states
        V: number of observation symbols
        n_iters: max number of EM iterations
        tol: stop if log-likelihood improvement < tol
        verbose: if True, print log-likelihood per iteration

    Returns:
        pi: (K,) initial state distribution
        A:  (K, K) transition matrix
        B:  (K, V) emission matrix
        loglik_history: list of log-likelihood values per iteration
    """
    rng = np.random.default_rng(seed)

    # Randomly initialize parameters
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)     # shape (K, K)
    B = rng.dirichlet(np.ones(V), size=K)     # shape (K, V)

    loglik_prev = -np.inf
    loglik_history = []

    for it in range(n_iters):
        # Accumulators for expected counts
        pi_acc = np.zeros(K)
        A_acc = np.zeros((K, K))
        B_acc = np.zeros((K, V))
        loglik_total = 0.0

        for obs in sequences:
            obs = np.asarray(obs, dtype=int)
            T = len(obs)

            # ---------- Forward pass with scaling ----------
            alpha = np.zeros((T, K))
            c = np.zeros(T)  # scaling factors

            alpha[0] = pi * B[:, obs[0]]
            c[0] = alpha[0].sum()
            if c[0] == 0:
                c[0] = 1e-16
            alpha[0] /= c[0]

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
                c[t] = alpha[t].sum()
                if c[t] == 0:
                    c[t] = 1e-16
                alpha[t] /= c[t]

            # Log-likelihood = - sum log(c_t)
            loglik_total += -np.sum(np.log(c + 1e-16))

            # ---------- Backward pass with scaling ----------
            beta = np.zeros((T, K))
            beta[-1] = 1.0 / c[-1]

            for t in range(T - 2, -1, -1):
                beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
                beta[t] /= c[t]

            # ---------- Gamma (state posteriors) ----------
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            # Accumulate initial state expectations
            pi_acc += gamma[0]

            # ---------- Xi (pairwise state posteriors) ----------
            for t in range(T - 1):
                # numerator: shape (K, K)
                num = A * (
                    alpha[t][:, None] * (B[:, obs[t + 1]] * beta[t + 1])[None, :]
                )
                denom = num.sum()
                if denom == 0:
                    denom = 1e-16
                xi_t = num / denom
                A_acc += xi_t

            # ---------- Emission counts ----------
            for t in range(T):
                B_acc[:, obs[t]] += gamma[t]

        # ---------- M-step: normalize counts ----------
        pi = pi_acc / pi_acc.sum()
        A = A_acc / A_acc.sum(axis=1, keepdims=True)
        B = B_acc / B_acc.sum(axis=1, keepdims=True)

        loglik_history.append(loglik_total)
        if verbose:
            print(f"iter {it:02d}  loglik = {loglik_total:.3f}")

        # Convergence check
        if abs(loglik_total - loglik_prev) < tol:
            break
        loglik_prev = loglik_total

    return pi, A, B, loglik_history

def viterbi(pi, A, B, obs):
    """
    Viterbi algorithm to find the most likely hidden-state sequence.

    Args:
        pi: (K,) initial state distribution
        A:  (K, K) transition matrix
        B:  (K, V) emission matrix
        obs: list[int] observation sequence (encoded indices)

    Returns:
        path: list[int] hidden-state sequence (same length as obs)
    """
    obs = np.asarray(obs, dtype=int)
    T = len(obs)
    K = len(pi)

    # Work in log-space to avoid underflow
    log_pi = np.log(pi + 1e-16)
    log_A = np.log(A + 1e-16)
    log_B = np.log(B + 1e-16)

    delta = np.zeros((T, K))       # best log-prob ending in state k at time t
    psi = np.zeros((T, K), dtype=int)  # backpointers

    # Initialization
    delta[0] = log_pi + log_B[:, obs[0]]
    psi[0] = 0

    # Recursion
    for t in range(1, T):
        for j in range(K):
            vals = delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(vals)
            delta[t, j] = vals[psi[t, j]] + log_B[j, obs[t]]

    # Backtracking
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path.tolist()

# enc_sequences: dict[bird_id] -> list[int]
all_seqs = list(enc_sequences.values())
K = 4                      # e.g., 4 hidden movement states
V = len(vocab)             # number of distinct obs symbols

pi, A, B, ll_hist = baum_welch(all_seqs, K=K, V=V, n_iters=30, verbose=True)

# Run Viterbi on a single bird to get its most likely migration "state path"
bird_id = next(iter(enc_sequences.keys()))
obs_seq = enc_sequences[bird_id]
state_path = viterbi(pi, A, B, obs_seq)

# print("bird:", bird_id)
# print("obs seq length:", len(obs_seq))
# print("state path length:", len(state_path))
# print("first 20 states:", state_path[:20])

# print("pi:", pi)
# print("row sums A:", A.sum(axis=1))
# print("row sums B:", B.sum(axis=1))

# # Check how many distinct states Viterbi ever uses for this bird
# unique_states = sorted(set(state_path))
# print("unique states in Viterbi path for this bird:", unique_states)

from collections import Counter

def summarize_state_usage(enc_sequences, pi, A, B):
    global_counts = Counter()
    for bird, obs_seq in enc_sequences.items():
        path = viterbi(pi, A, B, obs_seq)
        counts = Counter(path)
        global_counts.update(path)
        print(f"{bird}: {len(set(path))} states, counts={dict(counts)}")
    print("\nGlobal:", dict(global_counts))

# summarize_state_usage(enc_sequences, pi, A, B)

import matplotlib.pyplot as plt

def plot_viterbi_path(state_path, bird_id):
    plt.figure(figsize=(14, 3))
    plt.plot(state_path, linewidth=1)
    plt.title(f"Viterbi Hidden State Path for Bird {bird_id}")
    plt.xlabel("Time Index")
    plt.ylabel("Hidden State")
    plt.grid(True, alpha=0.3)
    plt.show()

# plot_viterbi_path(state_path, bird_id)

def plot_state_timeline(state_path, bird_id):
    plt.figure(figsize=(14, 1.2))
    plt.imshow([state_path], aspect='auto', cmap='tab20')
    plt.yticks([])
    plt.title(f"Viterbi State Timeline for Bird {bird_id}")
    plt.xlabel("Time Index")
    plt.show()

# plot_state_timeline(state_path, bird_id)


def plot_path_on_gps(df, bird_id, state_path):
    sub = df[df["tag-local-identifier"] == bird_id]
    lats = sub["location-lat"].values
    lons = sub["location-long"].values
    states = np.array(state_path)

    plt.figure(figsize=(7, 7))
    plt.scatter(lons, lats, c=states, cmap='tab20', s=8)
    plt.plot(lons, lats, color='lightgray', linewidth=0.5)
    plt.colorbar(label="Hidden State")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"GPS Path Color-Coded by Hidden States (Bird {bird_id})")
    plt.show()


plot_path_on_gps(df,bird_id,state_path)