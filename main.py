"""Main runner for HMM training and inference.

Usage examples:
  python main.py --data birds.csv --N 4 --iters 30
"""
import argparse
import numpy as np

from data import load_bird_csv, discretize_positions, build_sequences, build_obs_vocab, encode_sequences, sequences_to_list
from em import em
from viterbi import viterbi
from visualize import plot_viterbi_path, plot_state_timeline, plot_path_on_gps, summarize_state_usage
from state_characteristics import compute_features, summarize_states, plot_state_speeds



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="birds.csv", help="path to birds CSV")
    parser.add_argument("--N", type=int, default=4, help="number of hidden states")
    parser.add_argument("--iters", type=int, default=30, help="baum-welch iterations")
    parser.add_argument("--plot-bird", type=str, default=None, help="bird id to plot (default: first bird)")
    args = parser.parse_args()

    print("Loading data:", args.data)
    df = load_bird_csv(args.data)
    df = discretize_positions(df, lon_bin=0.1, lat_bin=0.1)

    sequences = build_sequences(df)
    vocab = build_obs_vocab(sequences)
    enc_sequences = encode_sequences(sequences, vocab)
    all_seqs = sequences_to_list(enc_sequences)
    
    print(f"num sequences: {len(all_seqs)}  vocab size: {len(vocab)}")

    # NOW define N and V
    N = args.N
    V = len(vocab)

    # Perform generalization test
    results = cross_bird_validation(enc_sequences, N=N, V=V, iters=args.iters)

    # Plot the results
    from visualize import plot_train_test_ll
    plot_train_test_ll(results)


    print(f"num sequences: {len(all_seqs)}  vocab size: {len(vocab)}")

    N = args.N
    V = len(vocab)
    print(f"Training HMM with N={N}, V={V}, iters={args.iters}")
    pi, A, B, ll_hist = em(all_seqs, N=N, V=V, n_iters=args.iters, verbose=True)

    default_bird = next(iter(enc_sequences.keys()))
    bird_to_use = args.plot_bird or default_bird
    if bird_to_use not in enc_sequences:
      print(f"Requested bird id '{bird_to_use}' not found; using '{default_bird}' instead.")
      bird_to_use = default_bird

    obs_seq = enc_sequences[bird_to_use]
    path = viterbi(B, pi, np.asarray(obs_seq, dtype=int), A)

    print(f"Ran Viterbi for bird {bird_to_use}: seq_len={len(obs_seq)} states_used={len(set(path))}")
    # Compute full feature dataframe for the bird
    features = compute_features(df, bird_to_use, path)

    # Summaries
    stats = summarize_states(features, N)
    print("\n=== State Movement Characteristics ===")
    for s, info in stats.items():
        print(f"State {s}:")
        for k, v in info.items():
            print(f"   {k}: {v}")
        print()

    # Show the speed dist graphs
    plot_state_speeds(features, N)

    print("First 30 states:", path[:30])

    print("\nSummarizing state usage across birds:")
    summarize_state_usage(enc_sequences, pi, A, B)

    print(f"Plotting for bird {bird_to_use}...")
    plot_viterbi_path(path, bird_to_use)
    plot_state_timeline(path, bird_to_use)
    plot_path_on_gps(df, bird_to_use, path)

def cross_bird_validation(enc_sequences, N, V, iters):
    """
    Leave-one-bird-out HMM validation.
    For each bird:
      - Train HMM on all other birds
      - Evaluate log-likelihood on held-out bird
      - Compare train LL vs test LL
    """
    from em import em
    from viterbi import viterbi
    import numpy as np

    results = []

    bird_ids = list(enc_sequences.keys())

    print("\n================ Cross-Bird Validation ================")
    for held_out in bird_ids:
        # Partition
        train_seqs = [seq for b, seq in enc_sequences.items() if b != held_out]
        test_seqs  = [enc_sequences[held_out]]

        # Train HMM on all other birds
        pi, A, B, ll_hist = em(
            sequences=train_seqs,
            N=N,
            V=V,
            n_iters=iters,
            verbose=False
        )

        # Compute log-likelihood on TRAINING sequences
        train_ll = 0
        train_T = 0
        for seq in train_seqs:
            # forward log-likelihood via the scaled alphas
            obs = np.asarray(seq, dtype=int)
            T = len(obs)
            # small forward impl with numeric guards to avoid division by zero
            eps = 1e-12
            alpha = pi * B[:, obs[0]]
            c0 = alpha.sum()
            alpha /= (c0 + eps)
            loglik = -np.log(c0 + eps)
            for t in range(1, T):
                alpha = (alpha @ A) * B[:, obs[t]]
                ct = alpha.sum()
                alpha /= (ct + eps)
                loglik += -np.log(ct + eps)
            train_ll += loglik
            train_T += T
        train_ll_per_token = train_ll / train_T

        # Compute log-likelihood on TEST bird
        test_ll = 0
        test_T = 0
        for seq in test_seqs:
            obs = np.asarray(seq, dtype=int)
            T = len(obs)
            eps = 1e-12
            alpha = pi * B[:, obs[0]]
            c0 = alpha.sum()
            alpha /= (c0 + eps)
            loglik = -np.log(c0 + eps)
            for t in range(1, T):
                alpha = (alpha @ A) * B[:, obs[t]]
                ct = alpha.sum()
                alpha /= (ct + eps)
                loglik += -np.log(ct + eps)
            test_ll += loglik
            test_T += T
        test_ll_per_token = test_ll / test_T

        results.append((held_out, train_ll_per_token, test_ll_per_token))

        print(f"Bird {held_out}:  train LL/token = {train_ll_per_token:.3f},   "
              f"test LL/token = {test_ll_per_token:.3f}")

    print("\n=== Summary (Train vs Test LL per token) ===")
    for bird, tr, te in results:
        print(f"{bird}:  train={tr:.3f}, test={te:.3f}")

    return results

if __name__ == "__main__":
    main()
