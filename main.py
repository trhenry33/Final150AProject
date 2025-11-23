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
    print("First 30 states:", path[:30])

    print("\nSummarizing state usage across birds:")
    summarize_state_usage(enc_sequences, pi, A, B)

    print(f"Plotting for bird {bird_to_use}...")
    plot_viterbi_path(path, bird_to_use)
    plot_state_timeline(path, bird_to_use)
    plot_path_on_gps(df, bird_to_use, path)


if __name__ == "__main__":
    main()
