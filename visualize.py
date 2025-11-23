import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from viterbi import viterbi
from typing import List, Dict


def plot_viterbi_path(state_path: List[int], bird_id: str) -> None:
    plt.figure(figsize=(14, 3))
    plt.plot(state_path, linewidth=1)
    plt.title(f"Viterbi Hidden State Path for Bird {bird_id}")
    plt.xlabel("Time Index")
    plt.ylabel("Hidden State")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_state_timeline(state_path: List[int], bird_id: str) -> None:
    plt.figure(figsize=(14, 1.2))
    plt.imshow([state_path], aspect='auto', cmap='tab20')
    plt.yticks([])
    plt.title(f"Viterbi State Timeline for Bird {bird_id}")
    plt.xlabel("Time Index")
    plt.show()


def plot_path_on_gps(df, bird_id: str, state_path: List[int]) -> None:
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


def summarize_state_usage(enc_sequences: Dict[str, List[int]], pi, A, B) -> None:
    global_counts = Counter()
    for bird, obs_seq in enc_sequences.items():
        path = viterbi(B, pi, np.asarray(obs_seq, dtype=int), A)
        counts = Counter(path)
        global_counts.update(path)
        print(f"{bird}: {len(set(path))} states, counts={dict(counts)}")
    print("\nGlobal:", dict(global_counts))
