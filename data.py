import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def load_bird_csv(path: str) -> pd.DataFrame:
    """Load bird CSV and ensure timestamps are parsed and sorted.

    Returns a DataFrame containing at least columns:
    'tag-local-identifier', 'timestamp', 'location-long', 'location-lat'
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["tag-local-identifier", "timestamp"]) 
    return df


def discretize_positions(df: pd.DataFrame, lon_bin: float = 0.1, lat_bin: float = 0.1) -> pd.DataFrame:
    """Create discretized observation labels from long/lat.

    Adds columns: 'lon_bin', 'lat_bin', and 'obs' (string like '220_627').
    """
    df = df.copy()
    df["lon_bin"] = (df["location-long"] // lon_bin).astype(int)
    df["lat_bin"] = (df["location-lat"] // lat_bin).astype(int)
    df["obs"] = df["lon_bin"].astype(str) + "_" + df["lat_bin"].astype(str)
    return df


def build_sequences(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return dict mapping bird_id -> ordered list of obs (string form).
    """
    sequences = {}
    for bird_id, group in df.groupby("tag-local-identifier"):
        sequences[bird_id] = list(group["obs"])
    return sequences


def build_obs_vocab(sequences: Dict[str, List[str]]) -> Dict[str, int]:
    vocab = {}
    idx = 0
    for seq in sequences.values():
        for obs in seq:
            if obs not in vocab:
                vocab[obs] = idx
                idx += 1
    return vocab


def encode_sequences(sequences: Dict[str, List[str]], vocab: Dict[str, int]) -> Dict[str, List[int]]:
    encoded = {}
    for bird, seq in sequences.items():
        encoded[bird] = [vocab[o] for o in seq]
    return encoded


def sequences_to_list(encoded: Dict[str, List[int]]) -> List[List[int]]:
    """Convert encoded dict to list of sequences (for EM training)."""
    return list(encoded.values())
