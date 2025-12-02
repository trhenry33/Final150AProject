# state_characteristics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Haversine distance (km)
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# -------------------------------
# Bearing (degrees 0–360)
# -------------------------------
def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360) % 360


# -------------------------------
# Build movement-feature dataframe
# -------------------------------
def compute_features(df: pd.DataFrame, bird_id: str, state_path: list[int]) -> pd.DataFrame:
    """
    Returns a dataframe where each row is a *movement step* for this bird:
      columns: timestamp, location-long/lat (current),
               dt_hours, dist_km, speed_kmh, bearing, turn_angle, state
    The state is the Viterbi state at the *end* of the step.
    """
    # All rows for this bird, in time order
    sub = df[df["tag-local-identifier"] == bird_id].sort_values("timestamp").copy()
    T_data = len(sub)
    T_states = len(state_path)

    if T_data < 2:
        print(f"[compute_features] Bird {bird_id} has <2 points, nothing to compute.")
        return pd.DataFrame()

    # Make lengths match if there is any tiny mismatch
    T = min(T_data, T_states)
    sub = sub.iloc[:T].reset_index(drop=True)
    state_path = state_path[:T]

    # We will compute features for steps between t-1 -> t,
    # so there are (T-1) movement rows. Use t = 1..T-1.
    prev = sub.iloc[:-1]
    curr = sub.iloc[1:]

    feat = curr.copy()  # each row corresponds to "end" of step
    feat["state"] = np.array(state_path[1:T])

    # Time diff in hours
    dt = (curr["timestamp"].values - prev["timestamp"].values)
    dt_hours = dt.astype("timedelta64[s]").astype(float) / 3600.0
    feat["dt_hours"] = dt_hours

    # Distances and speed
    feat["dist_km"] = haversine(
        prev["location-lat"].values,
        prev["location-long"].values,
        curr["location-lat"].values,
        curr["location-long"].values,
    )

    # Avoid divide-by-zero; speed will be NaN when dt_hours == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        feat["speed_kmh"] = feat["dist_km"] / feat["dt_hours"]

    # Bearing and turning angle
    feat["bearing"] = bearing(
        prev["location-lat"].values,
        prev["location-long"].values,
        curr["location-lat"].values,
        curr["location-long"].values,
    )

    # Turning angle = change in bearing from previous step
    feat["turn_angle"] = np.abs(feat["bearing"] - feat["bearing"].shift(1))
    feat["turn_angle"] = feat["turn_angle"].fillna(0.0)

    # Drop rows where dt_hours is zero or NaN (no real movement info)
    feat = feat[np.isfinite(feat["dt_hours"]) & (feat["dt_hours"] > 0)].copy()

    return feat


# -------------------------------
# Summaries per hidden state
# -------------------------------
def summarize_states(feature_df: pd.DataFrame, N: int) -> dict:
    stats = {}

    if feature_df.empty:
        print("[summarize_states] feature_df is empty; no movement steps found.")
        for s in range(N):
            stats[s] = {"count": 0}
        return stats

    for s in range(N):
        grp = feature_df[feature_df["state"] == s]
        if grp.empty:
            stats[s] = {"count": 0}
            continue

        speeds = grp["speed_kmh"].dropna()
        steps = grp["dist_km"].dropna()
        turns = grp["turn_angle"].dropna()

        stats[s] = {
            "count": int(len(grp)),
            "mean_speed": float(speeds.mean()) if not speeds.empty else None,
            "median_speed": float(speeds.median()) if not speeds.empty else None,
            "mean_step_km": float(steps.mean()) if not steps.empty else None,
            "median_step_km": float(steps.median()) if not steps.empty else None,
            "mean_turn_angle_deg": float(turns.mean()) if not turns.empty else None,
            "speed_95pct": float(speeds.quantile(0.95)) if len(speeds) > 0 else None,
        }

    return stats


# -------------------------------
# Plot speed distributions
# -------------------------------
def plot_state_speeds(feature_df: pd.DataFrame, N: int) -> None:
    if feature_df.empty:
        print("[plot_state_speeds] feature_df is empty; nothing to plot.")
        return

    plt.figure(figsize=(10, 5))
    for s in range(N):
        grp = feature_df[feature_df["state"] == s]
        speeds = grp["speed_kmh"].dropna()
        if speeds.empty:
            continue
        plt.hist(speeds, bins=40, alpha=0.5, label=f"State {s}")

    plt.xlabel("Speed (km/h)")
    plt.ylabel("Frequency")
    plt.title("Speed distributions per hidden state")
    plt.legend()
    plt.tight_layout()
    plt.show()
