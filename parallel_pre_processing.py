import pandas as pd
import torch
import os
from collections import defaultdict
from datetime import datetime

# ======================================================
# CONFIGURATION
# ======================================================
CSV_PATH = r"D:\Minor\Datasets\FIRMS\DL_FIRE_J1V-C2_711524\fire_archive_J1V-C2_711524.csv"
OUT_PATH = "node_centric_dataset.pt"

CELL_SIZE = 0.25          # grid resolution (same as TGNN)
WINDOW_DAYS = 7
CHUNK_SIZE = 300_000

CONF_MAP = {"l": 0.3, "m": 0.6, "h": 0.9, "n": 0.0}

USE_COLS = [
    "latitude", "longitude",
    "brightness", "frp", "bright_t31",
    "confidence", "acq_date", "acq_time"
]

# ======================================================
# GLOBAL GRID SETUP
# ======================================================
LAT_MIN, LAT_MAX = -90, 90
LON_MIN, LON_MAX = -180, 180

NUM_LAT = int((LAT_MAX - LAT_MIN) / CELL_SIZE)
NUM_LON = int((LON_MAX - LON_MIN) / CELL_SIZE)

def latlon_to_node(lat, lon):
    r = int((lat - LAT_MIN) // CELL_SIZE)
    c = int((lon - LON_MIN) // CELL_SIZE)
    return r * NUM_LON + c

# ======================================================
# STORAGE: node_id → date → feature accumulators
# ======================================================
daily_data = defaultdict(lambda: defaultdict(list))

print("Scanning FIRMS data (node-centric)...")

# ======================================================
# PASS 1: STREAM CSV AND AGGREGATE DAILY NODE DATA
# ======================================================
for chunk in pd.read_csv(CSV_PATH, usecols=USE_COLS, chunksize=CHUNK_SIZE):

    dt = pd.to_datetime(
        chunk.acq_date + " " +
        chunk.acq_time.astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M"
    )
    chunk["date"] = dt.dt.date

    for _, row in chunk.iterrows():
        node = latlon_to_node(row.latitude, row.longitude)

        daily_data[node][row.date].append((
            row.brightness,
            row.frp,
            CONF_MAP.get(row.confidence, 0.0),
            row.bright_t31
        ))

print("Finished scanning. Building node-centric samples...")

# ======================================================
# BUILD NODE-CENTRIC TEMPORAL SAMPLES
# ======================================================
samples_X = []
samples_y = []

for node, date_dict in daily_data.items():

    dates = sorted(date_dict.keys())
    if len(dates) < WINDOW_DAYS + 1:
        continue

    # Precompute daily feature vectors
    daily_features = {}

    for d in dates:
        vals = date_dict[d]
        fire_count = len(vals)

        brightness = [v[0] for v in vals]
        frp = [v[1] for v in vals]
        conf = [v[2] for v in vals]
        t31 = [v[3] for v in vals]

        daily_features[d] = torch.tensor([
            fire_count,
            sum(brightness) / fire_count,
            max(brightness),
            sum(frp) / fire_count,
            sum(conf) / fire_count,
            sum(t31) / fire_count
        ], dtype=torch.float32)

    # Sliding window over time
    for i in range(WINDOW_DAYS - 1, len(dates) - 1):
        window_days = dates[i - WINDOW_DAYS + 1:i + 1]
        next_day = dates[i + 1]

        # Build input sequence
        X_seq = torch.stack([daily_features[d] for d in window_days])

        # Label: new ignition
        prev_fire = sum(daily_features[d][0] for d in window_days)
        next_fire = len(date_dict[next_day])

        y = 1.0 if (prev_fire == 0 and next_fire > 0) else 0.0

        samples_X.append(X_seq)
        samples_y.append(torch.tensor(y))

# ======================================================
# FINAL DATASET TENSORS
# ======================================================
X = torch.stack(samples_X)      # [num_samples, 7, 6]
y = torch.stack(samples_y)      # [num_samples]

torch.save(
    {
        "X": X,
        "y": y
    },
    OUT_PATH
)

print("\nNode-centric dataset created.")
print("Saved to:", OUT_PATH)
print("X shape:", X.shape)
print("y shape:", y.shape)
