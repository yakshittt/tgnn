import pandas as pd
import torch
import os
from collections import defaultdict
from datetime import timedelta

# ======================================================
# CONFIGURATION
# ======================================================
CSV_PATH = r"D:\Minor\Datasets\FIRMS\DL_FIRE_J1V-C2_711524\fire_archive_J1V-C2_711524.csv"
OUT_DIR = "processed_tiles"

TILE_SIZE = 15.0          # degrees (region size)
CELL_SIZE = 0.25          # degrees (node grid inside tile)
WINDOW_DAYS = 7
CHUNK_SIZE = 300_000      # safe for local CPU

os.makedirs(OUT_DIR, exist_ok=True)

CONF_MAP = {"l": 0.3, "m": 0.6, "h": 0.9, "n": 0.0}

USE_COLS = [
    "latitude", "longitude",
    "brightness", "frp", "bright_t31",
    "confidence", "acq_date", "acq_time"
]

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def get_tile_id(lat, lon):
    lat_tile = int((lat + 90) // TILE_SIZE)
    lon_tile = int((lon + 180) // TILE_SIZE)
    return lat_tile, lon_tile

def latlon_to_cell(lat, lon, lat0, lon0):
    r = int((lat - lat0) // CELL_SIZE)
    c = int((lon - lon0) // CELL_SIZE)
    return r, c

def build_edge_index(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    v = nr * cols + nc
                    edges.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# ======================================================
# PASS 1: FIND ACTIVE TILES
# ======================================================
print("Scanning dataset to find active tiles...")
active_tiles = set()

for chunk in pd.read_csv(
    CSV_PATH,
    usecols=["latitude", "longitude"],
    chunksize=CHUNK_SIZE
):
    for lat, lon in zip(chunk.latitude, chunk.longitude):
        active_tiles.add(get_tile_id(lat, lon))

print(f"Active tiles found: {len(active_tiles)}")

# ======================================================
# PASS 2: PROCESS EACH TILE INDEPENDENTLY
# ======================================================
for tile_id in sorted(active_tiles):
    lat_tile, lon_tile = tile_id
    print(f"\nProcessing tile {tile_id}")

    lat_min = lat_tile * TILE_SIZE - 90
    lat_max = lat_min + TILE_SIZE
    lon_min = lon_tile * TILE_SIZE - 180
    lon_max = lon_min + TILE_SIZE

    rows = int(TILE_SIZE / CELL_SIZE)
    cols = int(TILE_SIZE / CELL_SIZE)
    NUM_NODES = rows * cols

    daily_data = defaultdict(lambda: defaultdict(list))

    # --------------------------------------------------
    # STREAM CSV AND COLLECT DAILY DATA
    # --------------------------------------------------
    for chunk in pd.read_csv(CSV_PATH, usecols=USE_COLS, chunksize=CHUNK_SIZE):

        chunk = chunk[
            (chunk.latitude >= lat_min) & (chunk.latitude < lat_max) &
            (chunk.longitude >= lon_min) & (chunk.longitude < lon_max)
        ]
        if len(chunk) == 0:
            continue

        dt = pd.to_datetime(
            chunk.acq_date + " " +
            chunk.acq_time.astype(str).str.zfill(4),
            format="%Y-%m-%d %H%M"
        )
        chunk["date"] = dt.dt.date

        for _, row in chunk.iterrows():
            r, c = latlon_to_cell(row.latitude, row.longitude, lat_min, lon_min)
            node = r * cols + c

            daily_data[row.date][node].append((
                row.brightness,
                row.frp,
                CONF_MAP.get(row.confidence, 0.0),
                row.bright_t31
            ))

    if len(daily_data) < WINDOW_DAYS + 1:
        print("  Skipping tile (not enough temporal data)")
        continue

    # --------------------------------------------------
    # BUILD SNAPSHOTS (7-DAY SLIDING WINDOW)
    # --------------------------------------------------
    dates = sorted(daily_data.keys())
    edge_index = build_edge_index(rows, cols)
    snapshots = []

    for i in range(WINDOW_DAYS - 1, len(dates) - 1):
        window = dates[i - WINDOW_DAYS + 1:i + 1]
        next_day = dates[i + 1]

        X = torch.zeros((NUM_NODES, 6), dtype=torch.float32)
        fire_count = torch.zeros(NUM_NODES, dtype=torch.float32)

        for d in window:
            for node, vals in daily_data[d].items():
                fire_count[node] += len(vals)
                for b, frp, conf, t31 in vals:
                    X[node, 1] += b
                    X[node, 3] += frp
                    X[node, 4] += conf
                    X[node, 5] += t31
                    X[node, 2] = max(X[node, 2], b)

        mask = fire_count > 0
        X[:, 0] = fire_count
        X[mask, 1] /= fire_count[mask]
        X[mask, 3] /= fire_count[mask]
        X[mask, 4] /= fire_count[mask]
        X[mask, 5] /= fire_count[mask]

        y = torch.zeros(NUM_NODES, dtype=torch.float32)
        if next_day in daily_data:
            for node in daily_data[next_day]:
                if fire_count[node] == 0:
                    y[node] = 1.0

        snapshots.append({
            "X": X,
            "edge_index": edge_index,
            "y": y,
            "date": dates[i]
        })

    # --------------------------------------------------
    # SAVE TILE DATA
    # --------------------------------------------------
    out_path = os.path.join(
        OUT_DIR,
        f"tile_{lat_tile}_{lon_tile}.pt"
    )
    torch.save(snapshots, out_path)
    print(f"  Saved {len(snapshots)} snapshots → {out_path}")

print("\nPreprocessing completed successfully.")

