import os
import torch
from torch.utils.data import Dataset, DataLoader

# ======================================================
# CONFIGURATION (LOCAL SYSTEM)
# ======================================================
DATA_DIR = r"D:\Minor\Code\processed_tiles"

# ======================================================
# DATASET CLASS
# ======================================================
class FireTGNN_Dataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []

        print("Scanning processed tile files...")

        for fname in os.listdir(data_dir):
            if not fname.endswith(".pt"):
                continue

            path = os.path.join(data_dir, fname)
            print(f"Loading {fname}")

            # weights_only=False because snapshots contain datetime objects
            snapshots = torch.load(path, weights_only=False)

            for s in snapshots:
                self.samples.append((
                    s["X"],           # node features
                    s["edge_index"],  # graph structure
                    s["y"]            # labels
                ))

        print(f"\nTotal snapshots loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ======================================================
# DATALOADER
# ======================================================
dataset = FireTGNN_Dataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=1,     # one graph snapshot at a time
    shuffle=True
)

# ======================================================
# SANITY CHECK
# ======================================================
if __name__ == "__main__":
    X, edge_index, y = next(iter(loader))

    print("\nSanity check:")
    print("X shape:", X.shape)               # [1, num_nodes, 6]
    print("edge_index shape:", edge_index.shape)  # [1, 2, E]
    print("y shape:", y.shape)               # [1, num_nodes]
