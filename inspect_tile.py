import torch

path = r"D:\Minor\Code\processed_tiles\tile_0_23.pt"
snapshots = torch.load(path, weights_only=False)

print("Loaded snapshots:", len(snapshots))
s = snapshots[0]

print("Keys:", s.keys())
print("X shape:", s["X"].shape)
print("edge_index shape:", s["edge_index"].shape)
print("y shape:", s["y"].shape)
print("Date:", s["date"])

print("Active nodes:",
      int((s["X"][:,0] > 0).sum()))

print("Next-day ignitions:",
      int(s["y"].sum()))
