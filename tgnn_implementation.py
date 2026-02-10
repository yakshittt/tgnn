# ======================================================
# FINAL TGNN IMPLEMENTATION (KAGGLE-READY, SINGLE BLOCK)
# ======================================================

# -------- INSTALL (RUN ONCE, THEN RESTART KERNEL) -----
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# ---------------- IMPORTS -----------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv

# ---------------- CONFIG ------------------------------
DATA_DIR = "/kaggle/input/processed-tiles"
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET -----------------------------
class FireSnapshotDataset(Dataset):
    def __init__(self, data_dir):
        self.snapshots = []

        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".pt"):
                path = os.path.join(data_dir, fname)
                tile_data = torch.load(path, weights_only=False)

                # snapshots inside each tile are time-ordered
                for s in tile_data:
                    self.snapshots.append(
                        (s["X"], s["edge_index"], s["y"])
                    )

        print(f"Total snapshots loaded: {len(self.snapshots)}")

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        return self.snapshots[idx]

# ---------------- TGNN MODEL --------------------------
class TGNN(nn.Module):
    def __init__(
        self,
        node_features=6,
        hidden_dim=64,
        gru_hidden_dim=128,
        num_gnn_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # IMPORTANT: batch_first = False
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=False
        )

        self.fc1 = nn.Linear(gru_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, hidden_state=None):
        num_nodes = x.size(0)

        # Input projection
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        # Spatial GCN
        for conv, bn in zip(self.convs, self.bns):
            res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + res

        # Temporal GRU (1 time step)
        x = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]

        if hidden_state is None:
            hidden_state = torch.zeros(
                1, num_nodes, self.gru_hidden_dim,
                device=x.device
            )

        out, hidden_state = self.gru(x, hidden_state)
        x = out.squeeze(0)

        # Prediction head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        logits = self.fc3(x)
        probs = torch.sigmoid(logits).squeeze(1)

        return probs, hidden_state

# ---------------- LOAD DATA ---------------------------
dataset = FireSnapshotDataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False  # CRITICAL for temporal learning
)

# ---------------- TRAINING SETUP ----------------------
model = TGNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN LOOP --------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    hidden_state = None  # reset at epoch start

    for X, edge_index, y in loader:
        X = X.squeeze(0).to(DEVICE)
        edge_index = edge_index.squeeze(0).to(DEVICE)
        y = y.squeeze(0).to(DEVICE)

        optimizer.zero_grad()

        preds, hidden_state = model(X, edge_index, hidden_state)

        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Detach hidden state to prevent backprop through full history
        hidden_state = hidden_state.detach()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")
