import torch
import torch.nn as nn

from dataset_loader import FireTGNN_Dataset
from torch.utils.data import DataLoader
from tgnn_model import FireTGNN

# ======================================================
# LOAD DATA
# ======================================================
DATA_DIR = r"D:\Minor\Code\processed_tiles"

dataset = FireTGNN_Dataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ======================================================
# MODEL SETUP
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FireTGNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ======================================================
# SINGLE TRAINING STEP (TEST RUN)
# ======================================================
X, edge_index, y = next(iter(loader))

X = X.squeeze(0).to(device)
edge_index = edge_index.squeeze(0).to(device)
y = y.squeeze(0).to(device)

model.train()
optimizer.zero_grad()

pred = model(X, edge_index)
loss = criterion(pred, y)

loss.backward()
optimizer.step()

print("Loss:", loss.item())
